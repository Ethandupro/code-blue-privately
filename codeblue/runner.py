"""Run an experiment grid, scan the cloud log, write a summary, and test the claim (claim.json)."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from codeblue.attacks import apply_attack, exposure_stats, get_attack
from codeblue.claim import evaluate_all, print_claim
from codeblue.config import make_model
from codeblue.endpoints import check_local_section
from codeblue.dataroom import load_data_room
from codeblue.detect import make_detector
from codeblue.gateway import CloudGateway, read_cloud_requests, request_text
from codeblue.llm import LoggedModel
from codeblue.loop import LoopConfig, run_episode
from codeblue.packets import make_builder
from codeblue.scan import scan_log
from codeblue.triggers import make_trigger


def conditions(exp: dict) -> list[tuple[str | None, str]]:
    conds: list[tuple[str | None, str]] = [(p, t) for t in exp.get("triggers", ["selfcheck"])
                                          for p in exp.get("packets", ["P0"])]
    if exp.get("include_local_only", True):
        conds.insert(0, (None, "never"))
    if exp.get("include_cloud_only", False):
        conds.append(("P0", "always"))
    return conds


def _cap(v) -> int | None:
    """max_rescues from config: absent, negative or "none" means no cap."""
    if v is None or (isinstance(v, str) and v.lower() == "none") or (isinstance(v, int) and v < 0):
        return None
    return int(v)


def run_experiment(cfg: dict, out_dir: str | Path | None = None, quiet: bool = False) -> Path:
    check_local_section(cfg.get("local", {}), cfg.get("cloud"))
    exp = cfg.get("experiment", {})
    base = Path(cfg.get("_dir", "."))
    room_path = Path(exp.get("data_room", "data_rooms/harbourline"))
    if not room_path.is_absolute():
        room_path = (base / room_path) if (base / room_path).exists() else room_path
    room = load_data_room(room_path)
    task_ids = exp.get("tasks", ["all"])
    tasks = list(room.tasks.values()) if task_ids == ["all"] else [room.tasks[t] for t in task_ids]

    out = Path(out_dir or Path(exp.get("out_dir", "runs")) / time.strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({k: v for k, v in cfg.items() if k != "_dir"}, indent=2))

    local = LoggedModel(make_model(cfg.get("local", {"kind": "mock-local"}), tasks), out / "local_log.jsonl")
    gateway = CloudGateway(make_model(cfg.get("cloud", {"kind": "mock-cloud"}), tasks), out / "cloud_log.jsonl")
    detector = make_detector(exp.get("detector", "regex"), room, local)
    loop_cfg = LoopConfig(
        takeover_steps=exp.get("takeover_steps", 2),
        max_rescues=_cap(exp.get("max_rescues")),
        cooldown=exp.get("cooldown", 0),
        max_steps=exp.get("max_steps"),
        max_obs_chars=exp.get("max_obs_chars", 2000),
    )

    # "none" is the clean setting; each attack gets its own copy of the room with the injection planted.
    placement = exp.get("attack_placement", "vectors")
    arms = []
    for name in exp.get("attacks", ["none"]):
        attack = get_attack(name)
        arms.append((name, attack, *apply_attack(room, attack, placement)))

    episodes = []
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for seed in exp.get("seeds", [0]):
            for attack_name, attack, arm_room, injected in arms:
                for packet, trig in conditions(exp):
                    for task in tasks:
                        ep_id = f"{task.id}__{packet or 'none'}__{trig}__{attack_name}__s{seed}"
                        local.context = {"episode_id": ep_id}
                        res = run_episode(ep_id, arm_room, arm_room.tasks[task.id], local, gateway,
                                          make_builder(packet) if packet else None,
                                          make_trigger(trig, exp, seed), detector, loop_cfg)
                        res.update(seed=seed, room=room.name, attack=attack_name, injected_files=injected)
                        episodes.append(res)
                        f.write(json.dumps(res) + "\n")
                        if not quiet:
                            extra = f" blocked={res['n_blocked_firings']}" if res["n_blocked_firings"] else ""
                            print(f"{ep_id:68s} score={res['score']:.2f} rescues={res['n_rescues']}{extra}")

    summarise(room, out, episodes, quiet=quiet)
    claim = evaluate_all(episodes, json.loads((out / "leakage.json").read_text()), exp.get("claim"))
    (out / "claim.json").write_text(json.dumps(claim, indent=2))
    if not quiet:
        print_claim(claim)
    return out


def summarise(room, out: Path, episodes: list[dict], quiet: bool = False) -> list[dict]:
    log = out / "cloud_log.jsonl"
    ep_tasks = {e["episode_id"]: e["task_id"] for e in episodes}
    leaks = scan_log(room, log, ep_tasks) if log.exists() else {}
    for ep in ep_tasks:
        leaks.setdefault(ep, {"n_calls": 0, "chars_sent": 0, "n_essential_leaked": 0, "n_bystander_leaked": 0,
                              "leaked": {}, "cumulative_curve": []})
    for v in leaks.values():
        v["n_secrets_total"] = len(room.secrets)

    texts: dict[str, list[str]] = defaultdict(list)
    if log.exists():
        for rec in read_cloud_requests(log):
            texts[rec.get("episode_id", "unknown")].append(request_text(rec))
    for e in episodes:
        e.setdefault("attack", "none")
        e["exposure"] = exposure_stats(get_attack(e["attack"]), e["steps"], texts.get(e["episode_id"], []))
        leaks[e["episode_id"]]["injection_to_cloud"] = e["exposure"]["n_cloud_calls_with_injection"] > 0
    (out / "leakage.json").write_text(json.dumps(leaks, indent=2))

    groups = defaultdict(list)
    for e in episodes:
        groups[(e["attack"], e["packet"] or "local-only", e["trigger"])].append(e)
    rows = []
    for (attack, packet, trig), eps in groups.items():
        ls = [leaks[e["episode_id"]] for e in eps]
        rows.append({
            "attack": attack, "packet": packet, "trigger": trig, "episodes": len(eps),
            "mean_score": round(mean(e["score"] for e in eps), 3),
            "success_rate": round(mean(e["score"] == 1.0 for e in eps), 3),
            "mean_rescues": round(mean(e["n_rescues"] for e in eps), 2),
            "mean_blocked_firings": round(mean(e.get("n_blocked_firings", 0) for e in eps), 2),
            "mean_cloud_calls": round(mean(l["n_calls"] for l in ls), 2),
            "mean_chars_sent": round(mean(l["chars_sent"] for l in ls), 1),
            "mean_essential_leaked": round(mean(l["n_essential_leaked"] for l in ls), 2),
            "mean_bystander_leaked": round(mean(l["n_bystander_leaked"] for l in ls), 2),
            "mean_secrets_leaked": round(mean(l["n_essential_leaked"] + l["n_bystander_leaked"] for l in ls), 2),
            "injection_seen_rate": round(mean(e["exposure"]["injection_seen_by_agent"] for e in eps), 3),
            "injection_to_cloud_rate": round(mean(e["exposure"]["n_cloud_calls_with_injection"] > 0 for e in eps), 3),
        })
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if not quiet:
        print_table(rows)
        print(f"\nwrote {out}/summary.csv, leakage.json, episodes.jsonl, cloud_log.jsonl, local_log.jsonl")
    return rows


def print_table(rows: list[dict]) -> None:
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("\n" + "  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
