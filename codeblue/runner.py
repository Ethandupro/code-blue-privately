"""Run an experiment grid, then scan the cloud log and write a summary."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from codeblue.config import make_model
from codeblue.endpoints import check_local_section
from codeblue.dataroom import load_data_room
from codeblue.detect import make_detector
from codeblue.gateway import CloudGateway
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
        max_rescues=exp.get("max_rescues", 3),
        cooldown=exp.get("cooldown", 2),
        max_steps=exp.get("max_steps"),
        max_obs_chars=exp.get("max_obs_chars", 2000),
    )

    episodes = []
    with (out / "episodes.jsonl").open("w", encoding="utf-8") as f:
        for seed in exp.get("seeds", [0]):
            for packet, trig in conditions(exp):
                for task in tasks:
                    ep_id = f"{task.id}__{packet or 'none'}__{trig}__s{seed}"
                    local.context = {"episode_id": ep_id}
                    res = run_episode(ep_id, room, task, local, gateway,
                                      make_builder(packet) if packet else None,
                                      make_trigger(trig, exp, seed), detector, loop_cfg)
                    res["seed"] = seed
                    episodes.append(res)
                    f.write(json.dumps(res) + "\n")
                    if not quiet:
                        print(f"{ep_id:55s} score={res['score']:.2f} rescues={res['n_rescues']}")

    summarise(room, out, episodes, quiet=quiet)
    return out


def summarise(room, out: Path, episodes: list[dict], quiet: bool = False) -> list[dict]:
    log = out / "cloud_log.jsonl"
    ep_tasks = {e["episode_id"]: e["task_id"] for e in episodes}
    leaks = scan_log(room, log, ep_tasks) if log.exists() else {}
    for ep in ep_tasks:
        leaks.setdefault(ep, {"n_calls": 0, "chars_sent": 0, "n_essential_leaked": 0, "n_bystander_leaked": 0,
                              "leaked": {}, "cumulative_curve": []})
    (out / "leakage.json").write_text(json.dumps(leaks, indent=2))

    groups = defaultdict(list)
    for e in episodes:
        groups[(e["packet"] or "local-only", e["trigger"])].append(e)
    rows = []
    for (packet, trig), eps in groups.items():
        ls = [leaks[e["episode_id"]] for e in eps]
        rows.append({
            "packet": packet, "trigger": trig, "episodes": len(eps),
            "mean_score": round(mean(e["score"] for e in eps), 3),
            "success_rate": round(mean(e["score"] == 1.0 for e in eps), 3),
            "mean_rescues": round(mean(e["n_rescues"] for e in eps), 2),
            "mean_cloud_calls": round(mean(l["n_calls"] for l in ls), 2),
            "mean_chars_sent": round(mean(l["chars_sent"] for l in ls), 1),
            "mean_essential_leaked": round(mean(l["n_essential_leaked"] for l in ls), 2),
            "mean_bystander_leaked": round(mean(l["n_bystander_leaked"] for l in ls), 2),
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


def matched_rate(episodes_path: str | Path, trigger: str = "selfcheck") -> float:
    """Rescues per decision point in the given trigger's episodes: the rate to give the random control."""
    rescues = points = 0
    for line in Path(episodes_path).read_text().splitlines():
        e = json.loads(line)
        if e["trigger"] == trigger:
            rescues += e["n_rescues"]
            points += e["decision_points"]
    return rescues / points if points else 0.0
