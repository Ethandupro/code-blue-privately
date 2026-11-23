"""Run an experiment grid, then scan the cloud log and write a summary."""

from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from codeblue.attacks import apply_attack, exposure_stats, get_attack
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

    # Attacks (codeblue/attacks.py): "none" is the clean arm; each attack gets its own copy of the room.
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
                        ep_id = f"{task.id}__{packet or 'none'}__{trig}__s{seed}"
                        if attack is not None:
                            ep_id = f"{task.id}__{packet or 'none'}__{trig}__{attack_name}__s{seed}"
                        local.context = {"episode_id": ep_id}
                        res = run_episode(ep_id, arm_room, task, local, gateway,
                                          make_builder(packet) if packet else None,
                                          make_trigger(trig, exp, seed), detector, loop_cfg)
                        res["seed"] = seed
                        res["attack"] = attack_name
                        res["attack_placement"] = placement if attack else None
                        res["injected_files"] = injected
                        episodes.append(res)
                        f.write(json.dumps(res) + "\n")
                        if not quiet:
                            print(f"{ep_id:70s} score={res['score']:.2f} rescues={res['n_rescues']}")

    summarise(room, out, episodes, quiet=quiet)
    return out


def _cloud_texts_by_episode(log: Path) -> dict[str, list[str]]:
    by_ep: dict[str, list[str]] = defaultdict(list)
    if log.exists():
        for rec in read_cloud_requests(log):
            by_ep[rec.get("episode_id", "unknown")].append(request_text(rec))
    return by_ep


def summarise(room, out: Path, episodes: list[dict], quiet: bool = False) -> list[dict]:
    log = out / "cloud_log.jsonl"
    ep_tasks = {e["episode_id"]: e["task_id"] for e in episodes}
    leaks = scan_log(room, log, ep_tasks) if log.exists() else {}
    for ep in ep_tasks:
        leaks.setdefault(ep, {"n_calls": 0, "chars_sent": 0, "n_essential_leaked": 0, "n_bystander_leaked": 0,
                              "leaked": {}, "cumulative_curve": []})
    (out / "leakage.json").write_text(json.dumps(leaks, indent=2))

    texts = _cloud_texts_by_episode(log)
    for e in episodes:
        e.setdefault("attack", "none")
        e["exposure"] = exposure_stats(get_attack(e["attack"]), e["steps"], texts.get(e["episode_id"], []))

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
            "mean_cloud_calls": round(mean(l["n_calls"] for l in ls), 2),
            "mean_chars_sent": round(mean(l["chars_sent"] for l in ls), 1),
            "mean_essential_leaked": round(mean(l["n_essential_leaked"] for l in ls), 2),
            "mean_bystander_leaked": round(mean(l["n_bystander_leaked"] for l in ls), 2),
            "escalated_rate": round(mean(e["n_rescues"] > 0 for e in eps), 3),
            "escalation_rate": round(_ratio(sum(e["n_rescues"] for e in eps), sum(e["decision_points"] for e in eps)), 3),
            "injection_seen_rate": round(mean(e["exposure"]["injection_seen_by_agent"] for e in eps), 3),
            "injection_to_cloud_rate": round(mean(e["exposure"]["n_cloud_calls_with_injection"] > 0 for e in eps), 3),
        })
    _write_csv(out / "summary.csv", rows)
    attack_rows = attack_deltas(episodes, leaks)
    if attack_rows:
        _write_csv(out / "attack_summary.csv", attack_rows)
    if not quiet:
        print_table(rows)
        if attack_rows:
            print("\nattack effect, paired with the clean run of the same task, packet, trigger and seed:")
            print_table(attack_rows)
        print(f"\nwrote {out}/summary.csv, leakage.json, episodes.jsonl, cloud_log.jsonl, local_log.jsonl"
              + (", attack_summary.csv" if attack_rows else ""))
    return rows


def _ratio(a: float, b: float) -> float:
    return a / b if b else 0.0


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def attack_deltas(episodes: list[dict], leaks: dict[str, dict]) -> list[dict]:
    """Paired attack effects. Each attacked episode is compared with the clean ("none") episode that has the same
    task, packet, trigger and seed; the rows average those paired differences per (attack, packet, trigger).

    forced_rate           share of pairs where the clean run never escalated but the attacked run did
    earlier_rate          share of pairs where both escalated but the attacked run's first rescue came sooner
    rescue_inflation      total rescues attacked / clean (inf if the clean runs never escalated)
    d_cloud_calls         extra cloud calls per episode
    d_bystander_leaked    extra bystander secrets per episode (the main harm); can be negative when an
                          earlier escalation sends a shorter history
    d_bystander_forced    the same, averaged over forced pairs only (the escalations that should not have happened)
    d_essential_leaked    change in essential secrets per episode
    d_score               change in task score (an attack can also break the task)
    """
    key = lambda e: (e["task_id"], e["packet"], e["trigger"], e.get("seed", 0))  # noqa: E731
    clean = {key(e): e for e in episodes if e.get("attack", "none") == "none"}
    groups = defaultdict(list)
    for e in episodes:
        if e.get("attack", "none") != "none" and key(e) in clean:
            groups[(e["attack"], e["packet"] or "local-only", e["trigger"])].append((clean[key(e)], e))

    def first(e):
        return e["rescues"][0]["at_step"] if e["rescues"] else None

    out = []
    for (attack, packet, trig), pairs in groups.items():
        lc = lambda e, k: leaks.get(e["episode_id"], {}).get(k, 0)  # noqa: E731
        forced = [(c, a) for c, a in pairs if c["n_rescues"] == 0 and a["n_rescues"] > 0]
        earlier = [(c, a) for c, a in pairs if first(c) is not None and first(a) is not None and first(a) < first(c)]
        rc, ra = sum(c["n_rescues"] for c, _ in pairs), sum(a["n_rescues"] for _, a in pairs)
        out.append({
            "attack": attack, "packet": packet, "trigger": trig, "pairs": len(pairs),
            "forced_rate": round(len(forced) / len(pairs), 3),
            "earlier_rate": round(len(earlier) / len(pairs), 3),
            "rescue_inflation": round(ra / rc, 2) if rc else (float("inf") if ra else 1.0),
            "d_cloud_calls": round(mean(lc(a, "n_calls") - lc(c, "n_calls") for c, a in pairs), 2),
            "d_bystander_leaked": round(mean(lc(a, "n_bystander_leaked") - lc(c, "n_bystander_leaked")
                                             for c, a in pairs), 2),
            "d_bystander_forced": round(mean(lc(a, "n_bystander_leaked") - lc(c, "n_bystander_leaked")
                                             for c, a in forced), 2) if forced else 0.0,
            "d_essential_leaked": round(mean(lc(a, "n_essential_leaked") - lc(c, "n_essential_leaked")
                                             for c, a in pairs), 2),
            "d_score": round(mean(a["score"] - c["score"] for c, a in pairs), 3),
            "injection_seen_rate": round(mean(a["exposure"]["injection_seen_by_agent"] for _, a in pairs), 3),
            "injection_to_cloud_rate": round(mean(a["exposure"]["n_cloud_calls_with_injection"] > 0
                                                  for _, a in pairs), 3),
        })
    return out


def print_table(rows: list[dict]) -> None:
    cols = list(rows[0])
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("\n" + "  ".join(c.ljust(widths[c]) for c in cols))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def matched_rate(episodes_path: str | Path, trigger: str = "selfcheck") -> float:
    """Rescues per decision point in the given trigger's clean (no-attack) episodes: the rate for the random control."""
    rescues = points = 0
    for line in Path(episodes_path).read_text().splitlines():
        e = json.loads(line)
        if e["trigger"] == trigger and e.get("attack", "none") == "none":
            rescues += e["n_rescues"]
            points += e["decision_points"]
    return rescues / points if points else 0.0
