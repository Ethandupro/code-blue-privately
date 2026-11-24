"""Read EC2-Bench run folders and produce the Code Blue outputs (claim.json, curves, mosaic).

EC2-Bench writes <output-dir>/<batch timestamp>/repeat<N>/<task id>/ with score.json (overall_score),
cloud_assistant_audit.jsonl (every cloud request, logged before sending) and, under our patch,
codeblue_p3_table.json. Each arm is one run folder given as PACKET:ATTACK:PATH, e.g.
    local-only:none:out/edge  P0:none:out/p0  P3:none:out/p3  P0:stuck_and_stuff:out/p0-att  ...
In delegation mode every escalate() is one cloud call that performs one step, so rescue n = cloud call n.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from codeblue.attacks import get_attack, injection_present
from codeblue.claim import evaluate_all
from codeblue.curves import curve_by_rescue, rescue_tables
from codeblue.ec2bench.inject import attacked_tasks
from codeblue.ec2bench.labels import load_task_room
from codeblue.mosaic import mosaic_episode, mosaic_table
from codeblue.scan import scan_episode


def audit_texts(path: Path) -> list[str]:
    """One text per cloud request, in send order (as EC2-Bench's privacy audit reads them)."""
    texts = []
    if not path.exists():
        return texts
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not isinstance(e, dict) or e.get("event") != "request" or not e.get("input"):
            continue
        parts = []
        for m in e["input"]:
            c = m.get("content")
            if isinstance(c, str):
                parts.append(c)
            elif isinstance(c, list):
                parts += [p["text"] for p in c if isinstance(p, dict) and isinstance(p.get("text"), str)]
            if m.get("tool_calls"):
                parts.append(json.dumps(m["tool_calls"], ensure_ascii=False))
        texts.append("\n".join(parts))
    return texts


def _score(task_dir: Path) -> float | None:
    p = task_dir / "score.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text()).get("overall_score")
    if not isinstance(s, (int, float)):
        return None
    return s / 100 if s > 1 else float(s)


def task_dirs(run_dir: Path):
    """Yield (repeat, task_id, folder) for every task folder that has a score."""
    for score in sorted(Path(run_dir).glob("**/repeat*/*/score.json")):
        d = score.parent
        yield int(d.parent.name.removeprefix("repeat") or 0), d.name, d


def collect(arms: list[tuple[str, str, Path]], data_root: str | Path, labels: dict, out: str | Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rooms: dict[str, tuple] = {}
    hit = set(attacked_tasks(labels))
    episodes, leaks, skipped = [], {}, []
    for packet, attack, run_dir in arms:
        att = get_attack(attack)
        for rep, task_id, d in task_dirs(run_dir):
            if attack != "none" and task_id not in hit:
                continue
            if task_id not in rooms:
                try:
                    rooms[task_id] = load_task_room(data_root, task_id, labels)
                except FileNotFoundError as e:
                    skipped.append(f"{task_id}: {e}")
                    rooms[task_id] = None
            if rooms[task_id] is None or not rooms[task_id][1]["annotated"]:
                continue
            room = rooms[task_id][0]
            texts = [] if packet == "local-only" else audit_texts(d / "cloud_assistant_audit.jsonl")
            ep = f"{task_id}__{packet}__{attack}__r{rep}"
            lk = scan_episode(room, task_id, texts)
            lk["n_secrets_total"] = len(room.secrets) or 1  # as in the data-room runner: composites count too
            calls = list(range(len(texts)))
            lk["curve_by_rescue"] = curve_by_rescue(lk, calls)
            tp = d / "codeblue_p3_table.json"
            table = json.loads(tp.read_text()).get("placeholder_table", {}) if tp.exists() else {}
            lk["mosaic"] = mosaic_episode(room, texts, lk, table, calls)
            lk["injection_to_cloud"] = any(injection_present(att, t) for t in texts)
            leaks[ep] = lk
            score = _score(d)
            episodes.append({"episode_id": ep, "task_id": task_id, "room": task_id, "seed": rep,
                             "packet": None if packet == "local-only" else packet,
                             "trigger": "never" if packet == "local-only" else "escalate",
                             "attack": attack, "score": 0.0 if score is None else score, "score_missing": score is None,
                             "n_rescues": len(texts), "n_blocked_firings": 0})
    (out / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (out / "leakage.json").write_text(json.dumps(leaks, indent=2))
    claim = evaluate_all(episodes, leaks, {"trigger": "escalate"})
    curve_rows, count_rows, slopes = rescue_tables(episodes, leaks)
    mosaic_rows = mosaic_table(episodes, leaks)
    claim["secondary"] = {"secret_share_per_extra_rescue": slopes, "mosaic": mosaic_rows}
    claim["coverage"] = {"episodes": len(episodes), "tasks": len({e["task_id"] for e in episodes}),
                         "missing_scores": sum(e["score_missing"] for e in episodes), "skipped": skipped,
                         "unscannable_values": {t: len(r[1]["unscannable"]) for t, r in rooms.items() if r}}
    (out / "claim.json").write_text(json.dumps(claim, indent=2))
    for name, rows in (("rescue_curve.csv", curve_rows), ("leak_vs_rescues.csv", count_rows), ("mosaic.csv", mosaic_rows)):
        if rows:
            with (out / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return claim
