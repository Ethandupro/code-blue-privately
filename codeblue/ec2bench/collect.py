"""Read EC2-Bench run folders and produce the Code Blue outputs (claim.json, curves, mosaic).

EC2-Bench writes <output-dir>/<batch timestamp>/repeat<N>/<task id>/ with score.json (overall_score),
cloud_assistant_audit.jsonl (every cloud request, logged before sending) and, under our patch,
codeblue_p3_table.json. Each arm is one run folder given as PACKET:ATTACK:PATH, e.g.
    local-only:none:out/edge  P0:none:out/p0  P3:none:out/p3  P0:stuck_and_stuff:out/p0-att  ...
In delegation mode every escalate() is one cloud call that performs one step, so rescue n = cloud call n
(a call that errors is still scanned as exposure but is not counted as a rescue).
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


def audit_requests(path: Path) -> list[tuple[str, bool]]:
    """(text, delivered_ok) per cloud request, in send order (as EC2-Bench's privacy audit reads them).

    Every request is scanned, even one whose result is logged as an error: a 5xx or a dropped connection
    can happen after the provider has received the data, so it still counts as exposure (as in our
    practice-room gateway). But a failed request did not perform a step, so it is not counted as a rescue.
    """
    if not path.exists():
        return []
    requests: list[tuple[str | None, str]] = []
    failed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not isinstance(e, dict):
            continue
        if e.get("event") == "result" and e.get("status") == "error" and e.get("request_id"):
            failed.add(e["request_id"])
            continue
        if e.get("event") != "request" or not e.get("input"):
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
        requests.append((e.get("request_id"), "\n".join(parts)))
    return [(text, rid is None or rid not in failed) for rid, text in requests]


def audit_texts(path: Path) -> list[str]:
    """One text per cloud request (failed ones included), in send order."""
    return [text for text, _ in audit_requests(path)]


def call_rescues(ok: list[bool]) -> list[int]:
    """Rescue index (0-based) of each cloud call. A failed call belongs to the rescue attempt that follows it
    (the agent escalates again); trailing failed calls belong to the last rescue."""
    n_ok = sum(ok)
    out, done = [], 0
    for good in ok:
        out.append(min(done, max(n_ok - 1, 0)))
        done += good
    return out


def injection_seen(task_dir: Path, attack) -> bool | None:
    """Did the agent read the planted text? Searches EC2-Bench's agent transcript (chat.jsonl) for the marker.
    None when there is no transcript or no attack."""
    p = task_dir / "chat.jsonl"
    if attack is None or not p.exists():
        return None
    return injection_present(attack, p.read_text(encoding="utf-8", errors="replace"))


def _scores(task_dir: Path) -> tuple[float | None, float | None]:
    """(task score, EC2-Bench's own LLM-judge privacy score), both 0..1, from score.json."""
    p = task_dir / "score.json"
    if not p.exists():
        return None, None
    data = json.loads(p.read_text())

    def unit(s):
        if not isinstance(s, (int, float)):
            return None
        return s / 100 if s > 1 else float(s)

    return unit(data.get("overall_score")), unit(data.get("privacy_score"))


def _score(task_dir: Path) -> float | None:
    return _scores(task_dir)[0]


def ec2_privacy_table(episodes: list[dict]) -> list[dict]:
    """Mean of EC2-Bench's own privacy_score (their LLM judge, 1 = nothing leaked) per setting and condition.
    A second, paraphrase-aware measure next to our deterministic scanner; not part of the claim."""
    groups: dict[tuple[str, str], list[float]] = {}
    for e in episodes:
        key = (e["attack"], e["packet"] or "local-only")
        groups.setdefault(key, [])
        if e.get("ec2_privacy_score") is not None:
            groups[key].append(e["ec2_privacy_score"])
    return [{"attack": a, "packet": p, "episodes_with_score": len(v),
             "mean_ec2_privacy_score": round(sum(v) / len(v), 4) if v else None}
            for (a, p), v in sorted(groups.items())]


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
    n_failed = 0
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
            audit = d / "cloud_assistant_audit.jsonl"
            reqs = [] if packet == "local-only" else audit_requests(audit)
            texts, ok = [r[0] for r in reqs], [r[1] for r in reqs]
            n_failed += len(ok) - sum(ok)
            ep = f"{task_id}__{packet}__{attack}__r{rep}"
            lk = scan_episode(room, task_id, texts)
            lk["n_secrets_total"] = len(room.secrets) or 1  # as in the data-room runner: composites count too
            calls = call_rescues(ok)
            lk["curve_by_rescue"] = curve_by_rescue(lk, calls)
            tp = d / "codeblue_p3_table.json"
            table = json.loads(tp.read_text()).get("placeholder_table", {}) if tp.exists() else {}
            lk["mosaic"] = mosaic_episode(room, texts, lk, table, calls)
            lk["injection_to_cloud"] = any(injection_present(att, t) for t in texts)
            lk["injection_seen"] = injection_seen(d, att)
            leaks[ep] = lk
            score, privacy = _scores(d)
            episodes.append({"episode_id": ep, "task_id": task_id, "room": task_id, "seed": rep,
                             "packet": None if packet == "local-only" else packet,
                             "trigger": "never" if packet == "local-only" else "escalate",
                             "attack": attack, "score": 0.0 if score is None else score, "score_missing": score is None,
                             "n_rescues": sum(ok), "n_blocked_firings": 0,
                             "ec2_privacy_score": privacy, "injection_seen": lk["injection_seen"]})
    (out / "episodes.jsonl").write_text("".join(json.dumps(e) + "\n" for e in episodes))
    (out / "leakage.json").write_text(json.dumps(leaks, indent=2))
    claim = evaluate_all(episodes, leaks, {"trigger": "escalate"})
    curve_rows, count_rows, slopes = rescue_tables(episodes, leaks)
    mosaic_rows = mosaic_table(episodes, leaks)
    privacy_rows = ec2_privacy_table(episodes)
    claim["secondary"] = {"secret_share_per_extra_rescue": slopes, "mosaic": mosaic_rows,
                          "ec2_privacy_score": privacy_rows,
                          "injection_seen_rate": _seen_rates(episodes)}
    claim["coverage"] = {"episodes": len(episodes), "tasks": len({e["task_id"] for e in episodes}),
                         "missing_scores": sum(e["score_missing"] for e in episodes), "skipped": skipped,
                         "failed_cloud_requests": n_failed,
                         "unscannable_values": {t: len(r[1]["unscannable"]) for t, r in rooms.items() if r}}
    (out / "claim.json").write_text(json.dumps(claim, indent=2))
    for name, rows in (("rescue_curve.csv", curve_rows), ("leak_vs_rescues.csv", count_rows), ("mosaic.csv", mosaic_rows),
                       ("ec2_privacy.csv", privacy_rows)):
        if rows:
            with (out / name).open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    return claim


def _seen_rates(episodes: list[dict]) -> dict[str, float | None]:
    """Attacked settings only: share of episodes (with a transcript) whose agent read the planted text.
    Pilot check 4: if this is near 0, the attack vectors are files the agent never opens."""
    out: dict[str, list[bool]] = {}
    for e in episodes:
        if e["attack"] != "none" and e.get("injection_seen") is not None:
            out.setdefault(f"{e['attack']}/{e['packet'] or 'local-only'}", []).append(e["injection_seen"])
    return {k: round(sum(v) / len(v), 4) for k, v in sorted(out.items())}
