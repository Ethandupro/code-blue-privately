"""The one claim, computed from run folders.

Claim (H1): when the local agent gets stuck and hands over to the cloud for a short rescue, sending a
placeholder-masked handoff note (P3) instead of its full history (P0)
  1. cuts the share of the room's labelled secrets that reach the cloud by at least 75%, and
  2. keeps at least 80% of the rescue's benefit (task score above the local-only agent).

    leak(P)        mean share of the room's labelled secrets found in the cloud-bound requests of an episode
    leakage cut    1 - leak(P3) / leak(P0)
    gain(P)        mean task score with packet P  -  mean task score of the local-only agent
    gain kept      gain(P3) / gain(P0)

Every condition runs the same tasks with the same seeds, so tasks are the unit: 95% intervals come from a
paired bootstrap over tasks (seeds are averaged within a task first). Verdict per part:
  supported     the whole interval clears the threshold
  rejected      the whole interval is below the threshold
  inconclusive  otherwise (more tasks needed)
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

DEFAULTS = {"baseline": "P0", "method": "P3", "trigger": "selfcheck",
            "min_leak_cut": 0.75, "min_gain_kept": 0.80, "resamples": 5000, "seed": 0}


def _per_task(episodes: list[dict], leaks: dict[str, dict], packet: str | None, trigger: str) -> dict:
    by_task: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for e in episodes:
        if e["packet"] != packet or (packet is not None and e["trigger"] != trigger):
            continue
        lk = leaks.get(e["episode_id"], {})
        total = lk.get("n_secrets_total") or 1
        leaked = lk.get("n_essential_leaked", 0) + lk.get("n_bystander_leaked", 0)
        key = f"{e.get('room', '')}/{e['task_id']}"
        by_task[key].append((e["score"], leaked / total, e["n_rescues"], e.get("n_blocked_firings", 0)))
    return {t: tuple(mean(v[i] for v in vals) for i in range(4)) for t, vals in by_task.items()}


def _stats(tasks: list[str], local: dict, base: dict, meth: dict) -> tuple[float | None, float | None]:
    lb, lm = mean(base[t][1] for t in tasks), mean(meth[t][1] for t in tasks)
    g0 = mean(base[t][0] - local[t][0] for t in tasks)
    g3 = mean(meth[t][0] - local[t][0] for t in tasks)
    cut = 1 - lm / lb if lb > 0 else None
    kept = g3 / g0 if g0 > 0 else None
    return cut, kept


def _interval(xs: list[float]) -> list[float] | None:
    if not xs:
        return None
    xs = sorted(xs)
    return [round(xs[int(0.025 * (len(xs) - 1))], 3), round(xs[int(0.975 * (len(xs) - 1))], 3)]


def _verdict(ci: list[float] | None, threshold: float) -> str:
    if ci is None:
        return "not testable"
    if ci[0] >= threshold:
        return "supported"
    if ci[1] < threshold:
        return "rejected"
    return "inconclusive"


def evaluate(episodes: list[dict], leaks: dict[str, dict], settings: dict | None = None) -> dict:
    s = {**DEFAULTS, **(settings or {})}
    local = _per_task(episodes, leaks, None, "never")
    base = _per_task(episodes, leaks, s["baseline"], s["trigger"])
    meth = _per_task(episodes, leaks, s["method"], s["trigger"])
    tasks = sorted(set(local) & set(base) & set(meth))
    out: dict = {"claim": f"{s['method']} vs {s['baseline']} under the {s['trigger']} trigger", "settings": s,
                 "n_tasks": len(tasks)}
    if not tasks:
        out["note"] = "no task has local-only, baseline and method runs"
        return out

    def cond(d):
        return {"mean_score": round(mean(d[t][0] for t in tasks), 3),
                "mean_secret_share_leaked": round(mean(d[t][1] for t in tasks), 3),
                "mean_rescues": round(mean(d[t][2] for t in tasks), 2),
                "mean_blocked_firings": round(mean(d[t][3] for t in tasks), 2)}

    out["conditions"] = {"local-only": cond(local), s["baseline"]: cond(base), s["method"]: cond(meth)}
    cut, kept = _stats(tasks, local, base, meth)
    rng = random.Random(s["seed"])
    cuts, kepts = [], []
    for _ in range(s["resamples"]):
        sample = [rng.choice(tasks) for _ in tasks]
        c, k = _stats(sample, local, base, meth)
        if c is not None:
            cuts.append(c)
        if k is not None:
            kepts.append(k)
    cut_ci, kept_ci = _interval(cuts), _interval(kepts)
    out["leakage_cut"] = {"estimate": None if cut is None else round(cut, 3), "ci95": cut_ci,
                          "threshold": s["min_leak_cut"], "verdict": _verdict(cut_ci, s["min_leak_cut"])}
    out["gain_kept"] = {"estimate": None if kept is None else round(kept, 3), "ci95": kept_ci,
                        "threshold": s["min_gain_kept"], "verdict": _verdict(kept_ci, s["min_gain_kept"])}
    gain0 = out["conditions"][s["baseline"]]["mean_score"] - out["conditions"]["local-only"]["mean_score"]
    if gain0 < 0.1:
        out["note"] = (f"{s['baseline']} rescues add only {gain0:.2f} to the local-only score on these tasks, "
                       "so there is little benefit to keep; gain kept is not meaningful here")
    if cut is None:
        out["note"] = f"{s['baseline']} leaked nothing, so there is nothing to cut"
    if len(tasks) < 30:
        out.setdefault("note", f"only {len(tasks)} tasks; the plan needs at least 40 for a real result")
    return out


def load_runs(run_dirs: list[str | Path]) -> tuple[list[dict], dict]:
    episodes, leaks = [], {}
    for d in map(Path, run_dirs):
        episodes += [json.loads(l) for l in (d / "episodes.jsonl").read_text().splitlines() if l.strip()]
        leaks.update(json.loads((d / "leakage.json").read_text()))
    return episodes, leaks


def print_claim(res: dict) -> None:
    print(f"\nCLAIM  {res['claim']}  ({res['n_tasks']} tasks)")
    for name, c in res.get("conditions", {}).items():
        print(f"  {name:11s} score {c['mean_score']:.3f}   secrets leaked {c['mean_secret_share_leaked']:.1%}"
              f"   rescues {c['mean_rescues']}   blocked {c['mean_blocked_firings']}")
    for key, label in (("leakage_cut", "leakage cut"), ("gain_kept", "gain kept  ")):
        if key in res:
            r = res[key]
            est = "n/a" if r["estimate"] is None else f"{r['estimate']:.0%}"
            ci = "n/a" if r["ci95"] is None else f"[{r['ci95'][0]:.0%}, {r['ci95'][1]:.0%}]"
            print(f"  {label}  {est}  95% CI {ci}  need >= {r['threshold']:.0%}  -> {r['verdict']}")
    if res.get("note"):
        print(f"  note: {res['note']}")
