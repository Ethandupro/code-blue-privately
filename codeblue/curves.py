"""Secondary analysis: how leakage grows with the number of times the cloud is called.

Not part of the headline claim. Two views, per (attack setting, packet):

rescue_curve.csv     cumulative unique secrets leaked after the 1st, 2nd, ... nth rescue. A secret counts once,
                     at the rescue where it first reached the cloud. P0 resends the growing history at every
                     rescue, so its curve should keep rising; P3 sends a short masked note with the same
                     placeholders each time, so its curve should stay close to flat.
                     Two averages per point: over the episodes that reached that many rescues, and over all
                     episodes with at least one rescue, carrying each episode's final value forward.
leak_vs_rescues.csv  per episode, total secrets leaked against how many rescues it had, grouped by rescue count,
                     plus a least-squares slope (secret share leaked per extra rescue) in claim.json.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean


def curve_by_rescue(leak: dict, call_rescue: list[int]) -> list[int]:
    """Cumulative unique secrets after each rescue, from one episode's scan and the rescue index of each call."""
    if not call_rescue:
        return []
    n = max(call_rescue) + 1
    first = [0] * n
    for info in leak.get("leaked", {}).values():
        call = info.get("call", 0)
        if 0 <= call < len(call_rescue):
            first[call_rescue[call]] += 1
    out, total = [], 0
    for k in first:
        total += k
        out.append(total)
    return out


def _slope(xs: list[float], ys: list[float]) -> float | None:
    if len(set(xs)) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)


def rescue_tables(episodes: list[dict], leaks: dict[str, dict]) -> tuple[list[dict], list[dict], dict]:
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in episodes:
        if e["packet"] is not None:
            groups[(e.get("attack", "none"), e["packet"])].append(e)

    curve_rows, count_rows, slopes = [], [], {}
    for (attack, packet), eps in sorted(groups.items()):
        total = lambda e: leaks[e["episode_id"]].get("n_secrets_total") or 1  # noqa: E731
        curves = [(leaks[e["episode_id"]].get("curve_by_rescue") or [], total(e)) for e in eps]
        rescued = [(c, t) for c, t in curves if c]
        longest = max((len(c) for c, _ in rescued), default=0)
        for n in range(1, longest + 1):
            reached = [c[n - 1] / t for c, t in rescued if len(c) >= n]
            carried = [c[min(n, len(c)) - 1] / t for c, t in rescued]
            curve_rows.append({"attack": attack, "packet": packet, "rescue_n": n,
                               "episodes_reaching": len(reached),
                               "mean_cum_secret_share_reaching": round(mean(reached), 3),
                               "mean_cum_secret_share_carried": round(mean(carried), 3)})

        share = {e["episode_id"]: (leaks[e["episode_id"]].get("n_essential_leaked", 0)
                                   + leaks[e["episode_id"]].get("n_bystander_leaked", 0)) / total(e) for e in eps}
        by_count: dict[int, list[float]] = defaultdict(list)
        for e in eps:
            by_count[e["n_rescues"]].append(share[e["episode_id"]])
        for k in sorted(by_count):
            count_rows.append({"attack": attack, "packet": packet, "n_rescues": k, "episodes": len(by_count[k]),
                               "mean_secret_share_leaked": round(mean(by_count[k]), 3)})
        s = _slope([float(e["n_rescues"]) for e in eps], [share[e["episode_id"]] for e in eps])
        slopes[f"{attack}/{packet}"] = None if s is None else round(s, 4)
    return curve_rows, count_rows, slopes
