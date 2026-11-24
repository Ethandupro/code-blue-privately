"""Charts for the write-up, drawn from a results folder (a `codeblue run` folder or `codeblue ec2 collect --out`).

    python -m codeblue plot results/pilot          # writes results/pilot/plots/*.png

Needs matplotlib (`pip install -e ".[plots]"`); the rest of the harness stays dependency-free.

  claim.png             leakage cut and gain kept with 95% intervals against their thresholds, clean and attacked
  conditions.png        secret share leaked and task score per condition, clean and attacked (one panel each)
  rescue_curve.png      cumulative secret share leaked after the 1st, 2nd, ... rescue, per packet (secondary)
  leak_vs_rescues.png   secret share leaked per episode against its number of rescues, per packet (secondary)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

# Fixed colour per condition (validated categorical slots 1-3 of the reference palette); never re-ordered.
COLORS = {"P0": "#2a78d6", "P3": "#eb6834", "local-only": "#1baf7a"}
ORDER = ["local-only", "P0", "P3"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise SystemExit('plotting needs matplotlib: python -m pip install -e ".[plots]"') from e
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "lines.linewidth": 2,
    })
    return plt


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _settings(claim: dict) -> list[str]:
    return [k for k in claim if isinstance(claim[k], dict) and "conditions" in claim[k]]


def _label(setting: str) -> str:
    return "clean" if setting == "none" else f"attacked ({setting})"


def plot_claim(claim: dict, out: Path, plt) -> Path | None:
    items = []
    for s in _settings(claim):
        for metric, name in (("leakage_cut", "leakage cut"), ("gain_kept", "gain kept")):
            m = claim[s].get(metric) or {}
            if m.get("estimate") is not None:
                items.append((f"{name}\n{_label(s)}", m["estimate"], m.get("ci95"), m.get("threshold")))
    if not items:
        return None
    fig, ax = plt.subplots(figsize=(7, 0.9 * len(items) + 1.2))
    for i, (name, est, ci, thr) in enumerate(items):
        y = len(items) - 1 - i
        if ci:
            ax.plot(ci, [y, y], color=COLORS["P3"], linewidth=2, solid_capstyle="round")
        ax.plot([est], [y], "o", markersize=8, color=COLORS["P3"], markeredgecolor=SURFACE, markeredgewidth=2)
        if thr is not None:
            ax.plot([thr, thr], [y - 0.35, y + 0.35], color=INK, linewidth=1.2, linestyle=(0, (3, 2)))
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([f"{n}\n{e:.0%}" + (f"  [{c[0]:.0%}, {c[1]:.0%}]" if c else "")
                        for n, e, c, _ in reversed(items)])
    lo = min([0.0] + [it[2][0] for it in items if it[2]])
    ax.set_xlim(max(lo, -0.5) - 0.05, 1.08)  # very wide intervals run off the left edge; the label has the value
    ax.set_xlabel("P3 relative to P0 (dashed line = threshold the claim needs)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(axis="y", visible=False)
    ax.set_title("The claim: P3 vs P0, point estimate and 95% interval", loc="left", fontsize=11)
    fig.tight_layout()
    path = out / "claim.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_conditions(claim: dict, out: Path, plt) -> Path | None:
    settings = _settings(claim)
    if not settings:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    width = 0.8 / len(ORDER)
    for ax, key, title in ((axes[0], "mean_secret_share_leaked", "Share of secrets that reached the cloud"),
                           (axes[1], "mean_score", "Task score")):
        for j, cond in enumerate(ORDER):
            xs, ys = [], []
            for i, s in enumerate(settings):
                c = claim[s]["conditions"].get(cond)
                if c and c.get(key) is not None:
                    xs.append(i + (j - 1) * width)
                    ys.append(c[key])
            bars = ax.bar(xs, ys, width=width - 0.03, color=COLORS[cond], label=cond, edgecolor=SURFACE, linewidth=2)
            for b, v in zip(bars, ys):
                ax.annotate(f"{v:.0%}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                            textcoords="offset points", ha="center", fontsize=8, color=INK)
        ax.set_xticks(range(len(settings)))
        ax.set_xticklabels([_label(s) for s in settings])
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylim(0, 1.1)
        ax.grid(axis="x", visible=False)
        ax.set_title(title, loc="left", fontsize=11)
    axes[0].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    path = out / "conditions.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def _lines(rows: list[dict], x: str, y: str, out_path: Path, title: str, xlabel: str, plt,
           size: str | None = None) -> Path | None:
    rows = [r for r in rows if r.get(y) not in (None, "")]
    if not rows:
        return None
    settings = sorted({r["attack"] for r in rows}, key=lambda a: (a != "none", a))
    fig, axes = plt.subplots(1, len(settings), figsize=(4.6 * len(settings), 3.6), squeeze=False, sharey=True)
    for ax, s in zip(axes[0], settings):
        for cond in ORDER:
            pts = sorted((int(r[x]), float(r[y]), int(r[size]) if size else 0)
                         for r in rows if r["attack"] == s and r["packet"] == cond)
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS[cond], label=cond,
                    marker="o", markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
            last = pts[-1]
            ax.annotate(cond, (last[0], last[1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=9, color=INK)
        ax.set_title(_label(s), loc="left", fontsize=10, color=MUTED)
        ax.set_xlabel(xlabel)
        ax.xaxis.get_major_locator().set_params(integer=True)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    axes[0][0].set_ylabel("share of the task's secrets")
    handles, labels = axes[0][0].get_legend_handles_labels()
    for ax in axes[0][1:]:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, frameon=False, loc="lower center", ncol=len(labels))
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_all(results: str | Path) -> list[Path]:
    results = Path(results)
    claim_path = results / "claim.json"
    if not claim_path.exists():
        raise FileNotFoundError(f"{claim_path} not found; is {results} a run or collect folder?")
    plt = _plt()
    claim = json.loads(claim_path.read_text())
    out = results / "plots"
    out.mkdir(exist_ok=True)
    made = [plot_claim(claim, out, plt), plot_conditions(claim, out, plt),
            _lines(_rows(results / "rescue_curve.csv"), "rescue_n", "mean_cum_secret_share_carried",
                   out / "rescue_curve.png", "Secrets leaked so far, after each rescue (secondary)",
                   "rescue number", plt),
            _lines(_rows(results / "leak_vs_rescues.csv"), "n_rescues", "mean_secret_share_leaked",
                   out / "leak_vs_rescues.png", "Secrets leaked per episode, by its number of rescues (secondary)",
                   "rescues in the episode", plt, size="episodes")]
    return [p for p in made if p]
