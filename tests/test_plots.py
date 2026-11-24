"""Charts are drawn from a mock run folder (skipped when matplotlib is not installed)."""

import pytest

pytest.importorskip("matplotlib")

from codeblue.config import load_config  # noqa: E402
from codeblue.plots import plot_all  # noqa: E402
from codeblue.runner import run_experiment  # noqa: E402


def test_plots_from_mock_run(tmp_path):
    out = run_experiment(load_config("configs/mock.toml"), tmp_path / "run", quiet=True)
    made = {p.name for p in plot_all(out)}
    assert made == {"claim.png", "conditions.png", "rescue_curve.png", "leak_vs_rescues.png"}
