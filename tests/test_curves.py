import csv
import json

from codeblue.config import load_config
from codeblue.curves import curve_by_rescue, rescue_tables
from codeblue.runner import run_experiment

from .conftest import ROOT


def test_curve_counts_each_secret_once_at_its_first_rescue():
    leak = {"leaked": {"a": {"call": 0}, "b": {"call": 1}, "c": {"call": 3}}}
    # calls 0-1 belong to rescue 0, calls 2-3 to rescue 1, call 4 to rescue 2
    assert curve_by_rescue(leak, [0, 0, 1, 1, 2]) == [2, 3, 3]
    assert curve_by_rescue({"leaked": {}}, []) == []


def test_tables_and_slope():
    eps, leaks = [], {}
    for i, (packet, n) in enumerate([("P0", 1), ("P0", 2), ("P0", 3), ("P3", 1), ("P3", 3)]):
        eid = f"e{i}"
        eps.append({"episode_id": eid, "packet": packet, "n_rescues": n, "attack": "none"})
        k = n if packet == "P0" else 1
        leaks[eid] = {"n_essential_leaked": 0, "n_bystander_leaked": k, "n_secrets_total": 10,
                      "curve_by_rescue": list(range(1, k + 1)) + [k] * (n - k)}
    curve, counts, slopes = rescue_tables(eps, leaks)
    p0 = [r for r in curve if r["packet"] == "P0"]
    assert [r["episodes_reaching"] for r in p0] == [3, 2, 1]
    assert slopes["none/P0"] == 0.1 and slopes["none/P3"] == 0.0
    assert {(r["packet"], r["n_rescues"]) for r in counts} >= {("P0", 3), ("P3", 1)}


def test_run_writes_rescue_curves(tmp_path):
    out = run_experiment(load_config(ROOT / "configs" / "mock.toml"), tmp_path / "run", quiet=True)
    rows = list(csv.DictReader((out / "rescue_curve.csv").open()))
    assert {r["packet"] for r in rows} == {"P0", "P3"}
    assert (out / "leak_vs_rescues.csv").exists()
    claim = json.loads((out / "claim.json").read_text())
    assert "none/P0" in claim["secondary"]["secret_share_per_extra_rescue"]
    leaks = json.loads((out / "leakage.json").read_text())
    assert all("curve_by_rescue" in v for v in leaks.values())
