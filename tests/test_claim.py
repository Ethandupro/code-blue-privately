import json

from codeblue.claim import evaluate, load_runs
from codeblue.config import load_config
from codeblue.runner import run_experiment

from .conftest import ROOT


def _ep(task, packet, score, rescues=1):
    eid = f"{task}-{packet}"
    return {"episode_id": eid, "task_id": task, "room": "r", "packet": packet,
            "trigger": "never" if packet is None else "selfcheck", "score": score, "n_rescues": rescues}


def _leak(n):
    return {"n_essential_leaked": 0, "n_bystander_leaked": n, "n_secrets_total": 10}


def test_claim_maths():
    eps, leaks = [], {}
    for i in range(40):
        t = f"t{i}"
        for packet, score, n in ((None, 0.0, 0), ("P0", 1.0, 8), ("P3", 0.9, 1)):
            e = _ep(t, packet, score)
            eps.append(e)
            leaks[e["episode_id"]] = _leak(n)
    res = evaluate(eps, leaks, {"resamples": 200})
    assert res["n_tasks"] == 40
    assert res["leakage_cut"]["estimate"] == 0.875 and res["leakage_cut"]["verdict"] == "supported"
    assert res["gain_kept"]["estimate"] == 0.9 and res["gain_kept"]["verdict"] == "supported"


def test_claim_rejected_when_masking_breaks_the_rescue():
    eps, leaks = [], {}
    for i in range(40):
        for packet, score, n in ((None, 0.0, 0), ("P0", 1.0, 8), ("P3", 0.2, 0)):
            e = _ep(f"t{i}", packet, score)
            eps.append(e)
            leaks[e["episode_id"]] = _leak(n)
    res = evaluate(eps, leaks, {"resamples": 200})
    assert res["gain_kept"]["verdict"] == "rejected"


def test_run_writes_claim_and_pools(tmp_path):
    out = run_experiment(load_config(ROOT / "configs" / "mock.toml"), tmp_path / "run", quiet=True)
    claim = json.loads((out / "claim.json").read_text())
    assert set(claim["conditions"]) == {"local-only", "P0", "P3"}
    assert claim["leakage_cut"]["estimate"] > 0.75
    eps, leaks = load_runs([out])
    assert evaluate(eps, leaks)["n_tasks"] == claim["n_tasks"]
