import json

import pytest

from codeblue.config import load_config
from codeblue.detect import RegexDetector
from codeblue.gateway import CloudGateway, read_cloud_requests
from codeblue.llm import MockCloudModel, MockLocalModel
from codeblue.loop import LoopConfig, run_episode
from codeblue.packets import make_builder
from codeblue.optional.triggers_extra import RandomTrigger, matched_rate
from codeblue.runner import run_experiment
from codeblue.scan import scan_log
from codeblue.triggers import NeverTrigger, SelfCheckTrigger

from .conftest import ROOT


def _run(room, tmp_path, task_id, packet, trigger, cfg=None):
    tasks = list(room.tasks.values())
    log = tmp_path / "cloud.jsonl"
    gw = CloudGateway(MockCloudModel(tasks), log)
    ep = f"{task_id}-{packet}"
    res = run_episode(ep, room, room.tasks[task_id], MockLocalModel(tasks), gw,
                      make_builder(packet) if packet else None, trigger, RegexDetector(room.glossary), cfg or LoopConfig())
    leaks = scan_log(room, log, {ep: task_id})[ep] if log.exists() else None
    return res, leaks


def test_local_only_never_calls_cloud(room, tmp_path):
    res, _ = _run(room, tmp_path, "t1_board_cost_summary", None, NeverTrigger())
    assert res["n_cloud_steps"] == 0 and res["score"] == 0.0
    assert not (tmp_path / "cloud.jsonl").exists()


def test_selfcheck_rescue_and_hand_back(room, tmp_path):
    res, leaks = _run(room, tmp_path, "t1_board_cost_summary", "P0", SelfCheckTrigger())
    assert res["score"] == 1.0 and res["n_rescues"] == 1 and res["n_cloud_steps"] <= 2
    assert leaks["n_essential_leaked"] == 1 and leaks["n_bystander_leaked"] > 0


@pytest.mark.parametrize("task_id", ["t1_board_cost_summary", "t2_lowest_margin"])
def test_p3_keeps_success_and_cuts_leakage(room, tmp_path, task_id):
    p0, l0 = _run(room, tmp_path / "p0", task_id, "P0", SelfCheckTrigger())
    p3, l3 = _run(room, tmp_path / "p3", task_id, "P3", SelfCheckTrigger())
    assert p3["score"] == p0["score"] == 1.0
    assert len(l3["leaked"]) < len(l0["leaked"])
    assert l3["leaked"] == {}


def test_p2_breaks_and_p4_fails(room, tmp_path):
    p2, _ = _run(room, tmp_path / "p2", "t1_board_cost_summary", "P2", SelfCheckTrigger())
    p4, l4 = _run(room, tmp_path / "p4", "t1_board_cost_summary", "P4", SelfCheckTrigger())
    assert p2["score"] < 1.0
    assert p4["score"] == 0.0 and l4["leaked"] == {}


def test_no_cap_by_default_and_caps_log_blocked_firings(room, tmp_path):
    # P4 can never help, so the agent keeps asking. With no cap, every request becomes a rescue.
    free, _ = _run(room, tmp_path / "free", "t3_restructuring_checklist", "P4", SelfCheckTrigger())
    assert free["n_rescues"] > 3 and free["n_blocked_firings"] == 0
    capped, _ = _run(room, tmp_path / "cap", "t3_restructuring_checklist", "P4", SelfCheckTrigger(),
                     LoopConfig(max_rescues=3, cooldown=2))
    assert capped["n_rescues"] <= 3 < free["n_rescues"]
    # The cap does not hide demand: blocked firings are recorded with what blocked them.
    assert capped["n_blocked_firings"] > 0
    assert {b["blocked_by"] for b in capped["blocked_firings"]} <= {"max_rescues", "cooldown"}


def test_random_trigger_is_deterministic(room, tmp_path):
    a, _ = _run(room, tmp_path / "a", "t2_lowest_margin", "P0", RandomTrigger(0.5, seed=7))
    b, _ = _run(room, tmp_path / "b", "t2_lowest_margin", "P0", RandomTrigger(0.5, seed=7))
    assert [r["at_step"] for r in a["rescues"]] == [r["at_step"] for r in b["rescues"]]


def test_gateway_logs_before_sending(tmp_path):
    class Boom:
        name = "boom"

        def complete(self, messages, purpose):
            raise RuntimeError("network down")

    gw = CloudGateway(Boom(), tmp_path / "c.jsonl")
    with pytest.raises(RuntimeError):
        gw.complete([{"role": "user", "content": "SECRET"}], {"episode_id": "e"})
    assert read_cloud_requests(tmp_path / "c.jsonl")[0]["request"][0]["content"] == "SECRET"


def test_full_mock_experiment(tmp_path):
    cfg = load_config(ROOT / "configs" / "mock.toml")
    out = run_experiment(cfg, tmp_path / "run", quiet=True)
    for name in ["summary.csv", "leakage.json", "episodes.jsonl", "cloud_log.jsonl", "local_log.jsonl"]:
        assert (out / name).exists()
    # Local traffic never lands in the cloud log.
    purposes = {json.loads(l).get("purpose") for l in (out / "local_log.jsonl").read_text().splitlines()}
    assert "selfcheck" in purposes
    assert all("purpose" not in json.loads(l) for l in (out / "cloud_log.jsonl").read_text().splitlines())


def test_parked_full_grid_still_runs(tmp_path):
    out = run_experiment(load_config(ROOT / "configs" / "optional" / "mock-full-grid.toml"), tmp_path / "run", quiet=True)
    rate = matched_rate(out / "episodes.jsonl")
    assert 0 < rate < 1
