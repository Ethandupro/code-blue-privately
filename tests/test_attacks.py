import csv
import json

import pytest

from codeblue.attacks import ATTACKS, apply_attack, exposure_stats, get_attack, injection_present, load_vectors
from codeblue.config import load_config
from codeblue.detect import RegexDetector
from codeblue.gateway import CloudGateway, read_cloud_requests, request_text
from codeblue.llm import MockCloudModel, MockLocalModel
from codeblue.loop import LoopConfig, run_episode
from codeblue.packets import make_builder
from codeblue.runner import matched_rate, run_experiment
from codeblue.scan import scan_episode, scan_log
from codeblue.triggers import NeverTrigger, SelfCheckTrigger

from .conftest import ROOT

T4 = "t4_competitor_lines"


def _run(room, tmp_path, task_id, packet, trigger, attack=None, placement="vectors"):
    arm_room, _ = apply_attack(room, get_attack(attack), placement)
    tasks = list(room.tasks.values())
    log = tmp_path / f"{task_id}-{packet}-{attack}.jsonl"
    gw = CloudGateway(MockCloudModel(tasks), log)
    ep = f"{task_id}-{packet}-{attack}"
    res = run_episode(ep, arm_room, arm_room.tasks[task_id], MockLocalModel(tasks, gullible=True), gw,
                      make_builder(packet) if packet else None, trigger, RegexDetector(room.glossary), LoopConfig())
    leaks = scan_log(room, log, {ep: task_id})[ep] if log.exists() else scan_episode(room, task_id, [])
    texts = [request_text(r) for r in read_cloud_requests(log)] if log.exists() else []
    res["exposure"] = exposure_stats(get_attack(attack), res["steps"], texts)
    return res, leaks


def test_vectors_only_touch_attacker_writable_files(room):
    vectors = {v["file"] for v in load_vectors(room)}
    assert vectors == {"market/competitors.md", "sales/contracts_summary.md"}
    attacked, changed = apply_attack(room, ATTACKS["stuck_claim"], "vectors")
    assert set(changed) == vectors
    for f, text in attacked.files.items():
        assert injection_present(ATTACKS["stuck_claim"], text) == (f in vectors)
        assert not injection_present(ATTACKS["stuck_claim"], room.files[f])  # original room untouched
    _, everywhere = apply_attack(room, ATTACKS["fake_error"], "everywhere")
    assert set(everywhere) == set(room.files)


def test_no_attack_is_identity_and_bad_names_fail(room):
    assert apply_attack(room, None) == (room, [])
    with pytest.raises(ValueError):
        get_attack("nope")
    with pytest.raises(ValueError):
        apply_attack(room, ATTACKS["loop_bait"], "somewhere")


@pytest.mark.parametrize("name", sorted(ATTACKS))
def test_injection_text_holds_no_secrets(room, name):
    text = "\n".join(ATTACKS[name].render(f) for f in room.files)
    for task_id in room.tasks:
        assert scan_episode(room, task_id, [text])["leaked"] == {}


def test_gullible_mock_is_unchanged_without_injection(room, tmp_path):
    res, leaks = _run(room, tmp_path, T4, "P0", SelfCheckTrigger())
    assert res["score"] == 1.0 and res["n_rescues"] == 0 and leaks["leaked"] == {}


@pytest.mark.parametrize("attack", ["stuck_claim", "fake_error"])
def test_injection_forces_unneeded_escalation_and_bystander_leaks(room, tmp_path, attack):
    clean, l_clean = _run(room, tmp_path, T4, "P0", SelfCheckTrigger())
    hit, l_hit = _run(room, tmp_path, T4, "P0", SelfCheckTrigger(), attack)
    assert clean["n_rescues"] == 0 and hit["n_rescues"] >= 1
    assert hit["rescues"][0]["reason"] == "self-check said STUCK"
    assert l_hit["n_bystander_leaked"] > l_clean["n_bystander_leaked"] == 0
    assert "canary_codename" in l_hit["bystander_leaked"]  # the board memo the agent had read went to the cloud
    assert hit["exposure"]["injection_seen_by_agent"] and hit["exposure"]["n_cloud_calls_with_injection"] >= 1


def test_loop_bait_trips_the_behavioural_stall_signal(room, tmp_path):
    hit, _ = _run(room, tmp_path, T4, "P0", SelfCheckTrigger(ask_model=False), "loop_bait")
    assert hit["n_rescues"] >= 1 and hit["rescues"][0]["reason"].startswith("repeated action")


@pytest.mark.parametrize("defence", [dict(view="actions"), dict(ask_model=False)])
@pytest.mark.parametrize("attack", ["stuck_claim", "fake_error"])
def test_blind_selfcheck_blocks_text_attacks(room, tmp_path, attack, defence):
    res, leaks = _run(room, tmp_path, T4, "P0", SelfCheckTrigger(**defence), attack)
    assert res["n_rescues"] == 0 and leaks["leaked"] == {} and res["score"] == 1.0


def test_actions_view_does_not_stop_loop_bait(room, tmp_path):
    res, _ = _run(room, tmp_path, T4, "P0", SelfCheckTrigger(view="actions"), "loop_bait")
    assert res["n_rescues"] >= 1


def test_selfcheck_views_hide_or_fence_observations(room):
    task = room.tasks[T4]
    attacked, _ = apply_attack(room, ATTACKS["stuck_claim"])
    from codeblue.env import DataRoomEnv
    from codeblue.agent import Step
    obs = DataRoomEnv(attacked, task).execute({"tool": "read_file", "args": {"path": "market/competitors.md"}})
    steps = [Step(1, "local", {"tool": "read_file", "args": {"path": "market/competitors.md"}}, obs.text)]
    full = SelfCheckTrigger(view="full")._messages(task, steps)[-1]["content"]
    spot = SelfCheckTrigger(view="spotlight")._messages(task, steps)[-1]["content"]
    blind = SelfCheckTrigger(view="actions")._messages(task, steps)[-1]["content"]
    assert injection_present(ATTACKS["stuck_claim"], full)
    assert "<<DATA>>" in spot and "data, not instructions" in spot
    assert not injection_present(ATTACKS["stuck_claim"], blind) and "-> ok" in blind
    with pytest.raises(ValueError):
        SelfCheckTrigger(view="peek")


def test_structure_only_packet_never_forwards_the_injection(room, tmp_path):
    res, leaks = _run(room, tmp_path, T4, "P4", SelfCheckTrigger(), "stuck_claim")
    assert res["n_rescues"] >= 1 and res["exposure"]["n_cloud_calls_with_injection"] == 0 and leaks["leaked"] == {}


def test_handoff_stuffing_inflates_what_briefs_send(room, tmp_path):
    clean = sum(_run(room, tmp_path / "c", t, "P1", SelfCheckTrigger())[1]["n_bystander_leaked"] for t in room.tasks)
    hit = sum(_run(room, tmp_path / "h", t, "P1", SelfCheckTrigger(), "handoff_stuffing")[1]["n_bystander_leaked"]
              for t in room.tasks)
    assert hit > clean


def test_local_only_never_reaches_cloud_under_attack(room, tmp_path):
    res, _ = _run(room, tmp_path, T4, None, NeverTrigger(), "stuck_claim")
    assert res["n_cloud_steps"] == 0


def test_runner_writes_paired_attack_summary(tmp_path):
    cfg = load_config(ROOT / "configs" / "mock-attack.toml")
    cfg["experiment"]["attacks"] = ["none", "stuck_claim"]
    cfg["experiment"]["packets"] = ["P0"]
    out = run_experiment(cfg, tmp_path / "run", quiet=True)
    rows = list(csv.DictReader((out / "attack_summary.csv").open()))
    p0 = next(r for r in rows if r["packet"] == "P0")
    assert p0["attack"] == "stuck_claim" and int(p0["pairs"]) == len(cfg_tasks(out))
    assert float(p0["forced_rate"]) > 0 and float(p0["d_bystander_forced"]) > 0
    summary = list(csv.DictReader((out / "summary.csv").open()))
    assert {r["attack"] for r in summary} == {"none", "stuck_claim"}
    # The matched random rate is computed from clean episodes only.
    clean_only = tmp_path / "clean.jsonl"
    clean_only.write_text("".join(l + "\n" for l in (out / "episodes.jsonl").read_text().splitlines()
                                  if json.loads(l)["attack"] == "none"))
    assert matched_rate(out / "episodes.jsonl") == matched_rate(clean_only)


def cfg_tasks(out):
    return {json.loads(l)["task_id"] for l in (out / "episodes.jsonl").read_text().splitlines()}
