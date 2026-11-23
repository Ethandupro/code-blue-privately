import json

import pytest

from codeblue.attacks import ATTACKS, apply_attack, exposure_stats, get_attack, injection_present, load_vectors
from codeblue.config import load_config
from codeblue.detect import RegexDetector
from codeblue.gateway import CloudGateway, read_cloud_requests, request_text
from codeblue.llm import MockCloudModel, MockLocalModel
from codeblue.loop import LoopConfig, run_episode
from codeblue.packets import make_builder
from codeblue.runner import run_experiment
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


def test_main_attack_forces_rescues_and_stuffs_the_note(room, tmp_path):
    clean, _ = _run(room, tmp_path / "c", T4, "P1", SelfCheckTrigger())
    hit, l_hit = _run(room, tmp_path / "h", T4, "P1", SelfCheckTrigger(), "stuck_and_stuff")
    assert clean["n_rescues"] == 0 and hit["n_rescues"] >= 1
    assert l_hit["n_bystander_leaked"] > 0  # the unmasked note now carries the files it pasted


def test_p3_masks_the_stuffed_note(room, tmp_path):
    p0, l0 = _run(room, tmp_path / "p0", T4, "P0", SelfCheckTrigger(), "stuck_and_stuff")
    p3, l3 = _run(room, tmp_path / "p3", T4, "P3", SelfCheckTrigger(), "stuck_and_stuff")
    assert p3["n_rescues"] >= 1
    assert len(l3["leaked"]) < len(l0["leaked"])


def test_handoff_stuffing_inflates_what_briefs_send(room, tmp_path):
    clean = sum(_run(room, tmp_path / "c", t, "P1", SelfCheckTrigger())[1]["n_bystander_leaked"] for t in room.tasks)
    hit = sum(_run(room, tmp_path / "h", t, "P1", SelfCheckTrigger(), "handoff_stuffing")[1]["n_bystander_leaked"]
              for t in room.tasks)
    assert hit > clean


def test_local_only_never_reaches_cloud_under_attack(room, tmp_path):
    res, _ = _run(room, tmp_path, T4, None, NeverTrigger(), "stuck_and_stuff")
    assert res["n_cloud_steps"] == 0


def test_runner_reports_all_three_parts_of_the_claim(tmp_path):
    out = run_experiment(load_config(ROOT / "configs" / "mock.toml"), tmp_path / "run", quiet=True)
    claim = json.loads((out / "claim.json").read_text())
    assert set(claim["parts"]) == {"1 leakage cut (clean)", "2 gain kept (clean)",
                                   "3 leakage cut under stuck_and_stuff"}
    eff = claim["stuck_and_stuff"]["attack_effect"]
    assert eff["P0"]["forced_rate"] > 0 and eff["local-only"]["injection_to_cloud_rate"] == 0
