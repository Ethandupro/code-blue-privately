from codeblue.agent import Step
from codeblue.detect import LLMDetector, OracleDetector, RegexDetector
from codeblue.llm import MockLocalModel
from codeblue.packets import P0FullHistory, P1Brief, P2Blanked, P3Placeholders, P4StructureOnly, RescueContext

MEMO = "memos/2026-11-board-prep.md"


def _steps(room):
    return [
        Step(1, "local", {"tool": "list_files", "args": {}}, "\n".join(room.files)),
        Step(2, "local", {"tool": "read_file", "args": {"path": MEMO}}, room.files[MEMO]),
    ]


def _body(packet):
    return packet.messages[-1]["content"]


def test_regex_detector_finds_memo_secrets(room):
    found = {s.text for s in RegexDetector(room.glossary).find(room.files[MEMO])}
    assert {"BLUE HERON-7Q", "HL-ACQ-2291-KX", "S$41.5 million", "Mei Ling Tan",
            "14 Jan 2027", "meiling.tan@harbourline.example"} <= found


def test_p0_is_verbatim(room):
    task = room.tasks["t1_board_cost_summary"]
    body = _body(P0FullHistory().build(task, _steps(room), RescueContext()))
    assert "HL-ACQ-2291-KX" in body and task.goal in body


def test_p2_blanks(room):
    task = room.tasks["t1_board_cost_summary"]
    ctx = RescueContext(detector=RegexDetector(room.glossary))
    body = _body(P2Blanked().build(task, _steps(room), ctx))
    assert "HL-ACQ-2291-KX" not in body and "[REDACTED]" in body
    assert MEMO in body  # file paths survive


def test_p3_placeholders_consistent_and_restorable(room):
    task = room.tasks["t1_board_cost_summary"]
    ctx = RescueContext(detector=RegexDetector(room.glossary))
    p = P3Placeholders().build(task, _steps(room), ctx)
    body = _body(p)
    assert "BLUE HERON-7Q" not in body and "Mei Ling Tan" not in body
    ph = ctx.table.to_ph["BLUE HERON-7Q"]
    assert body.count(ph) == 2  # same value, same placeholder
    restored = p.restore_action({"tool": "submit", "args": {"answer": f"Ask {ctx.table.to_ph['Mei Ling Tan']} re {ph}"}})
    assert restored["args"]["answer"] == "Ask Mei Ling Tan re BLUE HERON-7Q"
    # A second rescue in the same episode reuses the table.
    p2 = P3Placeholders().build(task, _steps(room), ctx)
    assert ph in _body(p2)


def test_p4_has_no_content(room):
    task = room.tasks["t1_board_cost_summary"]
    body = _body(P4StructureOnly().build(task, _steps(room), RescueContext()))
    assert task.goal not in body and MEMO not in body and "BLUE" not in body
    assert "read_file" in body


def test_p1_uses_local_brief(room):
    task = room.tasks["t1_board_cost_summary"]
    ctx = RescueContext(local_model=MockLocalModel(room.tasks.values()))
    p = P1Brief().build(task, _steps(room), ctx)
    assert "HANDOFF NOTE" in _body(p) and p.info["brief"]


def test_llm_and_oracle_detectors(room):
    llm = LLMDetector(MockLocalModel(room.tasks.values()), room.glossary)
    assert any(s.text == "HL-ACQ-2291-KX" for s in llm.find(room.files[MEMO]))
    oracle = OracleDetector(room.secrets)
    assert any("62 roles" == s.text for s in oracle.find(room.files["hr/restructuring_plan_2027.md"]))
