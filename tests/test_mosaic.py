import csv
import json

from codeblue.config import load_config
from codeblue.mosaic import mosaic_episode, restore
from codeblue.runner import run_experiment
from codeblue.scan import scan_episode

from .conftest import ROOT

TASK = "t1_board_cost_summary"
C = "composite_tanaka_at_risk"


def _mosaic(room, texts, table=None, call_rescue=None):
    leak = scan_episode(room, TASK, texts)
    return mosaic_episode(room, texts, leak, table or {}, call_rescue or list(range(len(texts))))[C]


def test_pieces_sent_over_several_calls_add_up_to_a_leak(room):
    texts = ["Tanaka Foods renews on 1 Mar 2027.",
             "customer,contract,margin\nTanaka Foods,Reefer contract,3.1%",
             "Straits Reefer Lines has been pitching Tanaka Foods' reefer volumes."]
    first_two = _mosaic(room, texts[:2])
    assert first_two["status"] == "partial" and first_two["pieces_in_clear"] == 2
    m = _mosaic(room, texts)
    assert m["status"] == "leaked" and m["across_calls"] and m["completed_rescue"] == 2


def test_consistent_placeholders_let_the_cloud_link_the_pieces(room):
    table = {"Tanaka Foods": "<NAME_1>", "1 Mar 2027": "<DATE_1>", "3.1%": "<PCT_1>", "Straits Reefer Lines": "<NAME_2>"}
    texts = ["<NAME_1> renews on <DATE_1>.", "<NAME_1> margin is <PCT_1>.",
             "<NAME_2> has been pitching <NAME_1>'s reefer volumes."]
    assert restore(texts[0], table) == "Tanaka Foods renews on 1 Mar 2027."
    m = _mosaic(room, texts, table)
    assert m["status"] == "linked" and m["pieces_in_clear"] == 0 and m["pieces_reached"] == 3
    # Without the shared placeholder table nothing reaches the cloud.
    assert _mosaic(room, texts)["status"] == "none"


def test_run_writes_mosaic_table(tmp_path):
    out = run_experiment(load_config(ROOT / "configs" / "mock.toml"), tmp_path / "run", quiet=True)
    rows = list(csv.DictReader((out / "mosaic.csv").open()))
    assert {r["packet"] for r in rows} == {"local-only", "P0", "P3"}
    lo = [r for r in rows if r["packet"] == "local-only"]
    assert all(float(r["leaked_rate"]) == 0 and float(r["linked_rate"]) == 0 for r in lo)
    claim = json.loads((out / "claim.json").read_text())
    assert claim["secondary"]["mosaic"]
