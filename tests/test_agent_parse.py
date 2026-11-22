from codeblue.agent import parse_action


def test_parse_plain_json():
    a = parse_action('{"thought": "x", "tool": "read_file", "args": {"path": "a.md"}}')
    assert a["tool"] == "read_file" and a["args"] == {"path": "a.md"}


def test_parse_with_think_and_prose():
    text = '<think>{"tool": "submit"}</think>Sure! Here: {"tool": "search", "args": {"query": "a}b"}} done'
    a = parse_action(text)
    assert a["tool"] == "search" and a["args"]["query"] == "a}b"


def test_parse_invalid():
    assert parse_action("I don't know")["tool"] == "invalid"
    assert parse_action('{"tool": "rm_rf"}')["tool"] == "invalid"
