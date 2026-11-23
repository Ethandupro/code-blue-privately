"""P3 for EC2-Bench's delegation mode: send the cloud a masked handoff note instead of the full history.

This file runs INSIDE EC2-Bench's cloud proxy (skills/cloud-router/privacy_proxy.py), where `codeblue` is not
installed. `codeblue ec2 patch` copies it there as cb_p3.py, next to a copy of codeblue/detect.py (cb_detect.py).
Standard library only.

When the edge model calls `escalate`, the patched proxy calls `build_view` instead of forwarding the whole
conversation:
  1. the EDGE model writes a short handoff note from its own history (same prompt as the data-room harness);
  2. the task text and the note are scanned by the detector (regex by default, or the edge model + regex);
  3. every detected value is replaced by a typed placeholder (<NAME_1>, <MONEY_2>, ...). The table lives in the
     proxy for the whole episode, so the same value gets the same placeholder in every rescue;
  4. the cloud gets only: a role prompt, the masked task and note, and the same tool list as in P0.
Placeholders in the cloud's tool-call arguments are filled back in (`State.table.restore`) before the step runs.
The table is written to /tmp/codeblue_p3_table.json (kept on the device) so the mosaic analysis can check what
the placeholders let the cloud link. The edge model's prompt is NOT changed, so it escalates exactly as in P0.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

try:  # inside our repo (tests)
    from codeblue.detect import LLMDetector, RegexDetector, strip_thinking
except ImportError:  # inside EC2-Bench's proxy
    from cb_detect import LLMDetector, RegexDetector, strip_thinking  # type: ignore

TABLE_PATH = os.environ.get("CODEBLUE_P3_TABLE", "/tmp/codeblue_p3_table.json")
MAX_RESULT_CHARS = 2000  # per tool result in the history the edge model summarises
MAX_HISTORY_CHARS = 24000  # keep the head and tail of longer histories

# Same note prompt as the data-room harness (codeblue/agent.py BRIEF_PROMPT).
BRIEF_PROMPT = (
    "Write a short handoff note (at most 120 words) for an expert helper who will take over for a couple "
    "of steps. Say what the task is, what you have tried, which files you have seen, and where you are stuck."
)

CLOUD_ROLE_P3 = (
    "You are the stronger cloud model behind an `escalate` tool. A smaller edge model has been working on a task "
    "and escalated its NEXT single step to you. You do not see its history: you see the task and a short handoff "
    "note it wrote. Names, amounts, dates, codes and other details may be replaced by placeholders such as <NAME_1> "
    "or <MONEY_2>. The same placeholder always means the same value, and you may use placeholders in your tool "
    "arguments: they are filled in before the tool runs.\n\n"
    "Perform that one step now, using the tools available. Produce the actual action (a tool call, or the answer if "
    "the task is genuinely done), not an analysis. Do not invent paths, files or data the note does not mention; "
    "if you need to see something, your step can be to read or list it."
)


class Table:
    """Consistent value <-> placeholder mapping for one episode. Never sent to the cloud."""

    PH = re.compile(r"<\s*([A-Z]+_\d+)\s*>")

    def __init__(self):
        self.to_ph: dict[str, str] = {}
        self.to_value: dict[str, str] = {}
        self.counts: Counter = Counter()

    def placeholder(self, value: str, category: str) -> str:
        key = value.strip()
        for k, ph in self.to_ph.items():
            if k.lower() == key.lower():
                return ph
        self.counts[category] += 1
        ph = f"<{category}_{self.counts[category]}>"
        self.to_ph[key] = ph
        self.to_value[ph[1:-1]] = key
        return ph

    def mask(self, text: str) -> str:
        return replace_values(text, self.to_ph)

    def restore(self, text: str) -> str:
        return self.PH.sub(lambda m: self.to_value.get(m.group(1), m.group(0)), text)


def replace_values(text: str, values: dict[str, str]) -> str:
    """Replace every occurrence of each value (case-insensitive, longest first). Same as codeblue.packets."""
    if not values:
        return text
    keys = sorted(values, key=len, reverse=True)
    lower = {k.lower(): v for k, v in values.items()}
    pat = re.compile("|".join(rf"(?<!\w){re.escape(k)}(?!\w)" for k in keys), re.I)
    return pat.sub(lambda m: lower[m.group(0).lower()], text)


class State:
    """Per-episode P3 state kept on the proxy's SessionState."""

    def __init__(self, table_path: str | None = TABLE_PATH):
        self.table = Table()
        self.notes: list[str] = []
        self.table_path = table_path

    def save(self) -> None:
        if self.table_path:
            with open(self.table_path, "w", encoding="utf-8") as f:
                json.dump({"placeholder_table": self.table.to_ph, "n_notes": len(self.notes)}, f, ensure_ascii=False)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def task_text(messages: list[dict]) -> str:
    """The task: the first user message."""
    for m in messages:
        if m.get("role") == "user":
            return _text(m.get("content")).strip()
    return ""


def render_history(messages: list[dict], signal_tool: str = "escalate") -> str:
    """The edge agent's history as plain text, for the edge model to summarise. Never sent to the cloud."""
    lines: list[str] = []
    seen_task = False
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        if role == "user" and not seen_task:
            seen_task = True
            continue
        text = _text(m.get("content")).strip()
        if role == "assistant":
            if text:
                lines.append(f"YOU: {text}")
            for tc in m.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                if fn.get("name") == signal_tool:
                    continue
                lines.append(f"YOU CALLED {fn.get('name')}({str(fn.get('arguments', ''))[:500]})")
        elif role in ("tool", "toolResult"):
            lines.append(f"RESULT: {text[:MAX_RESULT_CHARS]}")
        elif text:
            lines.append(f"{str(role).upper()}: {text[:MAX_RESULT_CHARS]}")
    out = "\n".join(lines)
    if len(out) > MAX_HISTORY_CHARS:
        half = MAX_HISTORY_CHARS // 2
        out = out[:half] + "\n[... middle of the history omitted ...]\n" + out[-half:]
    return out or "(no steps yet)"


def brief_messages(messages: list[dict]) -> list[dict]:
    return [
        {"role": "system", "content": "You are an agent that has been working on the task below."},
        {"role": "user", "content": f"TASK:\n{task_text(messages)}\n\nWHAT YOU HAVE DONE SO FAR:\n"
                                    f"{render_history(messages)}\n\n{BRIEF_PROMPT}"},
    ]


async def build_view(messages: list[dict], edge_complete, state: State, detector: str = "regex",
                     glossary: list[str] | None = None) -> list[dict]:
    """The messages the cloud sees for one escalation. `edge_complete(messages) -> str` calls the edge model."""
    note = strip_thinking(await edge_complete(brief_messages(messages)))
    state.notes.append(note)
    body = (f"TASK:\n{task_text(messages)}\n\nHANDOFF NOTE FROM THE EDGE AGENT:\n{note}\n\n"
            "Perform the next single step.")
    if detector == "llm":
        llm = LLMDetector(None, glossary)
        spans = llm.parse(await edge_complete(llm.messages(body)), body)
    else:
        spans = RegexDetector(glossary).find(body)
    for s in spans:
        state.table.placeholder(s.text, s.category)
    state.save()
    return [{"role": "system", "content": CLOUD_ROLE_P3}, {"role": "user", "content": state.table.mask(body)}]


def restore_tool_calls(tool_calls: list[dict], state: State) -> list[dict]:
    """Fill placeholders back into the cloud's tool-call arguments, in place."""
    for tc in tool_calls or []:
        fn = (tc or {}).get("function") or {}
        if isinstance(fn.get("arguments"), str):
            fn["arguments"] = state.table.restore(fn["arguments"])
    return tool_calls
