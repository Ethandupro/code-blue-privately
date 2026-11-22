"""Rescue packets: what the cloud model is shown when the local agent asks for help.

P0  full history       the local agent's full context, verbatim (EC2-Bench-style baseline)
P1  brief              a handoff note written by the LOCAL model, plus the goal
P2  blanked            full history with detected sensitive spans replaced by [REDACTED]
P3  placeholders       full history with consistent typed placeholders (<NAME_1>, <MONEY_2>, ...).
                       The mapping stays on the device; placeholders in the cloud's reply are swapped
                       back locally before the action runs.
P4  structure only     task category, tool list and an action skeleton (no content at all)

Add a new packet by subclassing PacketBuilder and registering it in BUILDERS.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field

from codeblue.agent import BRIEF_PROMPT, Step, local_messages, render_history, strip_thinking
from codeblue.dataroom import Task
from codeblue.env import tools_text

CLOUD_SYSTEM = f"""You are an expert assistant helping a smaller agent that is stuck on a task in a company data room.
You act directly for the next step. Tools:
{tools_text()}

Some values may be hidden as [REDACTED] or replaced by placeholders such as <NAME_1> or <MONEY_2>.
Placeholders are consistent: the same placeholder always means the same value. You may use placeholders
in your arguments and answers; they will be filled in locally.

Reply with exactly ONE JSON object and nothing else:
{{"thought": "<short reasoning>", "tool": "<tool name>", "args": {{...}}}}"""


class PlaceholderTable:
    """Consistent value <-> placeholder mapping for one episode. Never leaves the device."""

    PH = re.compile(r"<\s*([A-Z]+_\d+)\s*>")

    def __init__(self):
        self.to_ph: dict[str, str] = {}
        self.to_value: dict[str, str] = {}
        self.counts: Counter = Counter()

    def placeholder(self, value: str, category: str) -> str:
        key = value.strip()
        if key.lower() in self._lower:
            return self.to_ph[self._lower[key.lower()]]
        self.counts[category] += 1
        ph = f"<{category}_{self.counts[category]}>"
        self.to_ph[key] = ph
        self.to_value[ph[1:-1]] = key
        return ph

    @property
    def _lower(self) -> dict[str, str]:
        return {k.lower(): k for k in self.to_ph}

    def restore(self, text: str) -> str:
        return self.PH.sub(lambda m: self.to_value.get(m.group(1), m.group(0)), text)

    def restore_obj(self, obj):
        if isinstance(obj, str):
            return self.restore(obj)
        if isinstance(obj, dict):
            return {k: self.restore_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.restore_obj(v) for v in obj]
        return obj


def replace_values(text: str, values: dict[str, str]) -> str:
    """Replace every occurrence of each value (case-insensitive, longest first) with its replacement."""
    if not values:
        return text
    keys = sorted(values, key=len, reverse=True)
    lower = {k.lower(): v for k, v in values.items()}
    pat = re.compile("|".join(rf"(?<!\w){re.escape(k)}(?!\w)" for k in keys), re.I)
    return pat.sub(lambda m: lower[m.group(0).lower()], text)


@dataclass
class Packet:
    kind: str
    messages: list[dict]
    table: PlaceholderTable | None = None
    info: dict = field(default_factory=dict)

    def restore_action(self, action: dict) -> dict:
        return self.table.restore_obj(action) if self.table else action


@dataclass
class RescueContext:
    """Per-episode state the builders may use. `table` persists across rescues in one episode."""

    detector: object | None = None
    local_model: object | None = None
    table: PlaceholderTable = field(default_factory=PlaceholderTable)
    max_obs_chars: int = 2000


class PacketBuilder:
    name = "base"

    def build(self, task: Task, steps: list[Step], ctx: RescueContext) -> Packet:
        raise NotImplementedError

    @staticmethod
    def _messages(body: str) -> list[dict]:
        return [{"role": "system", "content": CLOUD_SYSTEM}, {"role": "user", "content": body}]

    @staticmethod
    def _full_body(task: Task, steps: list[Step], ctx: RescueContext) -> str:
        return (
            f"GOAL: {task.goal}\n\nHISTORY OF THE STUCK AGENT:\n{render_history(steps, ctx.max_obs_chars)}\n\n"
            "Give the next action."
        )


class P0FullHistory(PacketBuilder):
    name = "P0"

    def build(self, task, steps, ctx):
        return Packet(self.name, self._messages(self._full_body(task, steps, ctx)))


class P1Brief(PacketBuilder):
    name = "P1"

    def build(self, task, steps, ctx):
        msgs = local_messages(task, steps, ctx.max_obs_chars)
        msgs[-1] = {"role": "user", "content": msgs[-1]["content"].rsplit("\n\nWhat is your next action?", 1)[0]
                    + "\n\n" + BRIEF_PROMPT}
        brief = strip_thinking(ctx.local_model.complete(msgs, "brief"))
        body = f"GOAL: {task.goal}\n\nHANDOFF NOTE FROM THE STUCK AGENT:\n{brief}\n\nGive the next action."
        return Packet(self.name, self._messages(body), info={"brief": brief})


class P2Blanked(PacketBuilder):
    name = "P2"
    MASK = "[REDACTED]"

    def build(self, task, steps, ctx):
        body = self._full_body(task, steps, ctx)
        spans = ctx.detector.find(body)
        body = replace_values(body, {s.text: self.MASK for s in spans})
        return Packet(self.name, self._messages(body), info={"n_spans": len(spans)})


class P3Placeholders(PacketBuilder):
    name = "P3"

    def build(self, task, steps, ctx):
        body = self._full_body(task, steps, ctx)
        spans = ctx.detector.find(body)
        for s in spans:
            ctx.table.placeholder(s.text, s.category)
        # Also replace values seen in earlier rescues so placeholders stay consistent.
        body = replace_values(body, dict(ctx.table.to_ph))
        return Packet(self.name, self._messages(body), table=ctx.table, info={"n_spans": len(spans)})


class P4StructureOnly(PacketBuilder):
    name = "P4"

    def build(self, task, steps, ctx):
        lines = []
        for s in steps:
            arg_shape = {k: f"<{type(v).__name__}, {len(str(v))} chars>" for k, v in s.action.get("args", {}).items()}
            status = "error" if s.error else "ok"
            lines.append(f"[{s.index}] {s.actor}: {s.action.get('tool')}({json.dumps(arg_shape)}) -> "
                         f"{status}, {len(s.observation)} chars")
        body = (
            f"TASK CATEGORY: {task.category or 'unspecified'}\n"
            "The goal and all file contents are withheld. Only the shape of the agent's actions is shown.\n\n"
            "ACTION SKELETON:\n" + ("\n".join(lines) or "(no steps yet)") + "\n\nGive the next action."
        )
        return Packet(self.name, self._messages(body))


BUILDERS: dict[str, type[PacketBuilder]] = {
    b.name: b for b in (P0FullHistory, P1Brief, P2Blanked, P3Placeholders, P4StructureOnly)
}


def make_builder(name: str) -> PacketBuilder:
    try:
        return BUILDERS[name]()
    except KeyError:
        raise ValueError(f"unknown packet {name!r}; choose from {sorted(BUILDERS)}") from None
