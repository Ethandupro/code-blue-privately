"""Parked packets, not part of the main study (see optional/README.md).

P2  blanked brief       the P1 brief with detected spans replaced by [REDACTED]
P4  structure only      task category, tool names and an action skeleton; no content
P2h blanked history     P0 with [REDACTED]
P3h placeholder history P0 with placeholders
"""

from __future__ import annotations

import json

from codeblue.packets import (BUILDERS, Packet, PacketBuilder, _blank, _brief_body, _local_brief,
                              _placeholder)


class P2Blanked(PacketBuilder):
    """The P1 brief, blanked (the proposal's P2)."""

    name = "P2"
    MASK = "[REDACTED]"

    def build(self, task, steps, ctx):
        brief = _local_brief(task, steps, ctx)
        body, n = _blank(_brief_body(task, brief), ctx)
        return Packet(self.name, self._messages(body), info={"n_spans": n, "brief": brief})


class P2hBlankedHistory(PacketBuilder):
    """Ablation: the full history (P0), blanked."""

    name = "P2h"

    def build(self, task, steps, ctx):
        body, n = _blank(self._full_body(task, steps, ctx), ctx)
        return Packet(self.name, self._messages(body), info={"n_spans": n})


class P3hPlaceholderHistory(PacketBuilder):
    """Ablation: the full history (P0) with placeholders."""

    name = "P3h"

    def build(self, task, steps, ctx):
        body, n = _placeholder(self._full_body(task, steps, ctx), ctx)
        return Packet(self.name, self._messages(body), table=ctx.table, info={"n_spans": n})


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


for _b in (P2Blanked, P4StructureOnly, P2hBlankedHistory, P3hPlaceholderHistory):
    BUILDERS.setdefault(_b.name, _b)
