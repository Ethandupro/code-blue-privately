"""When does the local agent ask the cloud for help?

- selfcheck: AgentCollab-style, used for every rescue in the main study. The local model judges its own
  progress after each step (PROGRESSING / STUCK), backed by simple stall signals (repeated action,
  consecutive errors).
- never:     local-only baseline (the agent with no rescue at all).
Parked triggers (random control, always = cloud-only) are in codeblue/optional/triggers_extra.py.
Optional caps (max rescues, cooldown) are applied by the loop, identically for every trigger, and every
firing they block is logged.
"""

from __future__ import annotations

import json

from codeblue.agent import SELFCHECK_PROMPT, Step, local_messages, strip_thinking


def _action_key(s: Step) -> str:
    return json.dumps([s.action.get("tool"), s.action.get("args", {})], sort_keys=True)


class Trigger:
    name = "base"
    capped = True

    def reset(self, episode_seed: str) -> None:
        pass

    def check(self, task, steps: list[Step], local_model) -> tuple[bool, str]:
        raise NotImplementedError


class NeverTrigger(Trigger):
    name = "never"

    def check(self, task, steps, local_model):
        return False, ""


class SelfCheckTrigger(Trigger):
    name = "selfcheck"

    def __init__(self, repeat_threshold: int = 2, error_threshold: int = 2, ask_model: bool = True):
        self.repeat_threshold = repeat_threshold
        self.error_threshold = error_threshold
        self.ask_model = ask_model

    def check(self, task, steps, local_model):
        local = [s for s in steps if s.actor == "local"]
        if not local:
            return False, ""
        tail = steps[-self.repeat_threshold:]
        if len(tail) == self.repeat_threshold and len({_action_key(s) for s in tail}) == 1:
            return True, f"repeated action x{self.repeat_threshold}"
        errs = steps[-self.error_threshold:]
        if len(errs) == self.error_threshold and all(s.error for s in errs):
            return True, f"{self.error_threshold} errors in a row"
        if self.ask_model:
            msgs = local_messages(task, steps)
            msgs[-1] = {"role": "user", "content": msgs[-1]["content"].rsplit("\n\nWhat is your next action?", 1)[0]
                        + "\n\n" + SELFCHECK_PROMPT}
            reply = strip_thinking(local_model.complete(msgs, "selfcheck")).upper()
            if "STUCK" in reply:
                return True, "self-check said STUCK"
        return False, ""


def make_trigger(name: str, cfg: dict | None = None, seed: int = 0) -> Trigger:
    cfg = cfg or {}
    if name == "selfcheck":
        return SelfCheckTrigger(cfg.get("repeat_threshold", 2), cfg.get("error_threshold", 2), cfg.get("ask_model", True))
    if name == "never":
        return NeverTrigger()
    if name in ("random", "always"):
        from codeblue.optional.triggers_extra import make_extra_trigger

        return make_extra_trigger(name, cfg, seed)
    raise ValueError(f"unknown trigger {name!r}")
