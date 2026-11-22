"""When does the local agent ask the cloud for help?

- selfcheck: AgentCollab-style. The local model judges its own progress after each step
  (PROGRESSING / STUCK), backed by simple stall signals (repeated action, consecutive errors).
- random:    control condition. Escalate with fixed probability per step; set the rate to match the
             self-check condition's observed rate (see `codeblue matched-rate`).
- never:     local-only baseline.
- always:    cloud-only baseline (every step goes to the cloud).
Caps (max rescues, cooldown) are applied by the loop, identically for every trigger.
"""

from __future__ import annotations

import json
import random
import zlib

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


class AlwaysTrigger(Trigger):
    name = "always"
    capped = False

    def check(self, task, steps, local_model):
        return True, "always"


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


class RandomTrigger(Trigger):
    name = "random"

    def __init__(self, rate: float = 0.2, seed: int = 0):
        self.rate = rate
        self.seed = seed
        self.rng = random.Random(seed)

    def reset(self, episode_seed: str) -> None:
        # Deterministic per (seed, episode) and independent of run order.
        self.rng = random.Random(zlib.crc32(f"{self.seed}:{episode_seed}".encode()))

    def check(self, task, steps, local_model):
        if not steps:
            return False, ""
        return (self.rng.random() < self.rate), f"random p={self.rate}"


def make_trigger(name: str, cfg: dict | None = None, seed: int = 0) -> Trigger:
    cfg = cfg or {}
    if name == "selfcheck":
        return SelfCheckTrigger(cfg.get("repeat_threshold", 2), cfg.get("error_threshold", 2), cfg.get("ask_model", True))
    if name == "random":
        return RandomTrigger(cfg.get("random_rate", 0.2), seed)
    if name == "never":
        return NeverTrigger()
    if name == "always":
        return AlwaysTrigger()
    raise ValueError(f"unknown trigger {name!r}")
