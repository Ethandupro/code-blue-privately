"""Parked triggers, not part of the main study (see optional/README.md).

- random: control condition. Escalate with fixed probability per step; set the rate to match the
          self-check condition's observed rate with `codeblue matched-rate`.
- always: cloud-only baseline (every step goes to the cloud, never capped).
"""

from __future__ import annotations

import json
import random
import zlib
from pathlib import Path

from codeblue.triggers import Trigger


class AlwaysTrigger(Trigger):
    name = "always"
    capped = False

    def check(self, task, steps, local_model):
        return True, "always"


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


def make_extra_trigger(name: str, cfg: dict, seed: int = 0) -> Trigger:
    if name == "random":
        return RandomTrigger(cfg.get("random_rate", 0.2), seed)
    if name == "always":
        return AlwaysTrigger()
    raise ValueError(f"unknown trigger {name!r}")


def matched_rate(episodes_path: str | Path, trigger: str = "selfcheck") -> float:
    """Rescues per decision point in the given trigger's episodes: the rate to give the random control."""
    rescues = points = 0
    for line in Path(episodes_path).read_text().splitlines():
        e = json.loads(line)
        if e["trigger"] == trigger and e.get("attack", "none") == "none":
            rescues += e["n_rescues"]
            points += e["decision_points"]
    return rescues / points if points else 0.0
