"""One episode: the local agent works; on a trigger the cloud takes over for a few steps, then hands back."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from codeblue.agent import Step, local_messages, parse_action
from codeblue.dataroom import DataRoom, Task
from codeblue.env import DataRoomEnv
from codeblue.packets import PacketBuilder, RescueContext


@dataclass
class LoopConfig:
    """Rescue settings. By default nothing limits how often the agent may ask for help except the task's
    step budget (max_steps), which is the same for every condition; the number of rescues is measured.

    max_rescues and cooldown are optional limits for sensitivity runs. When one is set, the trigger is
    still asked at every step, and each firing the limit blocks is recorded in the episode's
    `blocked_firings`, so a cap never hides how often the agent wanted help.
    """

    takeover_steps: int = 2  # cloud steps per rescue before handing back (the "short rescue")
    max_rescues: int | None = None  # None: no cap
    cooldown: int = 0  # local steps after a rescue before another may start; 0: none
    max_steps: int | None = None  # None: use the task's max_steps
    max_obs_chars: int = 2000


def _run_action(env: DataRoomEnv, steps: list[Step], actor: str, action: dict, raw: str, meta: dict | None = None):
    if action["tool"] == "invalid":
        obs_text, err = "ERROR: could not parse an action; reply with one JSON object", True
    else:
        obs = env.execute(action)
        obs_text, err = obs.text, obs.error
    steps.append(Step(len(steps) + 1, actor, {"tool": action["tool"], "args": action.get("args", {})},
                      obs_text, err, action.get("thought", ""), raw, meta or {}))


def run_episode(
    episode_id: str,
    room: DataRoom,
    task: Task,
    local_model,
    gateway,
    builder: PacketBuilder | None,
    trigger,
    detector=None,
    cfg: LoopConfig | None = None,
) -> dict:
    cfg = cfg or LoopConfig()
    max_steps = cfg.max_steps or task.max_steps
    env = DataRoomEnv(room, task)
    ctx = RescueContext(detector=detector, local_model=local_model, max_obs_chars=cfg.max_obs_chars)
    trigger.reset(episode_id)
    steps: list[Step] = []
    rescues: list[dict] = []
    blocked: list[dict] = []
    decision_points = 0
    last_rescue_end = -10**9

    while len(steps) < max_steps and not env.done:
        if builder is not None:
            limit = None
            if trigger.capped and cfg.max_rescues is not None and len(rescues) >= cfg.max_rescues:
                limit = "max_rescues"
            elif trigger.capped and len(steps) - last_rescue_end < cfg.cooldown:
                limit = "cooldown"
            if limit is None:
                decision_points += 1
            fire, reason = trigger.check(task, steps, local_model)
            if fire and limit is not None:
                blocked.append({"at_step": len(steps), "reason": reason, "blocked_by": limit})
            elif fire:
                r = {"index": len(rescues), "at_step": len(steps), "reason": reason, "cloud_steps": 0}
                rescues.append(r)
                for k in range(cfg.takeover_steps):
                    if env.done or len(steps) >= max_steps:
                        break
                    packet = builder.build(task, steps, ctx)
                    raw = gateway.complete(packet.messages, {
                        "episode_id": episode_id, "task_id": task.id, "packet": packet.kind,
                        "trigger": trigger.name, "rescue": r["index"], "takeover_step": k, "at_step": len(steps)})
                    action = packet.restore_action(parse_action(raw))
                    _run_action(env, steps, "cloud", action, raw, {"rescue": r["index"]})
                    r["cloud_steps"] += 1
                last_rescue_end = len(steps)
                continue
        raw = local_model.complete(local_messages(task, steps, cfg.max_obs_chars), "act")
        _run_action(env, steps, "local", parse_action(raw), raw)

    return {
        "episode_id": episode_id,
        "task_id": task.id,
        "packet": builder.name if builder else None,
        "trigger": trigger.name,
        "score": env.score(),
        "done": env.done,
        "answer": env.answer,
        "n_steps": len(steps),
        "n_local_steps": sum(s.actor == "local" for s in steps),
        "n_cloud_steps": sum(s.actor == "cloud" for s in steps),
        "n_rescues": len(rescues),
        "n_blocked_firings": len(blocked),
        "decision_points": decision_points,
        "rescues": rescues,
        "blocked_firings": blocked,
        "steps": [asdict(s) for s in steps],
    }
