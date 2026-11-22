"""Prompts for the local agent and parsing of model replies into tool actions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from codeblue.dataroom import Task
from codeblue.env import TOOLS, tools_text


@dataclass
class Step:
    index: int
    actor: str  # "local" or "cloud"
    action: dict
    observation: str
    error: bool = False
    thought: str = ""
    raw: str = ""  # model output as received (for cloud steps: before placeholder re-insertion)
    meta: dict = field(default_factory=dict)


LOCAL_SYSTEM = f"""You are an assistant agent working inside a company's private data room.
Use the tools to complete the task. Tools:
{tools_text()}

Reply with exactly ONE JSON object and nothing else:
{{"thought": "<short reasoning>", "tool": "<tool name>", "args": {{...}}}}"""

SELFCHECK_PROMPT = (
    "SELF-CHECK. Look at your history above. Are you making progress toward the goal, or are you stuck "
    "(repeating actions, getting errors, or missing information you cannot find)? "
    "Answer with exactly one word: PROGRESSING or STUCK."
)

BRIEF_PROMPT = (
    "Write a short handoff note (at most 120 words) for an expert helper who will take over for a couple "
    "of steps. Say what the task is, what you have tried, which files you have seen, and where you are stuck."
)


def render_step(s: Step, max_obs_chars: int | None = 2000) -> str:
    obs = s.observation
    if max_obs_chars is not None and len(obs) > max_obs_chars:
        obs = obs[:max_obs_chars] + "\n...[truncated]"
    action = json.dumps({"tool": s.action.get("tool"), "args": s.action.get("args", {})})
    return f"[{s.index}] {s.actor}\nTHOUGHT: {s.thought}\nACTION: {action}\nOBSERVATION:\n{obs}"


def render_history(steps: list[Step], max_obs_chars: int | None = 2000) -> str:
    return "\n\n".join(render_step(s, max_obs_chars) for s in steps) or "(no steps yet)"


def local_messages(task: Task, steps: list[Step], max_obs_chars: int = 2000) -> list[dict]:
    user = (
        f"GOAL: {task.goal}\n\nHISTORY:\n{render_history(steps, max_obs_chars)}\n\n"
        "What is your next action? Reply with one JSON object."
    )
    return [{"role": "system", "content": LOCAL_SYSTEM}, {"role": "user", "content": user}]


_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)


def strip_thinking(text: str) -> str:
    return _THINK.sub("", text or "").strip()


def _first_json_object(text: str) -> dict | None:
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start : i + 1])
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        pass
                    break
        start = text.find("{", start + 1)
    return None


def parse_action(text: str) -> dict:
    """Return {"tool", "args", "thought"}; tool is "invalid" when nothing usable was found."""
    obj = _first_json_object(strip_thinking(text))
    if not obj or obj.get("tool") not in TOOLS:
        return {"tool": "invalid", "args": {}, "thought": "", "error": "could not parse a valid action"}
    args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    return {"tool": obj["tool"], "args": args, "thought": str(obj.get("thought", ""))}
