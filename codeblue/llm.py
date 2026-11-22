"""Chat model clients: any OpenAI-compatible endpoint (vLLM, Ollama, llama.cpp, cloud APIs) plus offline mocks.

Every client has `complete(messages, purpose) -> str`. `purpose` says why the call is made
("act", "selfcheck", "brief", "detect", "rescue"); real clients ignore it, mocks and logs use it.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Protocol


class ChatModel(Protocol):
    name: str

    def complete(self, messages: list[dict], purpose: str = "act") -> str: ...


class OpenAICompatModel:
    """POST {base_url}/chat/completions. Works with vLLM, Ollama (/v1), llama.cpp server, OpenRouter, etc."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key_env: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: float = 180,
        extra_body: dict | None = None,
        extra_headers: dict | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = model
        self.api_key_env = api_key_env
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.extra_body = extra_body or {}
        self.extra_headers = extra_headers or {}

    def complete(self, messages: list[dict], purpose: str = "act") -> str:
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            **self.extra_body,
        }
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key_env:
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"environment variable {self.api_key_env} is not set")
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), headers=headers, method="POST"
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"].get("content") or ""


class LoggedModel:
    """Wraps a model and appends every call to a JSONL file. Used for LOCAL traffic, kept apart from the cloud log."""

    def __init__(self, inner: ChatModel, log_path: str | Path, context: dict | None = None):
        self.inner = inner
        self.name = inner.name
        self.log_path = Path(log_path)
        self.context = context or {}

    def complete(self, messages: list[dict], purpose: str = "act") -> str:
        t0 = time.time()
        out = self.inner.complete(messages, purpose)
        rec = {"ts": t0, "secs": round(time.time() - t0, 3), "purpose": purpose, "model": self.name,
               **self.context, "request": messages, "response": out}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        return out


# --------------------------------------------------------------------------------------------
# Mocks. They are deterministic and read only what is in the messages they receive, so the
# packet format really decides what the mock cloud can do. They exist to test the plumbing,
# not to estimate how real models behave.
# --------------------------------------------------------------------------------------------

_STEP_HDR = re.compile(r"^\[(\d+)\] (local|cloud)$", re.M)
_ACTION = re.compile(r"^ACTION: (.*)$", re.M)


def _all_text(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def _observations(text: str) -> list[str]:
    """Pull out the blocks that follow 'OBSERVATION:' up to the next step header."""
    out = []
    for m in re.finditer(r"OBSERVATION:\n?(.*?)(?=^\[\d+\] (?:local|cloud)$|\Z)", text, re.S | re.M):
        out.append(m.group(1))
    return out


class MockLocalModel:
    """A weak local agent. It follows each task's `mock.local_script`, then repeats its last action (stuck)."""

    name = "mock-local"

    def __init__(self, tasks):
        self.tasks = list(tasks)

    def _task(self, text: str):
        for t in self.tasks:
            if t.goal in text:
                return t
        return None

    def complete(self, messages: list[dict], purpose: str = "act") -> str:
        text = _all_text(messages)
        if purpose == "selfcheck":
            actions = _ACTION.findall(text)
            return "STUCK" if len(actions) >= 2 and actions[-1] == actions[-2] else "PROGRESSING"
        if purpose == "detect":
            from codeblue.detect import RegexDetector

            spans = RegexDetector().find(messages[-1]["content"])
            return json.dumps([{"text": s.text, "type": s.category} for s in spans])
        if purpose == "brief":
            task = self._task(text)
            obs = _observations(text)
            files = sorted(set(re.findall(r'"path": "([^"]+)"', text)))
            last = obs[-1].strip()[:400] if obs else "(none)"
            goal = task.goal if task else "(unknown)"
            return f"Task: {goal}\nFiles opened: {', '.join(files) or 'none'}\nI keep failing.\nOBSERVATION:\n{last}"
        task = self._task(text)
        script = task.mock.get("local_script", []) if task else []
        n_local = sum(1 for _, actor in _STEP_HDR.findall(text) if actor == "local")
        if not script:
            return '{"thought": "no idea", "tool": "list_files", "args": {}}'
        act = script[min(n_local, len(script) - 1)]
        return json.dumps({"thought": "following my plan", **act})


class MockCloudModel:
    """A strong helper that can only use what the packet shows it.

    It recognises the task only if the goal text is visible. Then: if a visible observation contains
    the task's solution lines, it submits them; if the solution file path is visible, it reads it;
    otherwise it lists files. Placeholders in what it saw are copied into its answer, so the local
    re-insertion step is exercised.
    """

    name = "mock-cloud"

    def __init__(self, tasks):
        self.tasks = [t for t in tasks if t.mock]

    def complete(self, messages: list[dict], purpose: str = "rescue") -> str:
        text = _all_text(messages)
        task = next((t for t in self.tasks if t.goal in text), None)
        if task is None:
            return json.dumps({"thought": "I cannot tell what the task is", "tool": "list_files", "args": {}})
        h = task.mock
        obs = "\n".join(_observations(text))
        pat = h.get("solution_line_pattern")
        lines = [ln.strip() for ln in obs.splitlines() if pat and re.search(pat, ln, re.I)]
        if lines:
            answer = " ".join(dict.fromkeys(lines))
            return json.dumps({"thought": "found it", "tool": "submit", "args": {"answer": answer}})
        f = h.get("solution_file")
        if f and f in text:
            return json.dumps({"thought": "read the right file", "tool": "read_file", "args": {"path": f}})
        return json.dumps({"thought": "need to see the files", "tool": "list_files", "args": {}})
