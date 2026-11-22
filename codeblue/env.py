"""Tool environment the agent acts in. Tools only touch the in-memory data room."""

from __future__ import annotations

from dataclasses import dataclass

from codeblue.dataroom import DataRoom, Task

TOOLS = {
    "list_files": "list_files() -> every file path in the data room",
    "read_file": 'read_file(path) -> the file contents, e.g. {"path": "finance/report.csv"}',
    "search": 'search(query) -> up to 10 matching lines across all files, e.g. {"query": "margin"}',
    "submit": 'submit(answer) -> finish the task with your final answer text, e.g. {"answer": "..."}',
}


def tools_text() -> str:
    return "\n".join(f"- {v}" for v in TOOLS.values())


def _norm(s: str) -> str:
    return " ".join(s.lower().replace(",", "").split())


def score_answer(task: Task, answer: str | None) -> float:
    """Fraction of required items present in the answer. Each item may list alternatives as 'a|b'."""
    if not answer or not task.answer_must_include:
        return 0.0
    a = _norm(answer)
    hits = sum(any(_norm(alt) in a for alt in item.split("|")) for item in task.answer_must_include)
    return hits / len(task.answer_must_include)


@dataclass
class Observation:
    text: str
    error: bool = False


class DataRoomEnv:
    def __init__(self, room: DataRoom, task: Task):
        self.room = room
        self.task = task
        self.answer: str | None = None
        self.done = False

    def execute(self, action: dict) -> Observation:
        tool = action.get("tool")
        args = action.get("args") or {}
        if tool == "list_files":
            return Observation("\n".join(self.room.files))
        if tool == "read_file":
            path = str(args.get("path", "")).strip().lstrip("/")
            if path not in self.room.files:
                return Observation(f"ERROR: no such file '{path}'", error=True)
            return Observation(self.room.files[path])
        if tool == "search":
            q = str(args.get("query", "")).lower().strip()
            if not q:
                return Observation("ERROR: empty query", error=True)
            hits = [
                f"{p}: {line.strip()}"
                for p, text in self.room.files.items()
                for line in text.splitlines()
                if q in line.lower()
            ]
            return Observation("\n".join(hits[:10]) or "no matches")
        if tool == "submit":
            self.answer = str(args.get("answer", ""))
            self.done = True
            return Observation("submitted")
        return Observation(f"ERROR: unknown or unparseable action {action!r}", error=True)

    def score(self) -> float:
        return score_answer(self.task, self.answer)
