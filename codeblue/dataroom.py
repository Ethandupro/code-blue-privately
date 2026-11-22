"""Load a synthetic company data room: files, labelled secrets, tasks and the local glossary."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Secret:
    id: str
    kind: str  # "canary" (exact string), "fact" (paraphrasable), "composite" (built from other secrets)
    value: str = ""
    variants: list[str] = field(default_factory=list)
    keywords: list[list[str]] = field(default_factory=list)  # all groups must match, any term per group
    numeric: float | None = None
    tolerance: float = 0.005  # relative tolerance for numeric matches
    components: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class Task:
    id: str
    goal: str
    category: str = ""
    essential: list[str] = field(default_factory=list)
    answer_must_include: list[str] = field(default_factory=list)  # each item may hold "a|b|c" alternatives
    max_steps: int = 12
    mock: dict = field(default_factory=dict)  # only used by the mock models


@dataclass
class DataRoom:
    name: str
    root: Path
    files: dict[str, str]
    secrets: dict[str, Secret]
    tasks: dict[str, Task]
    glossary: list[str]

    def bystanders(self, task: Task) -> list[str]:
        return [sid for sid in self.secrets if sid not in task.essential]


def load_data_room(path: str | Path) -> DataRoom:
    root = Path(path)
    files_dir = root / "files"
    files = {
        p.relative_to(files_dir).as_posix(): p.read_text(encoding="utf-8")
        for p in sorted(files_dir.rglob("*"))
        if p.is_file()
    }
    secrets = {s["id"]: Secret(**s) for s in json.loads((root / "secrets.json").read_text(encoding="utf-8"))}
    tasks = {t["id"]: Task(**t) for t in json.loads((root / "tasks.json").read_text(encoding="utf-8"))}
    glossary_path = root / "glossary.json"
    glossary = json.loads(glossary_path.read_text(encoding="utf-8"))["names"] if glossary_path.exists() else []

    for t in tasks.values():
        unknown = [s for s in t.essential if s not in secrets]
        if unknown:
            raise ValueError(f"task {t.id} lists unknown essential secrets {unknown}")
    for s in secrets.values():
        missing = [c for c in s.components if c not in secrets]
        if missing:
            raise ValueError(f"composite {s.id} lists unknown components {missing}")
    return DataRoom(root.name, root, files, secrets, tasks, glossary)
