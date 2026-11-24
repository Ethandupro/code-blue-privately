"""EC2-Bench sensitive values -> Code Blue secret labels, plus our own hand labels.

EC2-Bench (100 of its 128 tasks) ships `gt/sensitive_items.json` in each task workspace:
    {"items": [{"entity_id": ..., "name": ..., "sensitive_fields": [{"field": ..., "value": ..., "dimension": ...}]}]}
(format as read by their utils/privacy_audit.py). Each field becomes one secret with id "<entity_id>:<field>",
matched by our deterministic scanner (exact, letters-and-digits, numbers in other formats). Values too short or
too generic to match safely (under MIN_CHARS characters, or a bare small number) are kept out of the
deterministic score and counted in `unscannable`; EC2-Bench's own LLM judge still covers them.

Our hand labels live in labels/ec2bench.json, one entry per task id:
    {"essential": [secret ids the deliverable needs],
     "attack_vectors": [{"file": "exec/<path>", "channel": "why an outsider could write here"}],
     "composites": [{"id": "...", "components": [secret ids], "note": "what the pieces reveal together"}],
     "has_own_injection": false}
`codeblue ec2 labels-template` writes a skeleton to fill in by hand.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from codeblue.dataroom import DataRoom, Secret, Task

MIN_CHARS = 4
TEXT_SUFFIXES = {".md", ".txt", ".csv", ".json", ".eml", ".py", ".log", ".env", ".yaml", ".yml", ".html", ".tsv",
                 ".xml", ".ini", ".cfg", ".toml", ".sql", ""}


def task_dir_name(task_id: str) -> str:
    """'EC2Bench_task_28_spreadsheet_summary' -> 'task_28_spreadsheet_summary'."""
    return re.sub(r"^EC2Bench_", "", task_id)


def find_workspace(data_root: str | Path, task_id: str) -> Path:
    name = task_dir_name(task_id)
    hits = sorted(Path(data_root).glob(f"workspace/*/{name}")) + sorted(Path(data_root).glob(f"workspace/{name}"))
    if not hits:
        raise FileNotFoundError(f"no workspace folder {name} under {data_root}/workspace/")
    return hits[0]


def _numeric(value: str) -> float | None:
    v = value.replace(",", "").replace("$", "").replace("S$", "").strip()
    try:
        x = float(v)
    except ValueError:
        return None
    return x if abs(x) >= 1000 else None


def secrets_from_items(data: dict) -> tuple[list[Secret], list[str]]:
    secrets, unscannable = [], []
    for item in data.get("items", []):
        eid = str(item.get("entity_id", ""))
        for f in item.get("sensitive_fields") or []:
            sid = f"{eid}:{f.get('field', '')}"
            value = str(f.get("value", "")).strip()
            note = f"{f.get('dimension', '')}; {item.get('name', '')}".strip("; ")
            num = _numeric(value)
            if len(value) < MIN_CHARS or (value.replace(".", "").isdigit() and num is None):
                unscannable.append(sid)
                continue
            secrets.append(Secret(id=sid, kind="fact", value=value, numeric=num, note=note))
    return secrets, unscannable


def read_files(exec_dir: Path, max_bytes: int = 2_000_000) -> dict[str, str]:
    files = {}
    for p in sorted(exec_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES and p.stat().st_size <= max_bytes:
            try:
                files[str(p.relative_to(exec_dir.parent))] = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
    return files


def load_task_room(data_root: str | Path, task_id: str, labels: dict | None = None) -> tuple[DataRoom, dict]:
    """One EC2-Bench task as a DataRoom: its files, its secrets (ours + composites) and one task."""
    ws = find_workspace(data_root, task_id)
    gt = ws / "gt" / "sensitive_items.json"
    if not gt.exists():
        gt = ws / "gt" / "gt" / "sensitive_items.json"
    secrets, unscannable = secrets_from_items(json.loads(gt.read_text(encoding="utf-8"))) if gt.exists() else ([], [])
    lab = (labels or {}).get(task_id, {})
    by_id = {s.id: s for s in secrets}
    for c in lab.get("composites", []):
        missing = [x for x in c["components"] if x not in by_id]
        if missing:
            raise ValueError(f"{task_id}: composite {c['id']} names unknown secrets {missing}")
        by_id[c["id"]] = Secret(id=c["id"], kind="composite", components=list(c["components"]), note=c.get("note", ""))
    essential = [s for s in lab.get("essential", []) if s in by_id]
    task = Task(id=task_id, goal="", essential=essential)
    room = DataRoom(name=task_id, root=ws, files=read_files(ws / "exec") if (ws / "exec").exists() else {},
                    secrets=by_id, tasks={task_id: task}, glossary=[])
    return room, {"annotated": gt.exists(), "unscannable": unscannable, "labelled": bool(lab)}


def labels_template(data_root: str | Path) -> dict:
    out = {}
    for gt in sorted(Path(data_root).glob("workspace/**/gt/sensitive_items.json")):
        ws = gt.parent.parent if gt.parent.name == "gt" and gt.parent.parent.name != "gt" else gt.parent.parent.parent
        task_id = f"EC2Bench_{ws.name}"
        secrets, unscannable = secrets_from_items(json.loads(gt.read_text(encoding="utf-8")))
        out[task_id] = {
            "essential": [],
            "attack_vectors": [],
            "composites": [],
            "has_own_injection": "inject" in ws.name,
            "_secrets": {s.id: s.value[:80] for s in secrets},
            "_unscannable": unscannable,
            "_files": sorted(read_files(ws / "exec"))[:60] if (ws / "exec").exists() else [],
        }
    return out
