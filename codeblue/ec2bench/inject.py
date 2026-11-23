"""Make an attacked copy of an EC2-Bench data root: the injection appended to each task's labelled files.

EC2-Bench's runner takes `--data-root`, so the attacked arm simply points at the copy. Only tasks that have
`attack_vectors` in labels/ec2bench.json and do not already contain their own injection are changed; the
others are copied unchanged and should be left out of the attacked arm (`codeblue ec2 collect` does this).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from codeblue.attacks import get_attack
from codeblue.ec2bench.labels import find_workspace


def attacked_tasks(labels: dict) -> list[str]:
    return [t for t, lab in labels.items() if lab.get("attack_vectors") and not lab.get("has_own_injection")]


def make_attacked_root(data_root: str | Path, labels: dict, out_root: str | Path,
                       attack_name: str = "stuck_and_stuff") -> dict[str, list[str]]:
    attack = get_attack(attack_name)
    src, dst = Path(data_root), Path(out_root)
    if dst.exists():
        raise FileExistsError(f"{dst} exists; choose a new folder")
    shutil.copytree(src, dst)
    planted: dict[str, list[str]] = {}
    for task_id in attacked_tasks(labels):
        ws = find_workspace(dst, task_id)
        for v in labels[task_id]["attack_vectors"]:
            p = ws / v["file"]
            if not p.is_file():
                raise FileNotFoundError(f"{task_id}: attack vector {v['file']} not found in {ws}")
            text = attack.render(v["file"])
            body = p.read_text(encoding="utf-8").rstrip("\n")
            p.write_text(f"{text}\n\n{body}\n" if v.get("position") == "start" else f"{body}\n\n{text}\n",
                         encoding="utf-8")
            planted.setdefault(task_id, []).append(v["file"])
    (dst / "codeblue_attack.json").write_text(json.dumps({"attack": attack_name, "planted": planted}, indent=2))
    return planted
