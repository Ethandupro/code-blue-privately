"""The only way cloud-bound traffic leaves the harness. Every request is written to the cloud log
BEFORE it is sent, so failed calls are still counted as exposure.

The cloud log holds cloud traffic only. Local-model traffic goes to a separate log (see LoggedModel),
and any other LLM traffic (e.g. simulated colleagues in a benchmark) must not be routed through here.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


class CloudGateway:
    def __init__(self, model, log_path: str | Path):
        self.model = model
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, rec: dict) -> None:
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def complete(self, messages: list[dict], meta: dict) -> str:
        call_id = uuid.uuid4().hex[:12]
        self._write({"type": "request", "call_id": call_id, "ts": time.time(), "model": self.model.name,
                     **meta, "request": messages})
        try:
            out = self.model.complete(messages, "rescue")
        except Exception as e:  # noqa: BLE001 - we log and re-raise
            self._write({"type": "response", "call_id": call_id, "ts": time.time(), **meta, "error": repr(e)})
            raise
        self._write({"type": "response", "call_id": call_id, "ts": time.time(), **meta, "response": out})
        return out


def read_cloud_requests(log_path: str | Path) -> list[dict]:
    """All request records, in the order they were sent."""
    recs = []
    with Path(log_path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("type") == "request":
                    recs.append(r)
    return recs


def request_text(rec: dict) -> str:
    req = rec.get("request")
    if isinstance(req, list):
        parts = []
        for m in req:
            c = m.get("content", "") if isinstance(m, dict) else m
            if isinstance(c, list):  # multi-part content
                c = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
            parts.append(str(c))
        return "\n".join(parts)
    return json.dumps(req)
