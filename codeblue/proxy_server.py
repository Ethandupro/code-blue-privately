"""HTTP logging proxy for cloud traffic from harnesses we don't control (e.g. OpenHands in
TheAgentCompany, or EC2-Bench's runtime).

Point the harness's CLOUD base URL at http://127.0.0.1:<port>/v1. Every POST body is logged to the
cloud log before being forwarded upstream. Tag episodes with the `X-CodeBlue-Episode` header if the
harness lets you; otherwise records go under "unknown" and you split them by time.
Keep local-model and simulated-colleague traffic pointed elsewhere so the cloud log stays clean.
Streaming is not supported: requests with "stream": true are rejected with HTTP 400.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock


def make_server(upstream: str, log_path: str | Path, host: str = "127.0.0.1", port: int = 8787,
                api_key_env: str | None = None, timeout: float = 300) -> ThreadingHTTPServer:
    upstream = upstream.rstrip("/")
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lock = Lock()

    def write(rec: dict) -> None:
        with lock, log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                payload = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                payload = {"_raw": raw.decode("utf-8", "replace")}
            if isinstance(payload, dict) and payload.get("stream"):
                self._send(400, b'{"error": "codeblue proxy does not support streaming"}')
                return
            call_id = uuid.uuid4().hex[:12]
            meta = {"episode_id": self.headers.get("X-CodeBlue-Episode", "unknown"),
                    "task_id": self.headers.get("X-CodeBlue-Task"), "path": self.path, "via": "http-proxy"}
            request = payload.get("messages", payload) if isinstance(payload, dict) else payload
            write({"type": "request", "call_id": call_id, "ts": time.time(),
                   "model": payload.get("model") if isinstance(payload, dict) else None, **meta, "request": request})

            path = self.path
            if upstream.endswith("/v1") and path.startswith("/v1/"):
                path = path[3:]
            headers = {"Content-Type": "application/json"}
            if api_key_env and os.environ.get(api_key_env):
                headers["Authorization"] = f"Bearer {os.environ[api_key_env]}"
            elif self.headers.get("Authorization"):
                headers["Authorization"] = self.headers["Authorization"]
            req = urllib.request.Request(upstream + path, data=raw, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body, code = resp.read(), resp.status
            except urllib.error.HTTPError as e:
                body, code = e.read(), e.code
            except Exception as e:  # noqa: BLE001
                write({"type": "response", "call_id": call_id, "ts": time.time(), **meta, "error": repr(e)})
                self._send(502, json.dumps({"error": repr(e)}).encode())
                return
            try:
                parsed = json.loads(body)
                text = parsed["choices"][0]["message"].get("content")
            except Exception:  # noqa: BLE001
                text = body.decode("utf-8", "replace")[:2000]
            write({"type": "response", "call_id": call_id, "ts": time.time(), **meta, "status": code, "response": text})
            self._send(code, body)

    return ThreadingHTTPServer((host, port), Handler)
