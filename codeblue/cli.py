"""Command line: run, ping, scan, proxy, matched-rate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="codeblue", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run an experiment grid from a TOML config")
    r.add_argument("--config", default="configs/mock.toml")
    r.add_argument("--out", default=None, help="output folder (default runs/<timestamp>)")

    g = sub.add_parser("ping", help="check the [local] model endpoint (e.g. a vLLM server through an SSH tunnel)")
    g.add_argument("--config", required=True)
    g.add_argument("--timeout", type=float, default=60)

    s = sub.add_parser("scan", help="scan a cloud log for secrets")
    s.add_argument("log")
    s.add_argument("--room", default="data_rooms/harbourline")
    s.add_argument("--task", default=None, help="task id to use when the log has no task ids")

    p = sub.add_parser("proxy", help="HTTP logging proxy for cloud traffic from external harnesses")
    p.add_argument("--upstream", required=True, help="e.g. https://openrouter.ai/api/v1")
    p.add_argument("--log", default="runs/proxy/cloud_log.jsonl")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--api-key-env", default=None, help="env var holding the upstream API key")

    m = sub.add_parser("matched-rate", help="random-trigger rate that matches the self-check rate")
    m.add_argument("episodes", help="episodes.jsonl from a run")

    a = ap.parse_args(argv)
    if a.cmd == "run":
        from codeblue.config import load_config
        from codeblue.runner import run_experiment

        try:
            run_experiment(load_config(a.config), a.out)
        except ValueError as e:
            if "[local]" not in str(e):
                raise
            print(f"FAIL: {e}")
            return 1
    elif a.cmd == "ping":
        from codeblue.config import load_config
        from codeblue.endpoints import check_local_section, ping_local

        cfg = load_config(a.config)
        local = cfg.get("local", {})
        try:
            check_local_section(local, cfg.get("cloud"))
            res = ping_local(local, a.timeout)
        except (RuntimeError, ValueError) as e:
            print(f"FAIL: {e}")
            return 1
        print(f"endpoint {local['base_url']} serves: {', '.join(res['models']) or '(nothing)'}")
        if not res["model_listed"]:
            print(f"FAIL: config asks for {local['model']!r}, which the server does not list")
            return 1
        print(f"reply in {res['secs']} s: {res['reply']!r}")
        print("OK: local traffic goes to local_log.jsonl only; it is never scanned as cloud exposure")
    elif a.cmd == "scan":
        from codeblue.dataroom import load_data_room
        from codeblue.gateway import read_cloud_requests
        from codeblue.scan import scan_log

        room = load_data_room(a.room)
        eps = {r.get("episode_id", "unknown") for r in read_cloud_requests(a.log)}
        mapping = {e: a.task for e in eps} if a.task else None
        print(json.dumps(scan_log(room, Path(a.log), mapping), indent=2))
    elif a.cmd == "proxy":
        from codeblue.proxy_server import make_server

        srv = make_server(a.upstream, a.log, a.host, a.port, a.api_key_env)
        print(f"logging cloud proxy on http://{a.host}:{a.port}/v1 -> {a.upstream}; log: {a.log}")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    elif a.cmd == "matched-rate":
        from codeblue.runner import matched_rate

        print(f"{matched_rate(a.episodes):.4f}")
    return 0
