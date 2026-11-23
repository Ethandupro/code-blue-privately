"""Command line: run, claim, ping, scan; proxy and matched-rate are optional extras."""

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

    c = sub.add_parser("claim", help="test the main claim (P3 vs P0) on one or more run folders, pooled")
    c.add_argument("runs", nargs="+", help="run folders (each holds episodes.jsonl and leakage.json)")
    c.add_argument("--baseline", default="P0")
    c.add_argument("--method", default="P3")

    e = sub.add_parser("ec2", help="EC2-Bench: patch a checkout, label tasks, plant the injection, collect results")
    esub = e.add_subparsers(dest="ec2_cmd", required=True)
    e1 = esub.add_parser("patch", help="let an EC2-Bench checkout send P3 (CODEBLUE_PACKET=P3)")
    e1.add_argument("ec2_root")
    e2 = esub.add_parser("labels-template", help="write a skeleton labels file to fill in by hand")
    e2.add_argument("--data-root", required=True)
    e2.add_argument("--out", default="labels/ec2bench.template.json")
    e3 = esub.add_parser("inject", help="copy a data root with the injection planted in the labelled files")
    e3.add_argument("--data-root", required=True)
    e3.add_argument("--labels", default="labels/ec2bench.json")
    e3.add_argument("--out", required=True)
    e3.add_argument("--attack", default="stuck_and_stuff")
    e4 = esub.add_parser("collect", help="EC2-Bench run folders -> claim.json, rescue curves, mosaic")
    e4.add_argument("--arm", action="append", required=True, metavar="PACKET:ATTACK:PATH",
                    help="e.g. local-only:none:out/edge, P0:none:out/p0, P3:stuck_and_stuff:out/p3-att")
    e4.add_argument("--data-root", required=True)
    e4.add_argument("--labels", default="labels/ec2bench.json")
    e4.add_argument("--out", required=True)

    g = sub.add_parser("ping", help="check the [local] model endpoint (e.g. a vLLM server through an SSH tunnel)")
    g.add_argument("--config", required=True)
    g.add_argument("--cloud", action="store_true",
                   help="check the [cloud] endpoint instead (e.g. a stand-in cloud model served on the same VM)")
    g.add_argument("--timeout", type=float, default=60)

    s = sub.add_parser("scan", help="scan a cloud log for secrets")
    s.add_argument("log")
    s.add_argument("--room", default="data_rooms/harbourline")
    s.add_argument("--task", default=None, help="task id to use when the log has no task ids")

    p = sub.add_parser("proxy", help="(optional) HTTP logging proxy for cloud traffic from external harnesses")
    p.add_argument("--upstream", required=True, help="e.g. https://openrouter.ai/api/v1")
    p.add_argument("--log", default="runs/proxy/cloud_log.jsonl")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--api-key-env", default=None, help="env var holding the upstream API key")

    m = sub.add_parser("matched-rate", help="(optional) random-trigger rate that matches the self-check rate")
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
    elif a.cmd == "claim":
        from codeblue.claim import evaluate_all, load_runs, print_claim

        res = evaluate_all(*load_runs(a.runs), {"baseline": a.baseline, "method": a.method})
        print_claim(res)
        print(json.dumps(res, indent=2))
    elif a.cmd == "ec2":
        return _ec2(a)
    elif a.cmd == "ping":
        from codeblue.config import load_config
        from codeblue.endpoints import check_local_section, ping_local

        cfg = load_config(a.config)
        section = "cloud" if a.cloud else "local"
        target = cfg.get(section, {})
        try:
            check_local_section(cfg.get("local", {}), cfg.get("cloud"))
            if target.get("kind") != "openai":
                raise RuntimeError(f"[{section}] kind is {target.get('kind')!r}; ping only works for kind = \"openai\"")
            res = ping_local(target, a.timeout)
        except (RuntimeError, ValueError) as e:
            print(f"FAIL: {e}")
            return 1
        print(f"[{section}] endpoint {target['base_url']} serves: {', '.join(res['models']) or '(nothing)'}")
        if not res["model_listed"]:
            print(f"FAIL: config asks for {target['model']!r}, which the server does not list")
            return 1
        print(f"reply in {res['secs']} s: {res['reply']!r}")
        if a.cloud:
            print("OK: this ping is not logged; during a run, every [cloud] request goes to cloud_log.jsonl and is scanned")
        else:
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
        from codeblue.optional.proxy_server import make_server

        srv = make_server(a.upstream, a.log, a.host, a.port, a.api_key_env)
        print(f"logging cloud proxy on http://{a.host}:{a.port}/v1 -> {a.upstream}; log: {a.log}")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass
    elif a.cmd == "matched-rate":
        from codeblue.optional.triggers_extra import matched_rate

        print(f"{matched_rate(a.episodes):.4f}")
    return 0


def _ec2(a) -> int:
    def labels(path):
        p = Path(path)
        return json.loads(p.read_text()) if p.exists() else {}

    if a.ec2_cmd == "patch":
        from codeblue.ec2bench.patch import patch

        print("\n".join(patch(a.ec2_root)))
    elif a.ec2_cmd == "labels-template":
        from codeblue.ec2bench.labels import labels_template

        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        t = labels_template(a.data_root)
        out.write_text(json.dumps(t, indent=2, ensure_ascii=False))
        print(f"wrote {out}: {len(t)} annotated tasks; fill in essential, attack_vectors, composites")
    elif a.ec2_cmd == "inject":
        from codeblue.ec2bench.inject import make_attacked_root

        planted = make_attacked_root(a.data_root, labels(a.labels), a.out, a.attack)
        print(f"planted {a.attack} in {sum(map(len, planted.values()))} files across {len(planted)} tasks -> {a.out}")
    elif a.ec2_cmd == "collect":
        from codeblue.claim import print_claim
        from codeblue.ec2bench.collect import collect

        arms = []
        for spec in a.arm:
            packet, attack, path = spec.split(":", 2)
            arms.append((packet, attack, Path(path)))
        res = collect(arms, a.data_root, labels(a.labels), a.out)
        print_claim(res)
        print(f"\ncoverage: {json.dumps(res['coverage'])[:400]}\nwrote {a.out}")
    return 0
