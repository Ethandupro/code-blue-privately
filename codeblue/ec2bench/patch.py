"""Patch an EC2-Bench checkout so its delegation mode can send P3 instead of the full history (P0).

Tested against EC2-Bench commit 583e8ce (main, 23 Nov 2026). Every edit is an exact string replacement that
fails loudly if their code has changed, and the patch is idempotent. With CODEBLUE_PACKET unset (or P0) the
patched code behaves exactly like the original, so the same checkout runs both arms.

What it changes:
  skills/cloud-router/cb_detect.py   copy of codeblue/detect.py (our regex / LLM detector)
  skills/cloud-router/cb_p3.py       copy of codeblue/ec2bench/p3_view.py
  skills/cloud-router/privacy_proxy.py
      _build_cloud_view  -> returns the P3 view when CODEBLUE_PACKET=P3
      handle_cloud       -> skips their full-history role prompt under P3 (ours is in the view) and fills
                            placeholders back into the cloud's tool-call arguments and text
  eval/run_batch.py
      passes CODEBLUE_PACKET / CODEBLUE_DETECTOR into the container's proxy, and collects
      /tmp/codeblue_p3_table.json with the other delegation artifacts
The edge model's escalate prompt is untouched, so it decides when to escalate exactly as in P0.
"""

from __future__ import annotations

import shutil
from pathlib import Path

MARK = "# codeblue-p3"
HERE = Path(__file__).resolve().parent

PROXY_EDITS = [
    (
        "ROUTER_PORT = int(os.environ.get('ROUTER_PORT', '9303'))",
        "ROUTER_PORT = int(os.environ.get('ROUTER_PORT', '9303'))\n"
        f"CODEBLUE_PACKET = os.environ.get('CODEBLUE_PACKET', 'P0').upper()  {MARK}\n"
        f"CODEBLUE_DETECTOR = os.environ.get('CODEBLUE_DETECTOR', 'regex')  {MARK}\n"
        "if CODEBLUE_PACKET == 'P3':\n"
        "    import cb_p3  " + MARK,
    ),
    (
        "async def _build_cloud_view(session: SessionState) -> list[dict]:\n    return session.original_ctx",
        "async def _build_cloud_view(session: SessionState) -> list[dict]:\n"
        f"    if CODEBLUE_PACKET == 'P3':  {MARK}\n"
        "        async def _edge(msgs):\n"
        "            body = {'model': EDGE_MODEL, 'messages': msgs, 'stream': False, 'max_tokens': 1024}\n"
        "            (resp, _lat) = await _post_upstream_nonstream(EDGE_BASE_URL, EDGE_API_KEY, body)\n"
        "            return ((resp.get('choices') or [{}])[0].get('message') or {}).get('content') or ''\n"
        "        if not hasattr(session, 'cb_state'):\n"
        "            session.cb_state = cb_p3.State()\n"
        "        return await cb_p3.build_view(session.original_ctx, _edge, session.cb_state, detector=CODEBLUE_DETECTOR)\n"
        "    return session.original_ctx",
    ),
    (
        "        if CLOUD_ROLE_PROMPT:\n            body['messages'] = body['messages'] + [{'role': 'system', 'content': CLOUD_ROLE_PROMPT}]",
        f"        if CLOUD_ROLE_PROMPT and CODEBLUE_PACKET != 'P3':  {MARK}\n"
        "            body['messages'] = body['messages'] + [{'role': 'system', 'content': CLOUD_ROLE_PROMPT}]",
    ),
    (
        "    decoded_content = raw_content\n",
        "    decoded_content = raw_content\n"
        f"    if CODEBLUE_PACKET == 'P3' and hasattr(session, 'cb_state'):  {MARK}\n"
        "        decoded_content = session.cb_state.table.restore(decoded_content)\n"
        "        cb_p3.restore_tool_calls(raw_tool_calls, session.cb_state)\n",
    ),
]

RUNNER_EDITS = [
    (
        "for env_name in ('GLIMPSE_ENTROPY_THRESHOLD', 'GLIMPSE_TOP_K', 'EDGE_STREAM_READ_TIMEOUT_S'):",
        "for env_name in ('GLIMPSE_ENTROPY_THRESHOLD', 'GLIMPSE_TOP_K', 'EDGE_STREAM_READ_TIMEOUT_S', "
        "'CODEBLUE_PACKET', 'CODEBLUE_DETECTOR'):",
    ),
    (
        "'collect_artifacts': ['/tmp/cloud_assistant_audit.jsonl', '/tmp/router_proxy.log', '/tmp/router_proxy.stdout', "
        "'/tmp/router_failed_request.json', '/tmp/router_usage.json'], 'tools_also_allow': ['escalate']",
        "'collect_artifacts': ['/tmp/cloud_assistant_audit.jsonl', '/tmp/router_proxy.log', '/tmp/router_proxy.stdout', "
        "'/tmp/router_failed_request.json', '/tmp/router_usage.json', '/tmp/codeblue_p3_table.json'], "
        "'tools_also_allow': ['escalate']",
    ),
]


def _edit(path: Path, edits: list[tuple[str, str]], marker: str) -> bool:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return False
    for old, new in edits:
        n = text.count(old)
        if n == 0:
            raise RuntimeError(f"{path}: expected code not found (EC2-Bench changed?):\n{old[:200]}")
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    return True


def patch(ec2_root: str | Path) -> list[str]:
    root = Path(ec2_root)
    router = root / "skills" / "cloud-router"
    proxy, runner = router / "privacy_proxy.py", root / "eval" / "run_batch.py"
    for p in (proxy, runner):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found; is {root} an EC2-Bench checkout?")
    done = []
    shutil.copyfile(HERE.parent / "detect.py", router / "cb_detect.py")
    shutil.copyfile(HERE / "p3_view.py", router / "cb_p3.py")
    done += ["copied cb_detect.py, cb_p3.py"]
    done.append(f"privacy_proxy.py: {'patched' if _edit(proxy, PROXY_EDITS, MARK) else 'already patched'}")
    done.append(f"run_batch.py: {'patched' if _edit(runner, RUNNER_EDITS, 'CODEBLUE_PACKET') else 'already patched'}")
    return done
