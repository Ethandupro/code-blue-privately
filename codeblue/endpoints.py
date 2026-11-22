"""Checks for where the LOCAL model lives, and a quick health check for its endpoint.

The study's premise is that the local model runs on hardware the company controls. In practice that
may be a GPU VM (for example on Azure) reached through an SSH tunnel, so the harness sees it at
http://localhost:<port>/v1. Two mistakes would quietly break the measurement:

* pointing [local] at a public address, so "local" traffic actually crosses the internet in plain
  HTTP (and may hit an unauthenticated port), and
* pointing [local] at the same endpoint as [cloud], so local calls are counted (or not) as exposure.

`check_local_section` refuses both unless the config says `allow_remote = true` in [local].
`ping_local` lists the served models and asks for a two-word reply, without logging anything to
the cloud log.
"""

from __future__ import annotations

import ipaddress
import json
import os
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

_LOOPBACK_NAMES = {"localhost", "ip6-localhost"}


def is_private_host(host: str | None) -> bool:
    """True for loopback names, loopback IPs and private-range IPs (e.g. an Azure VNet address)."""
    if not host:
        return False
    host = host.strip("[]").lower()
    if host in _LOOPBACK_NAMES or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False  # a DNS name: we cannot tell where it resolves, so treat it as remote
    return ip.is_loopback or (ip.is_private and not ip.is_link_local)


def _norm(url: str) -> str:
    return url.rstrip("/").lower()


def check_local_section(local: dict, cloud: dict | None = None) -> None:
    """Raise ValueError if the [local] endpoint looks like it is not on hardware the run controls."""
    if local.get("kind") != "openai":
        return
    base = local.get("base_url", "")
    if cloud and cloud.get("kind") == "openai" and _norm(cloud.get("base_url", "")) == _norm(base):
        raise ValueError("[local] and [cloud] use the same base_url; local traffic would mix with cloud traffic")
    if local.get("allow_remote"):
        return
    host = urlparse(base).hostname
    if not is_private_host(host):
        raise ValueError(
            f"[local] base_url {base!r} is not a loopback or private address. Reach a remote GPU VM "
            "through an SSH tunnel (scripts/azure/tunnel.sh, then base_url = \"http://localhost:8000/v1\"), "
            "or set allow_remote = true in [local] if you really mean it."
        )


def _get(url: str, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def ping_local(local: dict, timeout: float = 60) -> dict:
    """List the models the [local] endpoint serves and time one tiny completion.

    Returns a dict with `models`, `model_listed`, `reply` and `secs`. Raises RuntimeError with a
    readable hint when the endpoint cannot be reached or refuses the key.
    """
    from codeblue.llm import OpenAICompatModel

    if local.get("kind") != "openai":
        raise RuntimeError(f"[local] kind is {local.get('kind')!r}; ping only works for kind = \"openai\"")
    base = local["base_url"].rstrip("/")
    headers = {}
    key_env = local.get("api_key_env")
    if key_env:
        key = os.environ.get(key_env)
        if not key:
            raise RuntimeError(f"environment variable {key_env} is not set (the vLLM server's API key)")
        headers["Authorization"] = f"Bearer {key}"
    try:
        listing = _get(f"{base}/models", headers, timeout)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            raise RuntimeError(f"{base} refused the API key (HTTP 401): check {key_env or 'api_key_env'}") from e
        raise RuntimeError(f"{base}/models answered HTTP {e.code}") from e
    except (urllib.error.URLError, OSError) as e:
        raise RuntimeError(
            f"cannot reach {base} ({e}). Is the SSH tunnel open (scripts/azure/tunnel.sh) "
            "and the server up on the VM (scripts/azure/serve_9b.sh)?"
        ) from e
    models = [m.get("id") for m in listing.get("data", [])]
    client = OpenAICompatModel(
        base_url=base,
        model=local["model"],
        api_key_env=key_env or None,
        temperature=0.0,
        max_tokens=16,
        timeout=timeout,
        extra_body=local.get("extra_body"),
    )
    t0 = time.time()
    reply = client.complete([{"role": "user", "content": "Reply with exactly: OK READY"}], "ping")
    return {"models": models, "model_listed": local["model"] in models, "reply": reply.strip(),
            "secs": round(time.time() - t0, 2)}
