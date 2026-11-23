import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from codeblue.cli import main
from codeblue.endpoints import check_local_section, is_private_host, ping_local
from codeblue.runner import run_experiment


@pytest.mark.parametrize("host,ok", [
    ("localhost", True), ("127.0.0.1", True), ("::1", True), ("10.0.0.4", True), ("172.20.1.9", True),
    ("192.168.1.5", True), ("169.254.169.254", False), ("20.44.1.7", False), ("my-vm.eastus.cloudapp.azure.com", False),
    ("openrouter.ai", False), (None, False),
])
def test_is_private_host(host, ok):
    assert is_private_host(host) is ok


def test_check_local_section():
    check_local_section({"kind": "openai", "base_url": "http://localhost:8000/v1"})
    check_local_section({"kind": "mock-local"})
    with pytest.raises(ValueError, match="SSH tunnel"):
        check_local_section({"kind": "openai", "base_url": "http://20.44.1.7:8000/v1"})
    check_local_section({"kind": "openai", "base_url": "http://20.44.1.7:8000/v1", "allow_remote": True})
    same = {"kind": "openai", "base_url": "http://localhost:8787/v1/"}
    with pytest.raises(ValueError, match="same base_url"):
        check_local_section(same, {"kind": "openai", "base_url": "http://localhost:8787/v1"})


def test_run_refuses_public_local_endpoint(tmp_path):
    cfg = {"local": {"kind": "openai", "base_url": "http://20.44.1.7:8000/v1", "model": "m"},
           "experiment": {"out_dir": str(tmp_path / "runs")}}
    with pytest.raises(ValueError):
        run_experiment(cfg, quiet=True)
    assert not (tmp_path / "runs").exists()  # refused before anything was written


class FakeVLLM(BaseHTTPRequestHandler):
    key = "s3cret"
    bodies = []

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        out = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _authed(self):
        if self.headers.get("Authorization") != f"Bearer {self.key}":
            self._send(401, {"error": "Unauthorized"})
            return False
        return True

    def do_GET(self):
        if self._authed():
            self._send(200, {"data": [{"id": "Qwen/Qwen3.5-9B"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeVLLM.bodies.append(body)
        if self._authed():
            if body.get("model") != "Qwen/Qwen3.5-9B":  # like vLLM: unknown model -> 404
                self._send(404, {"error": "model does not exist"})
                return
            self._send(200, {"choices": [{"message": {"role": "assistant", "content": "OK READY"}}]})


@pytest.fixture
def vllm_url():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def _local(url):
    return {"kind": "openai", "base_url": url, "model": "Qwen/Qwen3.5-9B", "api_key_env": "CB_KEY",
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}


def test_ping_local(vllm_url, monkeypatch):
    monkeypatch.setenv("CB_KEY", "s3cret")
    res = ping_local(_local(vllm_url))
    assert res["model_listed"] and res["reply"] == "OK READY"
    assert FakeVLLM.bodies[-1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_ping_local_errors(vllm_url, monkeypatch):
    monkeypatch.delenv("CB_KEY", raising=False)
    with pytest.raises(RuntimeError, match="not set"):
        ping_local(_local(vllm_url))
    monkeypatch.setenv("CB_KEY", "wrong")
    with pytest.raises(RuntimeError, match="401"):
        ping_local(_local(vllm_url))
    with pytest.raises(RuntimeError, match="tunnel"):
        ping_local(_local("http://127.0.0.1:9/v1"), timeout=2)


def test_ping_cli(vllm_url, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("CB_KEY", "s3cret")
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[local]\nkind = "openai"\nbase_url = "{vllm_url}"\nmodel = "Qwen/Qwen3.5-9B"\napi_key_env = "CB_KEY"\n')
    assert main(["ping", "--config", str(cfg)]) == 0
    assert "OK READY" in capsys.readouterr().out
    cfg.write_text(f'[local]\nkind = "openai"\nbase_url = "{vllm_url}"\nmodel = "other"\napi_key_env = "CB_KEY"\n')
    assert main(["ping", "--config", str(cfg)]) == 1


def test_ping_cli_cloud(vllm_url, monkeypatch, tmp_path, capsys):
    # a stand-in "cloud" model served on the same VM (another port); ping --cloud checks [cloud], not [local]
    monkeypatch.setenv("CB_KEY", "s3cret")
    cfg = tmp_path / "c.toml"
    cfg.write_text('[local]\nkind = "mock-local"\n'
                   f'[cloud]\nkind = "openai"\nbase_url = "{vllm_url}"\nmodel = "Qwen/Qwen3.5-9B"\napi_key_env = "CB_KEY"\n')
    assert main(["ping", "--cloud", "--config", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "[cloud] endpoint" in out and "OK READY" in out
    cfg.write_text('[local]\nkind = "mock-local"\n[cloud]\nkind = "mock-cloud"\n')
    assert main(["ping", "--cloud", "--config", str(cfg)]) == 1


def test_shipped_two_model_config_passes_endpoint_checks():
    from codeblue.config import load_config
    from pathlib import Path
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "azure-9b-local-27b-cloud.toml")
    check_local_section(cfg["local"], cfg["cloud"])
    assert cfg["local"]["base_url"] != cfg["cloud"]["base_url"]
    assert cfg["local"]["api_key_env"] != cfg["cloud"]["api_key_env"]


def test_ping_wrong_model_on_port_is_a_clear_failure(vllm_url, monkeypatch, tmp_path, capsys):
    # another server (for a different model) already on the port: no completion is sent, no traceback
    monkeypatch.setenv("CB_KEY", "s3cret")
    local = dict(_local(vllm_url), model="Qwen/Qwen3.8-27B")
    n = len(FakeVLLM.bodies)
    res = ping_local(local)
    assert not res["model_listed"] and len(FakeVLLM.bodies) == n
    cfg = tmp_path / "c.toml"
    cfg.write_text(f'[local]\nkind = "openai"\nbase_url = "{vllm_url}"\nmodel = "Qwen/Qwen3.8-27B"\napi_key_env = "CB_KEY"\n')
    assert main(["ping", "--config", str(cfg)]) == 1
    assert "free PORT" in capsys.readouterr().out
