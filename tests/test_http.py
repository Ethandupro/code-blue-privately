import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from codeblue.gateway import read_cloud_requests
from codeblue.llm import OpenAICompatModel
from codeblue.proxy_server import make_server


class FakeUpstream(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeUpstream.seen.append((self.path, self.headers.get("Authorization"), body))
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": "<think>hm</think>pong"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


@pytest.fixture
def upstream():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeUpstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_openai_compat_client(upstream, monkeypatch):
    monkeypatch.setenv("TEST_KEY", "k123")
    m = OpenAICompatModel(upstream, "tiny", api_key_env="TEST_KEY", extra_body={"top_p": 0.9})
    assert m.complete([{"role": "user", "content": "ping"}]) == "<think>hm</think>pong"
    path, auth, body = FakeUpstream.seen[-1]
    assert path == "/v1/chat/completions" and auth == "Bearer k123" and body["top_p"] == 0.9


def test_logging_proxy(upstream, tmp_path):
    log = tmp_path / "cloud.jsonl"
    proxy = make_server(upstream, log, port=0)
    threading.Thread(target=proxy.serve_forever, daemon=True).start()
    try:
        client = OpenAICompatModel(f"http://127.0.0.1:{proxy.server_address[1]}/v1", "big",
                                   extra_headers={"X-CodeBlue-Episode": "ep9"})
        assert client.complete([{"role": "user", "content": "deal HL-ACQ-2291-KX"}]).endswith("pong")
    finally:
        proxy.shutdown()
    reqs = read_cloud_requests(log)
    assert reqs[0]["episode_id"] == "ep9"
    assert reqs[0]["request"][0]["content"] == "deal HL-ACQ-2291-KX"
    assert FakeUpstream.seen[-1][0] == "/v1/chat/completions"
