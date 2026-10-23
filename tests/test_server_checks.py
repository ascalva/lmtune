import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

from lmtune import checks, flags, server

HELP = (Path(__file__).parent / "fixtures/llama-server-help-b11669.txt").read_text()


class Fake(BaseHTTPRequestHandler):
    seen: ClassVar[list[dict]] = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append(body)
        n = body["max_tokens"]
        res = {
            "choices": [{"finish_reason": "length", "message": {"content": "ok", "reasoning_content": "think"}}],
            "timings": {
                "prompt_n": 10,
                "prompt_ms": 100.0,
                "predicted_n": n,
                "predicted_ms": n * 100.0,
                "predicted_per_second": 10.0,
                "draft_n": 4,
                "draft_n_accepted": 3,
            },
        }
        data = json.dumps(res).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


class QuickServer(HTTPServer):
    def server_bind(self):  # skip HTTPServer's reverse-DNS lookup (slow on macOS)
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "127.0.0.1", self.server_address[1]


@pytest.fixture
def fake_url():
    srv = QuickServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_base_args_use_only_known_flags():
    f, a = flags.parse_help(HELP)
    tf = flags.ToolFlags("llama-server", 11669, None, f, a)
    argv = server.base_args(
        Path("/m.gguf"),
        ub=512,
        ctx=65536,
        parallel=1,
        kv="f16",
        cache_ram=2048,
        sampling={"temp": 1.0, "top_k": 20},
        host="127.0.0.1",
        port=8080,
        extra=["--cors-origins", "localhost"],
    )
    tf.require(argv)
    assert argv[argv.index("--top-k") + 1] == "20" and argv[argv.index("--min-p") + 1] == "0.05"


def test_serve_refuses_public_host(monkeypatch, tmp_path):
    f, a = flags.parse_help(HELP)
    monkeypatch.setattr(flags, "discover", lambda tool: flags.ToolFlags(tool, 1, None, f, a))
    with (
        pytest.raises(ValueError, match="loopback"),
        server.serve(["--host", "0.0.0.0", "--port", "1"], tmp_path / "log"),
    ):
        pass


def test_workload_and_aggregate(fake_url):
    Fake.seen.clear()
    prompts = [
        {"id": "p1", "messages": [{"role": "user", "content": "a"}]},
        {"id": "p2", "messages": [{"role": "user", "content": "b"}]},
    ]
    gens = server.run_workload(fake_url, "x", prompts, [1, 2], 50, greedy=True, log=lambda *_: None)
    assert len(gens) == 4 and all(g.text.startswith("=== reasoning ===\nthink") for g in gens)
    assert Fake.seen[0]["max_tokens"] == 8  # warmup first
    assert all(b["top_k"] == 1 and b["cache_prompt"] is False for b in Fake.seen[1:])
    agg = server.aggregate(gens)
    assert agg["gen_tps"] == pytest.approx(10.0) and agg["acceptance"] == pytest.approx(0.75)


def gen(prompt, seed, text, label="l"):
    return server.Generation(label, prompt, seed, False, {}, "stop", text)


def test_greedy_equivalence():
    ref = [gen("a", 1, "hello world"), gen("b", 1, "same")]
    cand = [gen("a", 1, "hello there"), gen("b", 1, "same"), gen("c", 1, "unmatched")]
    rows = checks.greedy_equivalence(ref, cand)
    assert [(r.prompt, r.identical, r.first_diff) for r in rows] == [("a", False, 6), ("b", True, None)]
    rows = checks.greedy_equivalence([gen("a", 1, "abc")], [gen("a", 1, "abcd")])
    assert rows[0].first_diff == 3 and not rows[0].identical


def test_stratified_removes_prompt_variance():
    ref = {"easy": [1.0, 1.02, 0.98], "hard": [3.0, 3.02, 2.98]}
    same = {"easy": [1.01, 0.99, 1.0], "hard": [3.01, 2.99, 3.0]}
    shifted = {"easy": [0.9, 0.92, 0.88], "hard": [2.9, 2.92, 2.88]}
    d, t = checks.stratified(same, ref)
    assert abs(d) < 0.01 and abs(t) < 1
    d, t = checks.stratified(shifted, ref)
    assert d == pytest.approx(-0.1) and t < -5
    assert checks.DistResult("x", 6, 1.0, d, t).passed() is False


def test_sampled_identity():
    ref = [gen("a", 1, "x"), gen("a", 2, "y"), gen("b", 1, "z"), gen("b", 2, "w")]
    cand = [gen("a", 1, "x"), gen("a", 2, "y"), gen("b", 1, "z"), gen("b", 2, "DIFFERENT")]
    assert checks.sampled_identity(ref, cand) == 0.75
    assert checks.sampled_identity(ref, []) == 0.0


def test_dist_ci():
    d = checks.DistResult("x", 12, 1.3, -0.05, -0.47)
    lo, hi = d.ci95
    assert lo < -0.2 and hi > 0.15  # the underpowered case seen in the Qwen run
