from __future__ import annotations

import http.client
import json
import shutil
import socket
import subprocess
import sys
import threading
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest

from lmtune import cli, config, doctor, harness, keys, machine, media
from lmtune.media import Mcp, MediaConfig, ToolError, Tools
from lmtune.paths import Paths
from lmtune.proc import Result

# -- config ---------------------------------------------------------------------------------------


def test_default_config_is_valid() -> None:
    assert MediaConfig().problems() == []


@pytest.mark.parametrize("sub", ["Documents/media", "Downloads", ".ssh/x", ".config/lmtune", "Library/m"])
def test_dir_must_not_be_a_protected_folder(sub: str) -> None:
    assert MediaConfig(dir=Path.home() / sub).problems()


def test_dir_must_be_a_dedicated_folder() -> None:
    for bad in (Path.home(), Path.home().parent, Path("/"), Path("/Users")):
        assert MediaConfig(dir=bad).problems(), bad
    assert MediaConfig(dir=Path("/Volumes/ext/media")).problems() == []


def test_load_expands_and_rejects_unknown_keys(tmp_path: Path) -> None:
    f = tmp_path / "media.toml"
    f.write_text('[media]\ndir = "~/lmtune-media-test"\nport = 8199\nextra_hosts = ["10.153.1.5:8199"]\n')
    cfg = media.load(f)
    assert cfg.dir == Path.home() / "lmtune-media-test"
    assert cfg.port == 8199
    assert cfg.extra_hosts == ("10.153.1.5:8199",)
    f.write_text("[media]\nnetwork = true\n")
    with pytest.raises(ValueError, match="unknown"):
        media.load(f)
    assert media.load(tmp_path / "missing.toml") == MediaConfig(key_file=tmp_path / "media-key")


# -- sandbox profile --------------------------------------------------------------------------------


def _cfg(tmp_path: Path) -> MediaConfig:
    home = tmp_path / "home"
    (home / "media").mkdir(parents=True)
    (home / "models" / "whisper").mkdir(parents=True)
    (home / "models" / "whisper" / "ggml-tiny.bin").write_bytes(b"x")
    (home / ".config" / "lmtune").mkdir(parents=True)
    (home / ".config" / "lmtune" / "media-key").write_text("k" * 40)
    return MediaConfig(
        dir=home / "media",
        whisper_model=home / "models" / "whisper" / "ggml-tiny.bin",
        key_file=home / ".config" / "lmtune" / "media-key",
        port=18095,
    )


def test_profile_denies_network_and_home_then_allows_only_what_it_needs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    home = (tmp_path / "home").resolve()
    prof = media.sandbox_profile(cfg, home=home, read_only=[])
    lines = prof.splitlines()
    assert lines[:2] == ["(version 1)", "(allow default)"]
    assert "(deny network*)" in lines
    assert '(allow network-bind network-inbound (local ip "localhost:18095"))' in lines
    deny_home = lines.index(f'(deny file-read* file-write* (subpath "{home}"))')
    allow_reads = next(i for i, x in enumerate(lines) if x.startswith("(allow file-read* "))
    assert deny_home < allow_reads  # later rules win: the allows must come after the deny
    assert f'(subpath "{home / "media"}")' in lines[allow_reads]
    assert f'(subpath "{home / "models" / "whisper"}")' in lines[allow_reads]
    assert f'(subpath "{home / ".config" / "lmtune" / "media-key"}")' in lines[allow_reads]
    writes = lines[lines.index("(deny file-write*)") + 1]
    assert writes == f'(allow file-write* (subpath "{home / "media"}") (literal "/dev/null"))'
    meta = next(x for x in lines if x.startswith("(allow file-read-metadata"))
    assert f'(literal "{home / "models"}")' in meta and f'(literal "{home}")' in meta


def test_program_wraps_the_worker_in_sandbox_exec(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    paths = Paths(tmp_path / "cfg", tmp_path / "state", tmp_path / "log", tmp_path / "la")
    launcher = media.Launcher(("/x/venv/bin/lmtune",), (tmp_path / "home" / "venv",))
    argv = media.program(cfg, paths, launcher)
    assert argv[:2] == [media.SANDBOX_EXEC, "-p"]
    assert argv[3:] == ["/x/venv/bin/lmtune", "media", "worker"]
    assert "(deny network*)" in argv[2]
    assert f'(subpath "{(tmp_path / "home" / "venv").resolve()}")' in argv[2]  # the launcher's code is readable
    assert f'(subpath "{(paths.config_dir / "media.toml").resolve()}")' in argv[2]
    assert media.program(cfg, paths, launcher, sub="probe")[-1] == "probe"


@pytest.mark.skipif(not Path(media.SANDBOX_EXEC).exists(), reason="macOS sandbox-exec only")
def test_profile_enforced_by_sandbox_exec(tmp_path: Path) -> None:
    """The real kernel sandbox with the generated profile (a temp folder stands in for home)."""
    cfg = _cfg(tmp_path)
    home = (tmp_path / "home").resolve()
    (home / "secret.txt").write_text("secret")
    (home / "media" / "in.txt").write_text("media")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    script = f"""
import socket, sys
def t(name, fn):
    try:
        fn(); print(name, "works")
    except Exception:
        print(name, "blocked")
t("read-media", lambda: open({str(home / "media" / "in.txt")!r}).read())
t("read-secret", lambda: open({str(home / "secret.txt")!r}).read())
t("read-model", lambda: open({str(cfg.whisper_model)!r}, "rb").read())
t("write-media", lambda: open({str(home / "media" / "out.txt")!r}, "w").write("x"))
t("write-home", lambda: open({str(home / "out.txt")!r}, "w").write("x"))
t("connect-loopback", lambda: socket.create_connection(("127.0.0.1", {port}), timeout=3))
"""
    prof = media.sandbox_profile(cfg, home=home, read_only=[])
    r = subprocess.run(
        [media.SANDBOX_EXEC, "-p", prof, sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    listener.close()
    got = dict(line.split() for line in r.stdout.splitlines())
    assert got == {
        "read-media": "works",
        "read-secret": "blocked",
        "read-model": "works",
        "write-media": "works",
        "write-home": "blocked",
        "connect-loopback": "blocked",
    }, r.stderr


def test_worker_refuses_to_run_unsandboxed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsandboxed"):
        media.serve_forever(_cfg(tmp_path))


# -- tools ------------------------------------------------------------------------------------------


class FakeExec:
    """Records commands; ffprobe answers with a fixed duration, whisper writes its JSON/TXT outputs."""

    def __init__(self, duration: float = 120.0, fail: str = "") -> None:
        self.calls: list[list[str]] = []
        self.duration, self.fail = duration, fail

    def __call__(self, cmd: Sequence[str], timeout: float) -> Result:
        cmd = list(cmd)
        self.calls.append(cmd)
        if self.fail and cmd[0] == self.fail:
            return Result(1, "", "boom")
        if cmd[0] == "ffprobe":
            probe = {"format": {"format_name": "wav", "duration": str(self.duration), "size": "1000"}, "streams": []}
            return Result(0, json.dumps(probe), "")
        if cmd[0] == "whisper-cli":
            stem = Path(cmd[cmd.index("-of") + 1])
            segs = [{"text": " Hello there."}, {"text": " General Kenobi."}]
            stem.with_suffix(".json").write_text(json.dumps({"result": {"language": "en"}, "transcription": segs}))
            stem.with_suffix(".txt").write_text("Hello there.\nGeneral Kenobi.\n")
        if cmd[0] == "ffmpeg":
            Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
            if "%03d" in cmd[-1]:
                for i in (1, 2):
                    Path(cmd[-1].replace("%03d", f"{i:03d}")).write_bytes(b"jpg")
            else:
                Path(cmd[-1]).write_bytes(b"out")
        return Result(0, "", "")


@pytest.fixture
def tools(tmp_path: Path) -> Iterator[tuple[Tools, FakeExec]]:
    cfg = _cfg(tmp_path)
    (cfg.dir / "talk.m4a").write_bytes(b"audio")
    (cfg.dir / "clip.mov").write_bytes(b"video")
    fake = FakeExec()
    ticks = iter([0.0, 10.0])
    yield Tools(cfg, exe=fake, clock=lambda: next(ticks)), fake


def test_paths_stay_inside_the_media_folder(tools: tuple[Tools, FakeExec], tmp_path: Path) -> None:
    t, _ = tools
    (tmp_path / "home" / "secret.txt").write_text("s")
    (t.root / "link").symlink_to(tmp_path / "home" / "secret.txt")
    for bad in ["../secret.txt", str(tmp_path / "home" / "secret.txt"), "link", ".work/x.wav", "", "missing.mp3"]:
        with pytest.raises(ToolError):
            t.path(bad)
    assert t.path("talk.m4a") == t.root / "talk.m4a"


def test_unknown_tool_and_arguments_are_refused(tools: tuple[Tools, FakeExec]) -> None:
    t, _ = tools
    with pytest.raises(ToolError, match="unknown tool"):
        t.call("shell", {})
    with pytest.raises(ToolError, match="unknown arguments: url"):
        t.call("probe", {"path": "talk.m4a", "url": "http://x"})


def test_transcribe_converts_then_runs_whisper_and_reports_speed(tools: tuple[Tools, FakeExec]) -> None:
    t, fake = tools
    out = t.call("transcribe", {"path": "talk.m4a", "language": "en"})
    ffmpeg = next(c for c in fake.calls if c[0] == "ffmpeg")
    assert ffmpeg[:7] == list(media.FFMPEG) and "-protocol_whitelist" in ffmpeg
    assert ffmpeg[ffmpeg.index("-ar") + 1] == "16000"
    whisper = next(c for c in fake.calls if c[0] == "whisper-cli")
    assert whisper[whisper.index("-m") + 1].endswith("ggml-tiny.bin")
    assert whisper[whisper.index("-l") + 1] == "en"
    assert "-osrt" not in whisper
    assert out.startswith("[ggml-tiny.bin · language en · 120 s of audio in 10 s (12x realtime) · saved out/talk.txt]")
    assert out.endswith("Hello there. General Kenobi.")
    assert not list((t.root / media.WORK).glob("*.wav"))  # intermediate WAV removed


def test_transcribe_validates_before_running(tools: tuple[Tools, FakeExec]) -> None:
    t, fake = tools
    with pytest.raises(ToolError, match="not installed"):
        t.call("transcribe", {"path": "talk.m4a", "model": "ggml-huge.bin"})
    with pytest.raises(ToolError, match="language"):
        t.call("transcribe", {"path": "talk.m4a", "language": "en; rm -rf"})
    fake.duration = 5 * 3600
    with pytest.raises(ToolError, match="limit"):
        t.call("transcribe", {"path": "talk.m4a"})
    assert not any(c[0] == "whisper-cli" for c in fake.calls)


def test_heavy_tools_run_one_at_a_time(tools: tuple[Tools, FakeExec]) -> None:
    t, _ = tools
    assert t._busy.acquire(blocking=False)
    try:
        with pytest.raises(ToolError, match="busy"):
            t.call("convert", {"path": "talk.m4a", "to": "mp3"})
    finally:
        t._busy.release()


def test_tool_failure_is_reported_with_the_command_error(tools: tuple[Tools, FakeExec]) -> None:
    t, fake = tools
    fake.fail = "ffmpeg"
    with pytest.raises(ToolError, match=r"ffmpeg failed \(1\): boom"):
        t.call("convert", {"path": "talk.m4a", "to": "mp3"})


def test_convert_builds_a_cut_and_scaled_output_in_out(tools: tuple[Tools, FakeExec]) -> None:
    t, fake = tools
    msg = t.call("convert", {"path": "clip.mov", "to": "mp4", "start": 5, "end": 12.5, "max_px": 720})
    cmd = fake.calls[-1]
    assert cmd[cmd.index("-ss") + 1] == "5" and cmd[cmd.index("-to") + 1] == "12.5"
    assert cmd.index("-ss") < cmd.index("-i")  # input seeking
    assert "min(720,iw)" in cmd[cmd.index("-vf") + 1]
    assert cmd[-1] == str(t.root / "out" / "clip_5-12.5s.mp4")
    assert msg.startswith("saved out/clip_5-12.5s.mp4")
    t.call("convert", {"path": "clip.mov", "to": "m4a"})
    assert "-vn" in fake.calls[-1]
    for bad in [{"to": "exe"}, {"to": "mp3", "start": -1}, {"to": "mp3", "start": 9, "end": 3}]:
        with pytest.raises(ToolError):
            t.call("convert", {"path": "clip.mov", **bad})


def test_frames_extracts_jpegs(tools: tuple[Tools, FakeExec]) -> None:
    t, fake = tools
    res = json.loads(t.call("frames", {"path": "clip.mov", "every_s": 2, "max_frames": 2}))
    cmd = fake.calls[-1]
    assert cmd[cmd.index("-vf") + 1].startswith("fps=1/2,")
    assert res["frames"] == ["out/clip_frames/frame_001.jpg", "out/clip_frames/frame_002.jpg"]


def test_list_files_hides_internal_files(tools: tuple[Tools, FakeExec]) -> None:
    t, _ = tools
    (t.root / media.WORK).mkdir()
    (t.root / media.WORK / "x.wav").write_bytes(b"")
    rows = json.loads(t.call("list_files", {}))
    assert [r["path"] for r in rows] == ["clip.mov", "talk.m4a"]


def test_summarize_probe() -> None:
    d = {
        "format": {"format_name": "mov,mp4", "duration": "61.5", "size": "2048"},
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "30000/1001",
            },
            {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 2, "tags": {"x": 1}},
        ],
    }
    s = media.summarize_probe(d)
    assert s["duration_s"] == 61.5 and s["bytes"] == 2048
    assert s["streams"][0]["fps"] == 29.97
    assert "tags" not in s["streams"][1]


# -- MCP ----------------------------------------------------------------------------------------


@pytest.fixture
def mcp(tools: tuple[Tools, FakeExec]) -> Mcp:
    return Mcp(tools[0])


def test_initialize_negotiates_the_protocol_version(mcp: Mcp) -> None:
    r = mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26"}})
    assert r["result"]["protocolVersion"] == "2025-03-26"
    assert r["result"]["capabilities"] == {"tools": {"listChanged": False}}
    r = mcp.handle({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {"protocolVersion": "1999-01-01"}})
    assert r["result"]["protocolVersion"] == media.PROTOCOL_VERSIONS[0]


def test_notifications_and_responses_get_no_reply(mcp: Mcp) -> None:
    assert mcp.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert mcp.handle({"jsonrpc": "2.0", "id": 9, "result": {}}) is None
    assert mcp.handle([{"jsonrpc": "2.0", "method": "notifications/initialized"}]) is None


def test_errors_follow_json_rpc(mcp: Mcp) -> None:
    assert mcp.handle({"id": 1, "method": "ping"})["error"]["code"] == -32600
    assert mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "resources/list"})["error"]["code"] == -32601
    assert mcp.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": []})["error"]["code"] == -32602
    assert mcp.handle([])["error"]["code"] == -32600


def test_tools_list_and_call(mcp: Mcp) -> None:
    batch = mcp.handle(
        [
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "probe", "arguments": {"path": "x"}},
            },
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "list_files"}},
        ]
    )
    names = [t["name"] for t in batch[0]["result"]["tools"]]
    assert names == ["list_files", "probe", "metadata", "transcribe", "convert", "frames"]
    for spec in batch[0]["result"]["tools"]:
        assert spec["inputSchema"]["type"] == "object" and spec["description"]
    assert batch[1]["result"]["isError"] is True and "no such file" in batch[1]["result"]["content"][0]["text"]
    assert batch[2]["result"]["isError"] is False


# -- HTTP -----------------------------------------------------------------------------------------


@pytest.fixture
def worker(mcp: Mcp) -> Iterator[tuple[int, str]]:
    key = "sk-test-" + "k" * 40
    srv = media.MediaServer(("127.0.0.1", 0), mcp, key, frozenset())
    port = srv.server_address[1]
    srv.hosts = frozenset({f"127.0.0.1:{port}"})
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    th.start()
    yield port, key
    srv.shutdown()
    srv.server_close()


def _req(port: int, method: str, body: object = None, **headers: str) -> tuple[int, bytes]:
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    data = json.dumps(body).encode() if body is not None else None
    hdrs = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"} | headers
    c.request(method, "/mcp", body=data, headers=hdrs)
    r = c.getresponse()
    out = r.status, r.read()
    c.close()
    return out


def test_http_requires_the_key_and_a_loopback_host(worker: tuple[int, str]) -> None:
    port, key = worker
    ping = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    assert _req(port, "POST", ping)[0] == 401
    assert _req(port, "POST", ping, Authorization="Bearer wrong")[0] == 401
    assert _req(port, "POST", ping, Authorization=f"Bearer {key}", Host="evil.example:80")[0] == 421
    assert _req(port, "POST", ping, Authorization=f"Bearer {key}", Origin="https://evil.example")[0] == 403
    code, body = _req(port, "POST", ping, Authorization=f"Bearer {key}")
    assert code == 200 and json.loads(body) == {"jsonrpc": "2.0", "id": 1, "result": {}}


def test_http_notification_is_accepted_without_body_and_get_is_405(worker: tuple[int, str]) -> None:
    port, key = worker
    code, body = _req(
        port, "POST", {"jsonrpc": "2.0", "method": "notifications/initialized"}, Authorization=f"Bearer {key}"
    )
    assert (code, body) == (202, b"")
    assert _req(port, "GET", Authorization=f"Bearer {key}")[0] == 405


def test_http_rejects_unparseable_json(worker: tuple[int, str]) -> None:
    port, key = worker
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    c.request("POST", "/mcp", body=b"{not json", headers={"Authorization": f"Bearer {key}"})
    r = c.getresponse()
    assert r.status == 400 and json.loads(r.read())["error"]["code"] == -32700


@pytest.mark.skipif(not shutil.which("sandbox-exec"), reason="macOS only")
def test_sandboxed_detects_sandbox_exec() -> None:
    assert media.sandboxed() is False
    code = "from lmtune import media; print(media.sandboxed())"
    r = subprocess.run(
        [media.SANDBOX_EXEC, "-p", "(version 1)(allow default)(deny network*)", sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert r.stdout.strip() == "True", r.stderr


def test_key_file_defaults_next_to_the_config(tmp_path: Path) -> None:
    assert media.load(tmp_path / "media.toml").key_file == tmp_path / "media-key"


# -- vision (serve config, doctor, harness) -------------------------------------------------------


@pytest.fixture
def serve_cfg(tmp_path: Path) -> config.ServeConfig:
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    key = tmp_path / "api-key"
    keys.write(key, keys.generate())
    return config.ServeConfig(model=model, api_key_file=key)


def test_mmproj_is_off_by_default_and_passed_when_set(serve_cfg: config.ServeConfig, tmp_path: Path) -> None:
    assert "--mmproj" not in serve_cfg.argv()
    v = config.with_overrides(serve_cfg, mmproj=tmp_path / "mmproj-F16.gguf")
    argv = v.argv()
    assert argv[argv.index("--mmproj") + 1] == str(tmp_path / "mmproj-F16.gguf")
    assert argv.index("--mmproj") < argv.index("--spec-type")
    assert v.problems() == []


def test_mmproj_round_trips_and_empty_means_off(serve_cfg: config.ServeConfig, tmp_path: Path) -> None:
    f = tmp_path / "serve.toml"
    v = config.with_overrides(serve_cfg, mmproj=tmp_path / "mmproj-F16.gguf")
    f.write_text(config.to_toml(v))
    assert config.load(f) == v
    f.write_text(config.to_toml(serve_cfg))
    assert 'mmproj = ""' in f.read_text()
    assert config.load(f).mmproj is None


def test_mmproj_must_not_go_through_extra(serve_cfg: config.ServeConfig) -> None:
    bad = config.with_overrides(serve_cfg, extra=("--mmproj", "/x.gguf"))
    assert any("through mmproj" in p for p in bad.problems())


def test_doctor_budgets_the_projector(
    serve_cfg: config.ServeConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Info:
        def kv_bytes(self, n: int, kv: str) -> int:
            return 4 * 2**30

    monkeypatch.setattr(doctor.gguf, "inspect", lambda p: Info())
    with serve_cfg.model.open("ab") as f:
        f.truncate(15 * 2**30)
    proj = tmp_path / "mmproj-F16.gguf"
    with proj.open("wb") as f:
        f.truncate(900 * 2**20)
    m = machine.Machine("Apple M2 Pro", 32 * 2**30, 25559, 0, 10**11, machine.Power("AC", 100, "charged", False))
    base = doctor.check_memory(serve_cfg, m)
    with_vision = doctor.check_memory(config.with_overrides(serve_cfg, mmproj=proj), m)
    assert base.level == with_vision.level == "ok"
    assert "vision" not in base.detail and "+ vision 1.0 " in with_vision.detail  # 900 + 128 MiB
    gone = doctor.check_memory(config.with_overrides(serve_cfg, mmproj=tmp_path / "nope.gguf"), m)
    assert gone.level == "fail" and "mmproj" in gone.detail


def test_harness_declares_images_only_with_a_projector(serve_cfg: config.ServeConfig, tmp_path: Path) -> None:
    text = harness.opencode_config(serve_cfg)["provider"]["lmtune"]["models"][serve_cfg.alias]
    assert "modalities" not in text and "attachment" not in text
    v = config.with_overrides(serve_cfg, mmproj=tmp_path / "mmproj-F16.gguf")
    model = harness.opencode_config(v)["provider"]["lmtune"]["models"][v.alias]
    assert model["attachment"] is True
    assert model["modalities"] == {"input": ["text", "image"], "output": ["text"]}


def test_harness_adds_the_media_worker_with_its_key_from_file(serve_cfg: config.ServeConfig, tmp_path: Path) -> None:
    assert "mcp" not in harness.opencode_config(serve_cfg)
    mc = MediaConfig(port=8095, key_file=tmp_path / "media-key", timeout_s=1800)
    keys.write(mc.key_file, keys.generate())
    oc = harness.opencode_config(serve_cfg, media=mc)
    assert oc["mcp"]["media"] == {
        "type": "remote",
        "url": "http://127.0.0.1:8095/mcp",
        "enabled": True,
        "headers": {"Authorization": f"Bearer {{file:{mc.key_file}}}"},
        "oauth": False,
        "timeout": 1_800_000,
    }
    assert "sk-lmtune-" not in json.dumps(oc)  # the key itself never lands in the config


def test_cli_media_parser() -> None:
    p = cli.build_parser()
    for argv in (["media", "serve"], ["media", "verify"], ["media", "install"], ["media", "status"]):
        assert p.parse_args(argv).fn is cli.media_cmd
    a = p.parse_args(["media", "call", "probe", '{"path": "a.mp4"}'])
    assert (a.tool, a.arguments) == ("probe", '{"path": "a.mp4"}')
