"""Serve stack: config, keys, launchd, tailscale, supervisor, CLI — all external commands faked."""

import json
import plistlib
import stat
from collections.abc import Sequence
from pathlib import Path

import pytest

from lmtune import cli, config, keys, launchd, serve, tailscale
from lmtune.paths import LABEL, Paths
from lmtune.proc import Result

TS_JSON = {
    "TCP": {"443": {"HTTPS": True}},
    "Web": {"mac.tail1234.ts.net:443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8080"}}}},
}
LAUNCHCTL_PRINT = """gui/501/com.lmtune.server = {
	active count = 1
	state = running
	program = /opt/homebrew/bin/llama-server
	runs = 2
	pid = 4242
	last exit code = (never exited)
}
"""
LSOF = """COMMAND     PID    USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
llama-ser  4242 ascalva    5u  IPv4 0x1234      0t0  TCP 127.0.0.1:8080 (LISTEN)
"""


@pytest.fixture
def paths(tmp_path):
    return Paths(tmp_path / "cfg", tmp_path / "state", tmp_path / "logs", tmp_path / "agents")


@pytest.fixture
def cfg(paths, tmp_path):
    model = tmp_path / "models" / "m.gguf"
    model.parent.mkdir()
    model.write_bytes(b"GGUF")
    keys.write(paths.api_key, keys.generate())
    return config.ServeConfig(model=model, api_key_file=paths.api_key)


class FakeRunner:
    """Records commands; answers launchctl/tailscale/lsof/ps/footprint like a running server."""

    def __init__(self, loaded=True):
        self.calls: list[list[str]] = []
        self.loaded = loaded

    def __call__(self, cmd: Sequence[str]) -> Result:
        c = list(cmd)
        self.calls.append(c)
        if c[:2] == ["launchctl", "print"]:
            return Result(0, LAUNCHCTL_PRINT, "") if self.loaded else Result(113, "", "Could not find service")
        if c[:2] == ["launchctl", "bootstrap"]:
            self.loaded = True
            return Result(0, "", "")
        if c[:2] == ["launchctl", "bootout"]:
            was, self.loaded = self.loaded, False
            return Result(0, "", "") if was else Result(3, "", "No such process")
        if c[0] == "lsof":
            return Result(0, LSOF, "") if self.loaded else Result(1, "", "")
        if c[0] == "ps":
            return Result(0, "21000000\n", "")
        if c[0] == "footprint":
            return Result(0, "llama-server [4242]: 64-bit    Footprint: 6.5 GB (16384 bytes per page)\n", "")
        if "serve" in c and "status" in c:
            return Result(0, json.dumps(TS_JSON), "")
        if "serve" in c and "--bg" in c:
            return Result(0, "Serve started", "")
        return Result(0, "", "")


def fake_http(state="ok"):
    def get(url: str, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
        if url.endswith("/health"):
            return (200, b'{"status":"ok"}') if state == "ok" else (503, b'{"status":"loading"}')
        assert headers.get("Authorization", "").startswith("Bearer sk-lmtune-")  # key is sent
        if url.endswith("/props"):
            return 200, json.dumps(
                {"model_alias": "qwen3.8-27b", "build_info": "b11669-x", "is_sleeping": True}
            ).encode()
        if url.endswith("/slots"):
            return 200, b'[{"id":0,"is_processing":false}]'
        return 404, b""

    return get


# -- config ---------------------------------------------------------------------------------


def test_defaults_are_the_measured_winner(cfg):
    argv = cfg.argv()
    for flag, val in [
        ("-ctk", "f16"),
        ("-ub", "512"),
        ("--cache-ram", "2048"),
        ("--spec-draft-n-max", "2"),
        ("--spec-draft-sampling", "probabilistic"),
        ("--host", "127.0.0.1"),
        ("-a", "qwen3.8-27b"),
    ]:
        assert argv[argv.index(flag) + 1] == val
    assert "--api-key-file" in argv and "--api-key" not in argv
    assert "--sleep-idle-seconds" in argv
    assert cfg.problems() == []


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"host": "0.0.0.0"}, "loopback"),
        ({"api_key_file": None}, "api_key_file is required"),
        ({"kv": "q8_0", "flash_attn": "off"}, "needs flash_attn on"),
        ({"extra": ("--api-key", "secret")}, "never put an API key"),
        ({"extra": ("--host", "0.0.0.0")}, "own keys"),
        ({"sleep_idle_s": 0}, "sleep_idle_s"),
        ({"kv": "q9_9"}, "kv"),
    ],
)
def test_config_rejects_unsafe_or_invalid(cfg, change, message):
    bad = config.with_overrides(cfg, **change)
    assert any(message in p for p in bad.problems())
    with pytest.raises(config.ConfigError):
        bad.check()


def test_toml_round_trip(cfg, tmp_path):
    p = tmp_path / "serve.toml"
    p.write_text(config.to_toml(cfg))
    assert config.load(p) == cfg


def test_toml_errors(tmp_path):
    p = tmp_path / "s.toml"
    p.write_text('[server]\nmodel = "/m.gguf"\nbogus = 1\n')
    with pytest.raises(config.ConfigError, match="bogus"):
        config.load(p)
    with pytest.raises(config.ConfigError, match="not found"):
        config.load(tmp_path / "missing.toml")
    p.write_text('[server]\nmodel = "/m.gguf"\nspec = "nope"\n')
    with pytest.raises(config.ConfigError, match="list of strings"):
        config.load(p)


# -- keys -----------------------------------------------------------------------------------


def test_key_file_is_private(paths):
    assert keys.ensure(paths.api_key) is True  # fixture-free path: fresh dir
    assert keys.ensure(paths.api_key) is False
    assert stat.S_IMODE(paths.api_key.stat().st_mode) == 0o600
    assert stat.S_IMODE(paths.api_key.parent.stat().st_mode) == 0o700
    assert keys.read(paths.api_key).startswith("sk-lmtune-")
    assert keys.problems(paths.api_key) == []
    paths.api_key.chmod(0o644)
    assert any("chmod 600" in p for p in keys.problems(paths.api_key))


# -- launchd / tailscale ----------------------------------------------------------------------


def test_plist_restarts_on_crash_only(tmp_path):
    d = plistlib.loads(launchd.plist(LABEL, ["/bin/llama-server", "-m", "x"], tmp_path / "log"))
    assert d["KeepAlive"] == {"SuccessfulExit": False}
    assert d["ThrottleInterval"] >= 10 and d["RunAtLoad"] is True
    assert d["ProcessType"] == "Interactive"


def test_parse_launchctl_print():
    st = launchd.parse_print(LAUNCHCTL_PRINT)
    assert (st.loaded, st.state, st.pid, st.runs) == (True, "running", 4242, 2)


def test_tailscale_serve_status_parse():
    assert tailscale.parse_serve_status(TS_JSON, 8080) == (
        "https://mac.tail1234.ts.net",
        "http://127.0.0.1:8080",
    )
    assert tailscale.parse_serve_status(TS_JSON, 9999) == (None, None)
    assert tailscale.parse_serve_status({}, 8080) == (None, None)
    webui = {"Web": {"box.ts.net:8443": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:3000"}}}}}
    assert tailscale.parse_serve_status(webui, 3000)[0] == "https://box.ts.net:8443"


def test_tailscale_never_uses_funnel():
    r = FakeRunner()
    ts = tailscale.Tailscale(r, exe="/bin/tailscale")
    ts.expose(8080)
    ts.unexpose()
    assert not any("funnel" in " ".join(c) for c in r.calls)


# -- supervisor -------------------------------------------------------------------------------


def test_status_running_sleeping(cfg, paths):
    srv = serve.Server(cfg, paths, FakeRunner(), fake_http())
    st = srv.status()
    # busy is not asked while sleeping: /slots would wake the model (see test_dash)
    assert st.running and st.health == "ok" and st.sleeping is True and st.busy is None
    assert st.model == "qwen3.8-27b" and st.loopback_only and st.listeners == ["127.0.0.1:8080"]
    assert st.exposure is not None and st.exposure.url == "https://mac.tail1234.ts.net"
    assert st.rss_gb == pytest.approx(20.03, abs=0.01) and st.footprint_gb == pytest.approx(6.5)
    assert json.loads(st.to_json())["loopback_only"] is True


def test_start_session_agent_then_install(cfg, paths):
    r = FakeRunner(loaded=False)
    srv = serve.Server(cfg, paths, r, fake_http(), sleep=lambda s: None)
    st = srv.start(wait_s=1)
    assert st.health == "ok"
    assert paths.session_plist.exists() and not paths.installed_plist.exists()
    prog = plistlib.loads(paths.session_plist.read_bytes())["ProgramArguments"]
    assert prog[0].endswith("llama-server") and "--api-key-file" in prog
    srv.install()
    assert paths.installed_plist.exists() and not paths.session_plist.exists()
    srv.uninstall()
    assert not paths.installed_plist.exists() and not r.loaded


def test_start_refuses_bad_key(cfg, paths):
    paths.api_key.chmod(0o644)
    with pytest.raises(RuntimeError, match="API key"):
        serve.Server(cfg, paths, FakeRunner(loaded=False), fake_http()).start(wait_s=1)


def test_stop_is_idempotent(cfg, paths):
    r = FakeRunner(loaded=False)
    serve.Server(cfg, paths, r, fake_http(), sleep=lambda s: None).stop(wait_s=0.1)  # bootout rc 3: fine


def test_podman_backend_program(cfg, paths):
    pc = config.with_overrides(cfg, backend="podman")
    prog = serve.Server(pc, paths, FakeRunner(), fake_http()).program()
    assert "--device" in prog and "/dev/dri" in prog
    assert "127.0.0.1:8080:8080" in prog  # published on host loopback only
    i = prog.index(pc.image)
    inner = prog[i + 1 :]
    assert inner[inner.index("-m") + 1] == "/models/m.gguf"
    assert inner[inner.index("--api-key-file") + 1] == "/run/lmtune/api-key"


def test_listener_parse_flags_public_bind():
    text = LSOF + "llama-ser  4242 a 6u IPv4 0x1 0t0 TCP *:8080 (LISTEN)\n"
    srv_listeners = serve.parse_lsof_listeners(text)
    st = serve.Status(installed=False, agent=launchd.AgentState(True), health="ok", listeners=srv_listeners)
    assert not st.loopback_only


# -- CLI ----------------------------------------------------------------------------------------


def test_cli_init_and_config(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LMTUNE_CONFIG_DIR", str(tmp_path / "c"))
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    assert cli.main(["serve", "init", "--model", str(model)]) == 0
    assert (tmp_path / "c" / "serve.toml").exists()
    assert stat.S_IMODE((tmp_path / "c" / "api-key").stat().st_mode) == 0o600
    assert cli.main(["serve", "init", "--model", str(model)]) == 1  # refuses to overwrite
    assert cli.main(["key", "path"]) == 0
    out = capsys.readouterr().out
    assert str(tmp_path / "c" / "api-key") in out and "sk-lmtune-" not in out  # never printed unless `show`


def test_cli_missing_config_is_a_clean_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("LMTUNE_CONFIG_DIR", str(tmp_path / "none"))
    assert cli.main(["serve", "status"]) == 1
    assert "lmtune serve init" in capsys.readouterr().err


def test_cli_parser_has_every_command():
    p = cli.build_parser()
    for argv in (
        ["serve", "start"],
        ["serve", "stop"],
        ["serve", "status", "--json"],
        ["serve", "logs", "-f"],
        ["serve", "install"],
        ["expose", "on"],
        ["key", "rotate"],
        ["doctor"],
        ["tune", "info", "x.gguf"],
    ):
        assert p.parse_args(argv).fn


def test_logs_tail(cfg, paths):
    paths.log_dir.mkdir(parents=True)
    paths.server_log.write_text("\n".join(f"line {i}" for i in range(100)))
    assert serve.Server(cfg, paths, FakeRunner(), fake_http()).logs(3) == ["line 97", "line 98", "line 99"]


def test_model_path_untouched(cfg):
    assert isinstance(cfg.model, Path)


def test_start_when_already_loaded_waits_for_health(cfg, paths):
    """Regression: `serve install` loads the agent; a following `serve start` must wait for the
    model to finish loading instead of reporting the 503 'loading' state as a failure."""
    calls = {"n": 0}

    def http(url, headers, timeout):
        if url.endswith("/health"):
            calls["n"] += 1
            return (503, b'{"status":"loading"}') if calls["n"] < 3 else (200, b'{"status":"ok"}')
        return fake_http()(url, headers, timeout)

    st = serve.Server(cfg, paths, FakeRunner(loaded=True), http, sleep=lambda s: None).start(wait_s=60)
    assert st.health == "ok" and calls["n"] >= 3


class FlakyLaunchd(FakeRunner):
    """Reproduces the measured launchd race: bootstrap right after bootout fails with error 5,
    and launchctl print keeps reporting the service for a moment after bootout."""

    def __init__(self):
        super().__init__(loaded=True)
        self.bootstrap_failures = 0
        self.linger = 0

    def __call__(self, cmd):
        c = list(cmd)
        if c[:2] == ["launchctl", "bootout"]:
            self.calls.append(c)
            self.loaded, self.linger, self.bootstrap_failures = False, 2, 2
            return Result(0, "", "")
        if c[:2] == ["launchctl", "print"] and self.linger:
            self.calls.append(c)
            self.linger -= 1
            return Result(0, LAUNCHCTL_PRINT, "")
        if c[:2] == ["launchctl", "bootstrap"] and self.bootstrap_failures:
            self.calls.append(c)
            self.bootstrap_failures -= 1
            return Result(5, "", "Bootstrap failed: 5: Input/output error")
        return super().__call__(cmd)


def test_restart_survives_launchd_race(cfg, paths):
    r = FlakyLaunchd()
    srv = serve.Server(cfg, paths, r, fake_http(), sleep=lambda s: None)
    st = srv.restart(wait_s=5)
    assert st.health == "ok" and r.loaded
    assert sum(1 for c in r.calls if c[:2] == ["launchctl", "bootstrap"]) == 3  # two error-5s, then success


def test_bootstrap_other_errors_fail_fast(tmp_path):
    class Denied(FakeRunner):
        def __call__(self, cmd):
            if list(cmd)[:2] == ["launchctl", "bootstrap"]:
                return Result(1, "", "Operation not permitted")
            return super().__call__(cmd)

    with pytest.raises(RuntimeError, match="not permitted"):
        launchd.Launchd(LABEL, Denied(loaded=False), sleep=lambda s: None).bootstrap(tmp_path / "x.plist")
