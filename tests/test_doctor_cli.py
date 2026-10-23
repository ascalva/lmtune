"""doctor checks and CLI command bodies with machine/server facts faked."""

import json
from pathlib import Path

import pytest

from lmtune import cli, config, doctor, flags, keys, machine, serve
from lmtune.paths import Paths
from test_gguf import hybrid_meta, write_gguf
from test_serve_stack import FakeRunner, fake_http

HELP = (Path(__file__).parent / "fixtures/llama-server-help-b11669.txt").read_text()


def mach(ram_gib=32, gpu_mib=25559, low_power=False):
    return machine.Machine(
        "Apple M2 Pro", ram_gib * 2**30, gpu_mib, 0, 10**11, machine.Power("AC", 100, "charged", low_power)
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    paths = Paths(tmp_path / "cfg", tmp_path / "state", tmp_path / "logs", tmp_path / "agents")
    model = write_gguf(tmp_path / "m.gguf", hybrid_meta(), ["blk.0.attn_q.weight"])
    with model.open("ab") as f:  # pretend 15 GiB of weights without writing them
        f.truncate(15 * 2**30)
    keys.write(paths.api_key, keys.generate())
    cfg = config.ServeConfig(model=model, api_key_file=paths.api_key)
    f, a = flags.parse_help(HELP)
    monkeypatch.setattr(flags, "discover", lambda tool: flags.ToolFlags(tool, 11669, "x", f, a))
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/opt/homebrew/bin/{name}")
    return cfg, paths


def test_memory_budget_ok_warn_fail(setup):
    cfg, _ = setup
    assert doctor.check_memory(cfg, mach()).level == "ok"  # 15 + 4 KV + 2 cache + 1.5 = 22.5 of 32
    assert doctor.check_memory(cfg, mach(ram_gib=24, gpu_mib=0)).level == "warn"
    assert doctor.check_memory(cfg, mach(gpu_mib=16000)).level == "fail"  # weights + KV exceed the GPU set
    missing = config.with_overrides(cfg, model=Path("/nope.gguf"))
    assert doctor.check_memory(missing, mach()).level == "fail"


def test_runtime_flags_checked_against_build(setup):
    cfg, _ = setup
    assert doctor.check_runtime(cfg).level == "ok"
    bad = config.with_overrides(cfg, spec=("--spec-type", "draft-mtp", "--spec-renamed-flag", "2"))
    c = doctor.check_runtime(bad)
    assert c.level == "fail" and "--spec-renamed-flag" in c.detail


def test_key_and_config_checks(setup):
    cfg, paths = setup
    assert doctor.check_key(cfg).level == "ok"
    paths.api_key.chmod(0o644)
    assert doctor.check_key(cfg).level == "fail"
    assert doctor.check_key(config.with_overrides(cfg, api_key_file=None)).level == "fail"
    assert doctor.check_config(config.with_overrides(cfg, host="0.0.0.0")).level == "fail"


def test_server_checks_report_state(setup):
    cfg, paths = setup
    checks = {c.name: c for c in doctor.check_server(serve.Server(cfg, paths, FakeRunner(), fake_http()))}
    assert checks["server"].level == "ok" and "sleeping" in checks["server"].detail
    assert checks["listen"].level == "ok" and checks["tailnet"].level == "ok"
    stopped = {c.name: c for c in doctor.check_server(serve.Server(cfg, paths, FakeRunner(loaded=False), fake_http()))}
    assert stopped["server"].level == "warn"


def test_run_all_never_crashes(setup, monkeypatch):
    cfg, paths = setup
    monkeypatch.setattr(machine, "detect", lambda *a, **k: mach(low_power=True))
    monkeypatch.setattr(machine, "contention", lambda **k: machine.Contention())
    monkeypatch.setattr(machine, "swap_used_mb", lambda: 4000.0)
    monkeypatch.setattr(machine, "gpu_idle_utilization", lambda **k: 20.0)
    monkeypatch.setattr(doctor.gguf, "inspect", lambda p: (_ for _ in ()).throw(OSError("boom")))
    checks = {c.name: c for c in doctor.run_all(cfg, serve.Server(cfg, paths, FakeRunner(), fake_http()))}
    assert checks["memory"].level == "fail" and "crashed" in checks["memory"].detail  # reported, not raised
    assert checks["gpu"].level == "warn" and checks["power"].level == "warn"
    assert checks["contention"].level == "ok"


# -- CLI command bodies -------------------------------------------------------------------------


@pytest.fixture
def cli_env(setup, monkeypatch, tmp_path):
    cfg, paths = setup
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.serve_config.write_text(config.to_toml(cfg))
    monkeypatch.setattr(cli, "_paths", lambda: paths)
    runner = FakeRunner()
    monkeypatch.setattr(
        cli,
        "_server",
        lambda p: serve.Server(config.load(p.serve_config), p, runner, fake_http(), sleep=lambda s: None),
    )
    return paths, runner


def test_cli_status_text_and_json(cli_env, capsys):
    assert cli.main(["serve", "status"]) == 0
    out = capsys.readouterr().out
    assert "sleeping, model unloaded" in out and "127.0.0.1:8080" in out and "ts.net" in out
    assert cli.main(["serve", "status", "--json"]) == 0
    assert '"loopback_only": true' in capsys.readouterr().out


def test_cli_lifecycle(cli_env, capsys):
    paths, runner = cli_env
    runner.loaded = False
    assert cli.main(["serve", "start", "--wait", "1"]) == 0
    assert "ready in" in capsys.readouterr().out
    assert cli.main(["serve", "install"]) == 0 and paths.installed_plist.exists()
    assert cli.main(["serve", "restart", "--wait", "1"]) == 0
    assert cli.main(["serve", "stop"]) == 0
    assert cli.main(["serve", "uninstall"]) == 0 and not paths.installed_plist.exists()


def test_cli_config_argv_never_contains_key(cli_env, capsys):
    paths, _ = cli_env
    assert cli.main(["serve", "config", "--argv"]) == 0
    out = capsys.readouterr().out
    assert "--api-key-file" in out and keys.read(paths.api_key) not in out


def test_cli_expose_and_key(cli_env, capsys):
    paths, _ = cli_env
    assert cli.main(["expose", "status"]) == 0
    assert "mac" in capsys.readouterr().out
    old = keys.read(paths.api_key)
    assert cli.main(["key", "rotate"]) == 0 and keys.read(paths.api_key) != old


def test_cli_expose_searxng_needs_the_published_port(cli_env, capsys):
    paths, runner = cli_env
    assert cli.main(["expose", "on", "searxng"]) == 1  # nothing on loopback to expose yet
    assert "searxng_port" in capsys.readouterr().err
    assert not any("--bg" in c for c in runner.calls)
    (paths.config_dir / "stack.toml").write_text("[stack]\nsearxng_port = 8888\n")
    assert cli.main(["expose", "on", "searxng"]) == 0
    assert any(c[-4:] == ["serve", "--bg", "--https=9443", "8888"] for c in runner.calls)
    assert "no login" in capsys.readouterr().out  # the ACL reminder
    assert cli.main(["expose", "off", "searxng"]) == 0
    assert any(c[-3:] == ["serve", "--https=9443", "off"] for c in runner.calls)


def test_harness_opencode_local_only(cli_env, tmp_path, capsys):
    paths, _ = cli_env
    cfgfile, wrapper = tmp_path / "oc" / "opencode.json", tmp_path / "bin" / "oc"
    cfgfile.parent.mkdir()
    cfgfile.write_text('{"old": true}')
    assert cli.main(["harness", "opencode", "--write", "--config", str(cfgfile), "--wrapper", str(wrapper)]) == 0
    data = json.loads(cfgfile.read_text())
    assert data["enabled_providers"] == ["lmtune"] and data["share"] == "disabled" and data["autoupdate"] is False
    prov = data["provider"]["lmtune"]["options"]
    assert prov["baseURL"] == "http://127.0.0.1:8080/v1"
    assert prov["apiKey"] == f"{{file:{paths.api_key}}}"  # a reference, never the key itself
    assert keys.read(paths.api_key) not in cfgfile.read_text()
    assert (cfgfile.parent / "opencode.json.bak").read_text() == '{"old": true}'  # existing config backed up
    w = wrapper.read_text()
    assert "OPENCODE_DISABLE_MODELS_FETCH=1" in w and "OPENCODE_DISABLE_LSP_DOWNLOAD=1" in w and "exec " in w
    assert wrapper.stat().st_mode & 0o111
