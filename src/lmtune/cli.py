"""lmtune: tune, serve and expose a local LLM — measured, gated, private by default.

lmtune serve init|start|stop|restart|status|logs|install|uninstall|config
lmtune expose on|off|status [webui|searxng|dash]  tailnet-only HTTPS via `tailscale serve`
lmtune dash [install|uninstall|snapshot]  live dashboard of the stack on 127.0.0.1:8090 (Mac + phone)
lmtune key init|rotate|path|show     API key file (0600)
lmtune harness opencode [--write]    local-only OpenCode config + privacy wrapper (`oc`)
lmtune media serve|install|uninstall|status|verify|call   ffmpeg/exiftool/whisper as MCP tools: no network, one folder
lmtune stack up|down|status|logs|verify|update|install   Open WebUI + SearXNG, web via Tor (research, phone)
lmtune bench [--tokens N]            measure the running server (same prompts as the tuning run)
lmtune doctor                        everything that would make serving fail or results lie
lmtune tune info|plan|fetch|run|report
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import (
    __version__,
    config,
    dash,
    doctor,
    gguf,
    harness,
    hf,
    keys,
    machine,
    media,
    pipeline,
    profile,
    report,
    server,
    stack,
)
from .paths import Paths
from .serve import Server, Status

DEFAULT_MODEL = Path("~/models/qwen38/Qwen3.8-27B-Q4_0.gguf").expanduser()


# -- helpers ----------------------------------------------------------------------------


def _paths() -> Paths:
    return Paths.default()


def _server(paths: Paths) -> Server:
    return Server(config.load(paths.serve_config), paths)


def _fail(msg: str, code: int = 1) -> int:
    print(f"lmtune: {msg}", file=sys.stderr)
    return code


def _print_status(st: Status) -> None:
    a = st.agent
    if not a.loaded:
        print("server   stopped" + (" (installed: starts at login)" if st.installed else ""))
    else:
        mode = "sleeping, model unloaded" if st.sleeping else ("busy" if st.busy else "idle")
        print(
            f"server   {st.health} · {mode} · pid {a.pid} · "
            f"{'installed (starts at login)' if st.installed else 'session-only'}"
        )
        if st.model:
            print(f"model    {st.model}  ({st.build})")
        if st.rss_gb is not None:
            print(f"memory   RSS {st.rss_gb:.1f} GiB · footprint {st.footprint_gb or 0:.1f} GiB")
        if a.last_exit and a.last_exit not in ("(never exited)", "0"):
            print(f"last     exit {a.last_exit} (runs: {a.runs})")
    if st.listeners:
        print(f"listen   {', '.join(st.listeners)}{'' if st.loopback_only else '  ⚠ NOT loopback-only'}")
    e = st.exposure
    if e is not None:
        print(f"tailnet  {e.url or e.detail}")
    for p in st.config_problems + st.key_problems:
        print(f"problem  {p}")


# -- serve ------------------------------------------------------------------------------


def serve_init(a: argparse.Namespace) -> int:
    p = _paths()
    if p.serve_config.exists() and not a.force:
        return _fail(f"{p.serve_config} exists (use --force to overwrite)")
    created = keys.ensure(p.api_key)
    cfg = config.ServeConfig(model=Path(a.model).expanduser(), api_key_file=p.api_key)
    if a.port:
        cfg = config.with_overrides(cfg, port=a.port)
    p.config_dir.mkdir(parents=True, exist_ok=True)
    p.serve_config.write_text(config.to_toml(cfg))
    print(f"wrote {p.serve_config}")
    print(f"{'created' if created else 'kept'} API key {p.api_key} (0600)")
    for prob in cfg.problems():
        print(f"warning: {prob}")
    return 0


def serve_start(a: argparse.Namespace) -> int:
    srv = _server(_paths())
    t0 = time.monotonic()
    st = srv.start(wait_s=a.wait)
    _print_status(st)
    if st.health != "ok":
        return _fail(f"not healthy after {time.monotonic() - t0:.0f}s; see `lmtune serve logs`")
    print(f"ready in {time.monotonic() - t0:.0f}s at {srv.cfg.url}")
    return 0


def serve_stop(_: argparse.Namespace) -> int:
    _server(_paths()).stop()
    print("stopped")
    return 0


def serve_restart(a: argparse.Namespace) -> int:
    st = _server(_paths()).restart(wait_s=a.wait)
    _print_status(st)
    return 0 if st.health == "ok" else 1


def serve_status(a: argparse.Namespace) -> int:
    st = _server(_paths()).status()
    if a.json:
        print(st.to_json())
    else:
        _print_status(st)
    return 0 if (not st.agent.loaded or st.health == "ok") and st.loopback_only else 1


def serve_logs(a: argparse.Namespace) -> int:
    srv = _server(_paths())
    for line in srv.logs(a.lines):
        print(line)
    if a.follow:
        log = srv.paths.server_log
        with log.open() as f:
            f.seek(0, 2)
            try:
                while True:
                    line = f.readline()
                    if line:
                        print(line, end="", flush=True)
                    else:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                return 0
    return 0


def serve_install(_: argparse.Namespace) -> int:
    path = _server(_paths()).install()
    print(f"installed {path}: starts at login, restarts after a crash")
    return 0


def serve_uninstall(_: argparse.Namespace) -> int:
    _server(_paths()).uninstall()
    print("uninstalled; the server no longer starts at login")
    return 0


def serve_config(a: argparse.Namespace) -> int:
    srv = _server(_paths())
    if a.argv:
        print(" ".join(srv.program()))
    else:
        print(srv.paths.serve_config.read_text(), end="")
    return 0


# -- expose / key / doctor ----------------------------------------------------------------


def expose(a: argparse.Namespace) -> int:
    p = _paths()
    srv = _server(p)
    if a.target in ("webui", "searxng"):
        sc = stack.load(p.config_dir / "stack.toml")
        port, https = (
            (sc.webui_port, sc.webui_https_port) if a.target == "webui" else (sc.searxng_port, sc.searxng_https_port)
        )
        if not port and a.action != "off":
            return _fail("SearXNG isn't published: set searxng_port in stack.toml, then `lmtune stack up`")
    elif a.target == "dash":
        dc = dash.load(p.config_dir / "dash.toml")
        port, https = dc.port, dc.https_port
    else:
        port, https = srv.cfg.port, 443
    if a.action == "on":
        e = srv.tailscale.expose(port, https)
        print(f"tailnet-only: {e.url} -> {e.target}")
        if a.target == "searxng":
            print("SearXNG has no login: your tailnet ACL decides who can search with it")
        if a.target == "dash":
            print("the dashboard has no login (read-only, no prompt text): your tailnet ACL decides who sees it")
    elif a.action == "off":
        srv.tailscale.unexpose(https)
        print(f"{a.target} no longer exposed on the tailnet")
    else:
        e = srv.tailscale.status(port)
        print(e.url or e.detail)
    return 0


def dash_cmd(a: argparse.Namespace) -> int:
    p = _paths()
    if a.action == "install":
        print(f"installed {dash.install(p)}: the dashboard runs at login on 127.0.0.1 (`lmtune expose on dash`)")
        return 0
    if a.action == "uninstall":
        dash.uninstall(p)
        print("the dashboard no longer runs at login")
        return 0
    dc = dash.load(p.config_dir / "dash.toml")
    sc = stack.load(p.config_dir / "stack.toml")
    srv = _server(p)
    col = dash.Collector(srv, sc, dc, p, runner=srv.run)
    if a.action == "snapshot":
        print(json.dumps(col.state(), indent=1, default=str))
        return 0
    print(f"lmtune dash: http://127.0.0.1:{dc.port} (loopback only; tailnet: `lmtune expose on dash`)", flush=True)
    try:
        dash.serve_forever(col, dc, dc.tailnet_host or sc.tailnet_host)
    except OSError as e:
        return _fail(f"can't serve on 127.0.0.1:{dc.port}: {e}")
    except KeyboardInterrupt:
        pass
    return 0


def media_cmd(a: argparse.Namespace) -> int:
    p = _paths()
    mc = media.load(p.config_dir / "media.toml")
    if a.sub == "worker":  # what launchd runs, inside the sandbox; refuses to run outside one
        with contextlib.suppress(KeyboardInterrupt):
            media.serve_forever(mc, log=lambda m: print(m, flush=True))
        return 0
    if a.sub == "probe":  # `verify` runs this inside the sandbox
        for row in media.probes(mc, a.server_port, Path(a.secret)):
            print(json.dumps(dataclasses.asdict(row)), flush=True)
        return 0
    if a.sub == "serve":  # foreground, sandboxed, with this checkout's code
        keys.ensure(mc.key_file)
        mc.dir.mkdir(parents=True, exist_ok=True)
        argv = media.program(mc, p, media.Launcher.current())
        os.execv(argv[0], argv)  # noqa: S606 - fixed argv, no shell
    if a.sub == "install":
        path = media.install(mc, p)
        print(f"installed {path}: sandboxed media worker on 127.0.0.1:{mc.port}, folder {mc.dir}")
        print("next: `lmtune media verify`; `lmtune harness opencode --write` adds its tools to OpenCode")
        return 0
    if a.sub == "uninstall":
        media.uninstall(p)
        print("the media worker no longer runs at login")
        return 0
    if a.sub == "status":
        state = media.agent_state()
        print(f"agent   {'running, pid ' + str(state.pid) if state.pid else 'not running'}")
        print(f"folder  {mc.dir}")
        print(f"models  {', '.join(media.Tools(mc).models()) or 'none in ' + str(mc.models_dir)}")
        print(f"key     {'; '.join(keys.problems(mc.key_file)) or str(mc.key_file) + ' (0600)'}")
        return 0
    if a.sub == "verify":
        sc = config.load(p.serve_config)
        rows = media.verify(mc, p, sc.port, sc.api_key_file or p.api_key)
        for r in rows:
            print(f"{'✓' if r.ok else '✗'} {r.name:<20} must be {r.expect:<8} {r.detail}")
        bad = [r.name for r in rows if not r.ok]
        print(f"\n{len(rows) - len(bad)}/{len(rows)} as expected" + (f"; FAILED: {', '.join(bad)}" if bad else ""))
        return 1 if bad else 0
    # call: one tool, through the running worker (what a client sees)
    t0 = time.monotonic()
    code, res = media.call(mc, "tools/call", {"name": a.tool, "arguments": json.loads(a.arguments)})
    if code != 200 or not isinstance(res, dict) or "result" not in res:
        return _fail(f"HTTP {code}: {res}")
    print("\n".join(c.get("text", "") for c in res["result"]["content"]))
    print(f"\n[{time.monotonic() - t0:.1f} s{', tool error' if res['result'].get('isError') else ''}]")
    return 1 if res["result"].get("isError") else 0


def key(a: argparse.Namespace) -> int:
    p = _paths()
    if a.action == "init":
        print(f"{'created' if keys.ensure(p.api_key) else 'exists'}: {p.api_key}")
    elif a.action == "rotate":
        keys.write(p.api_key, keys.generate())
        print(f"rotated {p.api_key}; run `lmtune serve restart` and update clients")
    elif a.action == "path":
        print(p.api_key)
    elif a.action == "show":
        print(keys.read(p.api_key))
    return 0


def doctor_cmd(a: argparse.Namespace) -> int:
    p = _paths()
    try:
        cfg = config.load(p.serve_config)
    except config.ConfigError as e:
        return _fail(str(e))
    checks = doctor.run_all(cfg, Server(cfg, p))
    if a.json:
        print(json.dumps([c.__dict__ for c in checks], indent=1))
    else:
        icon = {"ok": "✓", "warn": "!", "fail": "✗"}
        for c in checks:
            print(
                f"{icon[c.level]} {c.name:<10} {c.detail}"
                + (f"\n    fix: {c.fix}" if c.fix and c.level != "ok" else "")
            )
    return 1 if any(c.level == "fail" for c in checks) else 0


def harness_cmd(a: argparse.Namespace) -> int:
    p = _paths()
    cfg = config.load(p.serve_config)
    mc = media.load(p.config_dir / "media.toml")
    m = mc if mc.key_file.exists() else None  # the media worker's tools, once it has a key
    if not a.write:
        print(json.dumps(harness.opencode_config(cfg, media=m), indent=2))
        return 0
    for path in harness.write_opencode(cfg, Path(a.config).expanduser(), Path(a.wrapper).expanduser(), media=m):
        print(f"wrote {path}")
    print(f"run `{Path(a.wrapper).name}` (opencode with update checks, catalog fetch and downloads disabled)")
    return 0


STACK_SERVICES = {"webui": stack.WEBUI, "searxng": stack.SEARXNG, "tor": stack.GATEWAY, "relay": stack.RELAY}


def stack_cmd(a: argparse.Namespace) -> int:
    p = _paths()
    st = stack.Stack(stack.load(p.config_dir / "stack.toml"), p)
    if a.action == "update":
        for ref in st.update():
            print(f"pinned {ref}")
        return 0
    if a.action == "install":
        print(f"installed {st.install()}: `lmtune stack up` runs at every login")
        return 0
    if a.action == "uninstall":
        st.uninstall()
        print("the stack no longer starts at login")
        return 0
    if a.action == "up":
        states = st.up()
    elif a.action == "down":
        st.down()
        print("stack stopped")
        return 0
    elif a.action == "logs":
        print(st.logs(STACK_SERVICES[a.service], a.lines))
        return 0
    elif a.action == "verify":
        probes = st.verify(fail_closed=a.fail_closed)
        for pr in probes:
            print(f"{'✓' if pr.ok else '✗'} {pr.name:38s} {pr.detail}")
        return 0 if all(pr.ok for pr in probes) else 1
    else:
        states = st.status()
    for c in states:
        state = c.status if c.exists else "absent"
        print(f"{c.name:16s} {state:10s} {c.mem_usage:22s} {c.image}")
    pct = stack.tor_bootstrap(st.logs(stack.GATEWAY, 200))
    print(f"Tor: {'bootstrapped' if pct == 100 else f'{pct}% bootstrapped' if pct is not None else 'not running'}")
    print(f"Open WebUI: http://127.0.0.1:{st.cfg.webui_port} (local) · {st.cfg.webui_url}")
    if st.cfg.searxng_port:
        print(f"SearXNG: http://127.0.0.1:{st.cfg.searxng_port} (local) · `lmtune expose on searxng` for the tailnet")
    return 0 if all(c.running for c in states) or a.action == "status" else 1


def bench_cmd(a: argparse.Namespace) -> int:
    srv = _server(_paths())
    st = srv.status()
    if st.health != "ok":
        return _fail(f"server is {st.health}; start it with `lmtune serve start`")
    prompts = json.loads(Path(a.prompts).expanduser().read_text())
    key = keys.read(srv.cfg.api_key_file) if srv.cfg.api_key_file else None
    gens = server.run_workload(srv.cfg.url, "bench", prompts, [a.seed], a.tokens, api_key=key)
    agg = server.aggregate(gens)
    acc = "-" if agg["acceptance"] is None else f"{agg['acceptance']:.0%}"
    print(
        f"\ngen {agg['gen_tps']:.2f} tok/s · prompt {agg['prompt_tps']:.1f} tok/s · MTP acceptance {acc} "
        f"· {agg['requests']} requests × {a.tokens} tokens"
    )
    return 0


# -- tune (measurement pipeline) ----------------------------------------------------------


def tune_info(a: argparse.Namespace) -> int:
    info = gguf.inspect(a.gguf)
    print(f"{info.name}  arch={info.arch}  blocks={info.block_count}  mtp_layers={info.mtp_layers}")
    print(
        f"attention layers with KV: {info.attention_layers}  kv heads={info.head_count_kv}  "
        f"k/v dims={info.key_length}/{info.value_length}"
    )
    for t in ("f16", "q8_0"):
        print(
            f"  KV {t}: {info.kv_bytes_per_token(t) / 1024:.1f} KiB/token, "
            f"{info.kv_bytes(a.ctx, t) / 2**30:.2f} GiB at {a.ctx} ctx"
        )
    print(f"publisher sampling defaults: {info.sampling or 'none'}")
    print(f"template reasoning defaults: {info.reasoning}")
    return 0


def tune_plan(a: argparse.Namespace) -> int:
    p, c = profile.load_profile(a.profile), profile.load_constraints(a.constraints)
    m = machine.detect(str(p.model.dir))
    files = hf.list_files(p.model.hf_repo)
    # weights must fit what the GPU may allocate AND leave the user's headroom in RAM
    gpu = (m.gpu_working_set_mib or 0) * 2**20 or m.ram_bytes
    budget = min(gpu, m.ram_bytes - c.memory_headroom_gb * 2**30)
    items, problems = hf.plan(
        files,
        p.model.candidates + p.model.extra_files,
        p.model.dir,
        budget_bytes=budget,
        ask_bytes=c.download_ask_gb * 1e9,
        disk_free=m.disk_free_bytes,
        min_free=c.min_free_disk_gb * 1e9,
    )
    print(
        f"{m.chip}, {m.ram_bytes / 2**30:.0f} GiB RAM, GPU working set {m.gpu_working_set_mib} MiB, "
        f"{m.disk_free_bytes / 1e9:.0f} GB free disk, power {m.power.source} {m.power.percent}%"
    )
    for i in items:
        state = "on disk" if i.have else ("NEEDS APPROVAL" if i.needs_approval else "will fetch")
        print(f"  {i.file.size / 1e9:6.2f} GB  {i.file.path:45s} {state}{'  ' + i.note if i.note else ''}")
    for pr in problems:
        print("  problem:", pr)
    cont = machine.contention()
    if not cont.clean:
        print("  contention:", cont.describe())
    return 0


def tune_fetch(a: argparse.Namespace) -> int:
    p, c = profile.load_profile(a.profile), profile.load_constraints(a.constraints)
    files = hf.list_files(p.model.hf_repo)
    for name in p.model.candidates + p.model.extra_files:
        f = files.get(name)
        if f is None:
            print(f"skip {name}: not in repo")
            continue
        if f.size > c.download_ask_gb * 1e9 and name not in a.approve:
            print(
                f"skip {name} ({f.size / 1e9:.1f} GB): over the {c.download_ask_gb:g} GB threshold; "
                f"rerun with --approve {name}"
            )
            continue
        out = hf.fetch(p.model.hf_repo, f, p.model.dir)
        print(f"ok {out} sha256 verified")
    return 0


def tune_run(a: argparse.Namespace) -> int:
    p, c = profile.load_profile(a.profile), profile.load_constraints(a.constraints)
    run_dir = Path(a.run_dir or Path("runs") / p.model.name / time.strftime("%Y%m%d-%H%M%S"))
    r = pipeline.Run(p, c, run_dir)
    stages = a.stages.split(",") if a.stages else pipeline.STAGES
    unknown = [s for s in stages if s not in pipeline.STAGES]
    if unknown:
        return _fail(f"unknown stage(s) {', '.join(unknown)}; choose from {', '.join(pipeline.STAGES)}")
    print(f"run dir: {run_dir}")
    for st in stages:
        print(f"== {st}")
        if st == "preflight":
            r.preflight(allow_contention=a.allow_contention)
        elif st == "report":
            md, sh = report.write(r)
            print(f"wrote {md} and {sh}")
        else:
            getattr(r, st)()
    return 0


def tune_report(a: argparse.Namespace) -> int:
    run_dir = Path(a.run_dir)
    r = pipeline.Run(profile.load_profile(a.profile), profile.load_constraints(a.constraints), run_dir)
    r.state = json.loads((run_dir / "state.json").read_text())
    md, sh = report.write(r)
    print(md.read_text())
    print(f"\nrun script: {sh}")
    return 0


# -- parser ---------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="lmtune", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--version", action="version", version=f"lmtune {__version__}")
    top = ap.add_subparsers(dest="cmd", required=True, metavar="command")

    sv = top.add_parser("serve", help="run the local model server").add_subparsers(dest="sub", required=True)
    s = sv.add_parser("init", help="write ~/.config/lmtune/serve.toml (+ API key)")
    s.add_argument("--model", default=str(DEFAULT_MODEL))
    s.add_argument("--port", type=int)
    s.add_argument("--force", action="store_true")
    s.set_defaults(fn=serve_init)
    for name, fn, hlp in (
        ("start", serve_start, "start under launchd (this login session)"),
        ("restart", serve_restart, "stop, then start"),
    ):
        s = sv.add_parser(name, help=hlp)
        s.add_argument("--wait", type=float, default=300, help="seconds to wait for healthy")
        s.set_defaults(fn=fn)
    sv.add_parser("stop", help="stop the server").set_defaults(fn=serve_stop)
    s = sv.add_parser("status", help="state, model, memory, listeners, tailnet")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=serve_status)
    s = sv.add_parser("logs", help="server log")
    s.add_argument("-n", "--lines", type=int, default=50)
    s.add_argument("-f", "--follow", action="store_true")
    s.set_defaults(fn=serve_logs)
    sv.add_parser("install", help="start at login (launchd agent)").set_defaults(fn=serve_install)
    sv.add_parser("uninstall", help="stop starting at login").set_defaults(fn=serve_uninstall)
    s = sv.add_parser("config", help="print serve.toml (or the full command with --argv)")
    s.add_argument("--argv", action="store_true")
    s.set_defaults(fn=serve_config)

    s = top.add_parser("expose", help="tailnet-only HTTPS access (tailscale serve)")
    s.add_argument("action", choices=["on", "off", "status"])
    s.add_argument(
        "target",
        nargs="?",
        choices=["server", "webui", "searxng", "dash"],
        default="server",
        help="server: llama-server on :443; webui: Open WebUI on :8443; searxng: SearXNG on :9443; "
        "dash: the dashboard on :8444",
    )
    s.set_defaults(fn=expose)
    s = top.add_parser("dash", help="live dashboard of the stack (loopback; `expose on dash` for the phone)")
    s.add_argument(
        "action",
        nargs="?",
        choices=["run", "install", "uninstall", "snapshot"],
        default="run",
        help="run (default): serve it; install: at login (launchd); snapshot: print the data once as JSON",
    )
    s.set_defaults(fn=dash_cmd)
    s = top.add_parser("key", help="API key file")
    s.add_argument("action", choices=["init", "rotate", "path", "show"])
    s.set_defaults(fn=key)
    s = top.add_parser("harness", help="client configs that use only the local server")
    s.add_argument("client", choices=["opencode"])
    s.add_argument("--write", action="store_true", help="write config + wrapper (backs up existing config)")
    s.add_argument("--config", default="~/.config/opencode/opencode.json")
    s.add_argument("--wrapper", default="~/.local/bin/oc")
    s.set_defaults(fn=harness_cmd)
    md = top.add_parser(
        "media", help="ffmpeg, exiftool, whisper.cpp as MCP tools: no network, one folder (sandboxed)"
    ).add_subparsers(dest="sub", required=True)
    for name, hlp in (
        ("serve", "run the worker in the foreground (sandboxed)"),
        ("install", "run it at login (launchd, sandboxed); creates the folder and its key"),
        ("uninstall", "stop running it at login"),
        ("status", "agent, folder, models, key"),
        ("verify", "probe the sandbox from inside: no network, no home folder, the media folder works"),
    ):
        md.add_parser(name, help=hlp).set_defaults(fn=media_cmd)
    s = md.add_parser("call", help="call one tool through the running worker")
    s.add_argument("tool")
    s.add_argument("arguments", nargs="?", default="{}", help='JSON, e.g. \'{"path": "talk.m4a"}\'')
    s.set_defaults(fn=media_cmd)
    md.add_parser("worker").set_defaults(fn=media_cmd)  # internal: the sandboxed server
    s = md.add_parser("probe")  # internal: verify's probes, run inside the sandbox
    s.add_argument("--server-port", type=int, default=8080)
    s.add_argument("--secret", default=str(Path("~/.config/lmtune/api-key").expanduser()))
    s.set_defaults(fn=media_cmd)
    s = top.add_parser("stack", help="Open WebUI + private SearXNG, web only via Tor (podman)")
    s.add_argument("action", choices=["up", "down", "status", "logs", "verify", "update", "install", "uninstall"])
    s.add_argument("service", nargs="?", choices=sorted(STACK_SERVICES), default="webui", help="for logs")
    s.add_argument("-n", "--lines", type=int, default=50)
    s.add_argument(
        "--fail-closed", action="store_true", help="verify: also stop Tor and check research fails, then restart it"
    )
    s.set_defaults(fn=stack_cmd)
    s = top.add_parser("bench", help="measure decode speed of the running server")
    s.add_argument("--tokens", type=int, default=512)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--prompts", default=str(Path(__file__).resolve().parent / "data" / "prompts.json"))
    s.set_defaults(fn=bench_cmd)
    s = top.add_parser("doctor", help="check config, runtime, memory, security, contention")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=doctor_cmd)

    tn = top.add_parser("tune", help="measure and pick the fastest correct config").add_subparsers(
        dest="sub", required=True
    )
    s = tn.add_parser("info", help="inspect a GGUF")
    s.add_argument("gguf")
    s.add_argument("--ctx", type=int, default=65536)
    s.set_defaults(fn=tune_info)
    for name, fn, hlp in (
        ("plan", tune_plan, "what fits, what would download"),
        ("fetch", tune_fetch, "download + verify candidates"),
        ("run", tune_run, "run the tuning pipeline"),
        ("report", tune_report, "re-render a run's report"),
    ):
        s = tn.add_parser(name, help=hlp)
        s.add_argument("profile")
        s.add_argument("--constraints", default=str(profile.DEFAULT_CONSTRAINTS))
        if name == "fetch":
            s.add_argument("--approve", nargs="*", default=[])
        if name == "run":
            s.add_argument("--stages")
            s.add_argument("--run-dir")
            s.add_argument("--allow-contention", action="store_true")
        if name == "report":
            s.add_argument("run_dir")
        s.set_defaults(fn=fn)
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        code: int = a.fn(a)
    except (config.ConfigError, RuntimeError, FileNotFoundError) as e:
        return _fail(str(e))
    return code


if __name__ == "__main__":
    sys.exit(main())
