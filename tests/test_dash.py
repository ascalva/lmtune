"""`lmtune dash`: log parsing, source parsers, collection (never /slots), HTTP surface, CLI wiring."""

import base64
import hashlib
import http.client
import json
import plistlib
import re
import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from lmtune import cli, config, dash, flags, keys, launchd, serve, stack, tailscale
from lmtune.paths import Paths
from lmtune.proc import Result
from test_serve_stack import LAUNCHCTL_PRINT, TS_JSON

FIXTURES = Path(__file__).parent / "fixtures"
LOG = (FIXTURES / "server-log-b11669.txt").read_text()
T0 = 1_792_687_744.0  # a server process start (Unix)


def feed(text: str) -> dash.LogState:
    st = dash.LogState()
    for line in text.splitlines():
        st.feed(line)
    return st


# -- server log -----------------------------------------------------------------------------------


def test_log_finished_requests_carry_speeds_and_acceptance():
    st = feed(LOG)
    first, last = st.done[0], st.done[-1]
    assert [r.task for r in st.done] == [3, 315]
    assert (first.prompt_tokens, first.prompt_tps, first.gen_tokens, first.gen_tps) == (55, 44.92, 8, 14.49)
    assert (first.draft_accepted, first.draft_generated, first.ctx_tokens) == (4, 5, 62)
    assert last.prompt_tokens == 6672 and last.prompt_tps == 93.71 and last.gen_tps == 14.69
    assert last.acceptance == pytest.approx(246 / 350) and last.phase == "done"
    assert last.ended - last.started == pytest.approx(109 * 60 + 6.006489 - (107 * 60 + 26.144753))
    j = last.to_json(T0)
    assert j["started"] == pytest.approx(T0 + 107 * 60 + 26.144753) and j["elapsed_s"] == pytest.approx(99.9, abs=0.1)


def test_log_lifecycle_sleep_clears_unfinished_and_wake_counts_as_activity():
    lines = LOG.splitlines()

    def upto(prefix):
        return feed("\n".join(lines[: next(i for i, x in enumerate(lines) if x.startswith(prefix)) + 1]))

    mid = upto("0.57.852.166")  # task 10 decoding
    assert mid.active[10].phase == "decode" and mid.active[10].gen_tokens == 100 and mid.active[10].live_tps == 18.53
    asleep = upto("45.53.255.849")
    assert asleep.lifecycle == "sleep" and not asleep.active  # the unload ends whatever was running
    woke = upto("92.39.407.415")
    assert woke.lifecycle == "wake" and woke.last_activity == pytest.approx(92 * 60 + 39.407415)
    prefill = upto("108.08.646.306")
    r = prefill.active[315]
    assert (r.phase, r.progress, r.live_tps) == ("prefill", 0.61, 111.14)


def test_log_restart_discards_the_previous_process():
    st = feed(LOG + "\n0.00.010.000 I srv  llama_server: initializing ...\n")
    assert not st.done and not st.active and st.lifecycle == "started" and st.clock == pytest.approx(0.01)


def test_log_ignores_noise_and_unknown_tasks():
    st = feed("garbage\n1.00.000.000 I slot print_timing: id  0 | task 99 | n_gen = 5, tg = 1 t/s, tg_3s = 1 t/s\n")
    assert not st.done and not st.active


def test_log_follower_is_incremental_and_survives_rotation(tmp_path):
    log = tmp_path / "server.log"
    lines = LOG.splitlines(keepends=True)
    log.write_text("".join(lines[:18]) + lines[18][:20])  # task 3 released, then a line cut short
    f = dash.LogFollower(log)
    assert [r.task for r in f.update().done] == [3]
    with log.open("a") as fh:
        fh.write(lines[18][20:] + "".join(lines[19:]))
    assert [r.task for r in f.update().done] == [3, 315]
    assert f.update() is f.state  # nothing new: nothing reparsed
    log.unlink()
    log.write_text("".join(lines[:9]))  # rotated: a new file
    assert not f.update().done and f.state.lifecycle == "loaded"


def test_log_follower_starts_from_the_tail_mid_line(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(LOG)
    tail = "".join(LOG.splitlines(keepends=True)[37:])  # from just before task 315 starts
    st = dash.LogFollower(log, tail_bytes=len(tail.encode()) + 5).update()
    assert [r.task for r in st.done] == [315]  # first (partial) line dropped, earlier requests out of the window


def test_summary_medians_and_pooled_acceptance():
    reqs = [
        dash.Request(1, 0, phase="done", gen_tokens=500, gen_tps=14.0, prompt_tokens=900, prompt_tps=90.0),
        dash.Request(2, 0, phase="done", gen_tokens=500, gen_tps=16.0, prompt_tokens=900, prompt_tps=80.0),
        dash.Request(3, 0, phase="done", gen_tokens=3, gen_tps=40.0, draft_accepted=3, draft_generated=4),
        dash.Request(4, 0, phase="done", gen_tokens=100, gen_tps=15.0, draft_accepted=60, draft_generated=76),
    ]
    s = dash.summarize(reqs)
    assert s["gen_tps"] == 15.0  # the 3-token request is too short to count
    assert s["prompt_tps"] == 85.0 and s["acceptance"] == pytest.approx(63 / 80) and s["requests"] == 4
    assert dash.summarize([])["gen_tps"] is None


@pytest.mark.parametrize(
    "case",
    [
        (False, 0, None, None, False, None, "stopped"),
        (False, 200, False, None, False, None, "idle"),  # started by hand, no agent: still serving
        (True, 0, None, None, False, None, "down"),
        (True, 503, None, None, False, None, "loading"),
        (True, 500, None, None, False, None, "error"),
        (True, 200, True, "sleep", False, None, "asleep"),
        (True, 200, True, "wake", False, None, "waking"),  # /props says sleeping until the reload ends
        (True, 200, True, "stale-wake", False, None, "asleep"),  # a wake minutes ago has finished
        (True, 200, False, "wake", True, None, "busy"),  # no metrics: the log decides
        (True, 200, False, "wake", False, None, "idle"),
        (True, 200, False, "wake", True, 0.0, "idle"),  # metrics outrank a log line not yet written
        (True, 200, False, "wake", False, 1.0, "busy"),
    ],
)
def test_model_state(case):
    loaded, health, sleeping, lifecycle, active, processing, want = case
    log = dash.LogState(lifecycle="wake" if lifecycle == "stale-wake" else lifecycle, lifecycle_at=1000.0)
    if active:
        log.active[1] = dash.Request(1, 0)
    now = 1000.0 + (dash.WAKE_MAX_S + 1 if lifecycle == "stale-wake" else 5)
    assert dash.model_state(loaded, health, sleeping, log, processing, now_clock=now) == want


# -- machine and stack parsers --------------------------------------------------------------------

SYSCTL = """kern.memorystatus_level: 81
kern.memorystatus_vm_pressure_level: 2
vm.swapusage: total = 3072.00M  used = 1754.50M  free = 1317.50M  (encrypted)
hw.memsize: 34359738368
"""
VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                     4155.
Pages speculative:                              3191.
Pages wired down:                             156787.
Pages purgeable:                               10774.
"Translation faults":                     3557058560.
File-backed pages:                           1301319.
Anonymous pages:                              363493.
Pages occupied by compressor:                 218954.
"""
IOREG = (
    '"PerformanceStatistics" = {"In use system memory (driver)"=0,"Alloc system memory"=23881383936,'
    '"Renderer Utilization %"=19,"Device Utilization %"=20,"In use system memory"=395460608}'
)


def test_parse_sysctl_and_vm_stat():
    m = dash.parse_sysctl(SYSCTL)
    assert m["pressure"] == "warning" and m["free_pct"] == 81 and m["ram_bytes"] == 32 * 2**30
    assert m["swap_total_bytes"] == 3 * 2**30 and m["swap_used_bytes"] == int(1754.5 * 2**20)
    assert dash.parse_sysctl("")["pressure"] == "unknown"
    b = dash.parse_vm_stat(VM_STAT)
    page = 16384
    assert b["app"] == (363493 - 10774) * page and b["cached"] == (1301319 + 10774) * page
    assert b["wired"] == 156787 * page and b["compressed"] == 218954 * page and b["free"] == (4155 + 3191) * page


def test_parse_gpu_ps_footprint():
    assert dash.parse_gpu(IOREG) == {"device_pct": 20, "renderer_pct": 19, "in_use_bytes": 395460608}
    assert dash.parse_gpu("")["device_pct"] is None
    p = dash.parse_ps("14604176   0.1 Thu Oct  2 12:49:04 2026    \n")
    assert p["rss_bytes"] == 14604176 * 1024 and p["cpu_pct"] == 0.1 and p["started"] is not None
    assert dash.parse_ps("") == {}
    assert dash.parse_footprint("llama-server [1]: 64-bit    Footprint: 4.9 GB (16384 bytes per page)") == int(
        4.9 * 2**30
    )
    assert dash.parse_footprint("nope") is None


def test_cpu_busy_handles_counter_wrap():
    assert dash.cpu_busy_pct((100, 100, 800, 0), (200, 200, 1400, 0)) == pytest.approx(25.0)
    assert dash.cpu_busy_pct((2**32 - 10, 0, 0, 0), (10, 0, 20, 0)) == pytest.approx(50.0)
    assert dash.cpu_busy_pct((1, 1, 1, 1), (1, 1, 1, 1)) is None


PODMAN_PS = json.dumps(
    [
        {"Names": ["lmtune-webui"], "State": "running", "Status": "Up 7 hours", "StartedAt": 1792704255},
        {"Names": ["anon-tor"], "State": "running", "Status": "Up 8 hours", "StartedAt": 1792699693},
        {"Names": ["lmtune-tor"], "State": "exited", "Status": "Exited (1) 2 minutes ago"},
    ]
)
PODMAN_STATS = json.dumps(
    [
        {"name": "lmtune-webui", "mem_usage": "688.2MB / 1.611GB", "cpu_percent": "0.83%"},
        {"name": "anon-tor", "mem_usage": "17.14MB / 536.9MB", "cpu_percent": "--"},
    ]
)


def test_parse_containers_sizes_and_order():
    assert dash.parse_size("688.2MB") == 688_200_000 and dash.parse_size("1.611GB") == 1_611_000_000
    assert dash.parse_size("512kB") == 512_000 and dash.parse_size("0B") == 0 and dash.parse_size("?") is None
    cs = dash.parse_containers(PODMAN_PS, PODMAN_STATS)
    assert [c["name"] for c in cs] == ["lmtune-tor", "lmtune-webui", "anon-tor"]  # ours first
    web = cs[1]
    assert web["mem_bytes"] == 688_200_000 and web["mem_limit_bytes"] == 1_611_000_000 and web["cpu_pct"] == 0.83
    assert cs[0]["state"] == "exited" and cs[0]["mem_bytes"] is None and cs[2]["cpu_pct"] is None
    assert dash.parse_containers("not json", "") == []


def test_parse_meminfo():
    m = dash.parse_meminfo("MemTotal:        1988200 kB\nMemAvailable:     640828 kB\n")
    assert m == {"total_bytes": 1988200 * 1024, "available_bytes": 640828 * 1024}


TOR_LOG = """2026-10-22T17:25:51.000000000-04:00 Oct 22 21:25:51.000 [notice] Bootstrapped 95% (circuit_create)
2026-10-22T17:25:51.100000000-04:00 Oct 22 21:25:51.000 [notice] Bootstrapped 100% (done): Done
2026-10-22T17:40:00.000000000-04:00 Oct 22 21:40:00.000 [warn] Problem bootstrapping to some.example.org
2026-10-22T19:00:00.000000000-04:00 Oct 22 23:00:00.000 [warn] Have tried resolving secret-site.example
"""


def test_parse_tor_counts_recent_warnings_without_their_text():
    now = dash._iso_time("2026-10-22T19:30:00-04:00")
    t = dash.parse_tor(TOR_LOG, now)
    assert t["bootstrap_pct"] == 100 and t["warnings_1h"] == 1
    assert t["last_warning"] == dash._iso_time("2026-10-22T19:00:00-04:00")
    assert "secret-site" not in json.dumps(t)


SEARX_LOG = "".join(
    f"2026-10-23T00:29:{s}.000-04:00 2026-10-23 04:29:{s},000 {line}\n"
    for s, line in (
        ("03", "ERROR:searx.engines.brave: engine timeout"),
        ("11", "WARNING:searx.network.brave: ConnectionError: the server has disconnected, retrying"),
        ("12", "ERROR:searx.engines.bing: Too many requests (suspended_time=180)"),
        ("13", "WARNING:searx.network.yep: HTTP Request failed: GET https://api.yep.com/search?query=private+question"),
        ("14", "ERROR:searx.engines.google: SearXNG is blocked (CAPTCHA)"),
        ("15", "ERROR:searx: call to ResultContainer.add_unresponsive_engine after close"),
    )
)


def test_parse_searxng_classifies_per_engine_and_never_keeps_queries():
    engines = dash.parse_searxng(SEARX_LOG, stack.SEARCH_ENGINES)
    by = {e["name"]: e for e in engines}
    assert [e["name"] for e in engines] == list(stack.SEARCH_ENGINES)
    assert by["brave"]["errors"] == {"timeout": 1, "connection": 1}
    assert by["bing"]["errors"] == {"rate-limited": 1}
    assert by["yep"]["errors"] == {"error": 1} and by["wikidata"]["errors"] == {}
    assert "google" not in by  # not one of ours
    assert "private" not in json.dumps(engines)
    assert dash.classify_engine_error("SearXNG is blocked (CAPTCHA)") == "captcha"
    assert dash.classify_engine_error("access denied") == "blocked"


TS_STATUS = {
    "MagicDNSSuffix": "tail1234.ts.net",
    "Self": {"DNSName": "mac.tail1234.ts.net."},
    "BackendState": "Running",
    "Peer": {
        "a": {
            "DNSName": "phone.tail1234.ts.net.",
            "HostName": "localhost",
            "OS": "iOS",
            "TailscaleIPs": ["100.64.0.2", "fd7a::1"],
            "Online": True,
            "Active": True,
            "CurAddr": "192.0.2.1:41641",
            "Relay": "nyc",
            "LastHandshake": "2026-10-23T00:18:25.228497833-04:00",
        },
        "b": {
            "DNSName": "pi.tail1234.ts.net.",
            "OS": "linux",
            "Online": False,
            "LastHandshake": "0001-01-01T00:00:00Z",
        },
        "c": {"DNSName": "us-lax-wg-402.mullvad.ts.net.", "Online": True, "Tags": ["tag:mullvad-exit-node"]},
    },
}


def test_tailnet_peers_are_this_tailnets_devices_only():
    peers = dash.tailnet_peers(TS_STATUS)
    assert [p["name"] for p in peers] == ["phone", "pi"]  # active first; Mullvad left out
    ph = peers[0]
    assert ph["ip"] == "100.64.0.2" and ph["direct"] and ph["last_handshake"] == pytest.approx(1792729105.228497)
    assert peers[1]["last_handshake"] is None and not peers[1]["online"]


def test_serve_exposures_flag_funnel():
    data = {**TS_JSON, "AllowFunnel": {"mac.tail1234.ts.net:8444": True}}
    data["Web"] = {
        **TS_JSON["Web"],
        "mac.tail1234.ts.net:8444": {"Handlers": {"/": {"Proxy": "http://127.0.0.1:8090"}}},
    }
    ex = dash.serve_exposures(data)
    assert ex[0] == {"url": "https://mac.tail1234.ts.net", "target": "http://127.0.0.1:8080", "funnel": False}
    assert ex[1]["url"].endswith(":8444") and ex[1]["funnel"] is True
    assert dash.serve_exposures({}) == []


def test_parse_listeners_flags_non_loopback():
    lsof = (
        "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\n"
        "llama-ser 1 u 3u IPv4 0x1 0t0 TCP 127.0.0.1:8080 (LISTEN)\n"
        "gvproxy 2 u 3u IPv4 0x1 0t0 TCP *:3000 (LISTEN)\n"
        "ControlCe 3 u 3u IPv4 0x1 0t0 TCP *:7000 (LISTEN)\n"
    )
    ls = dash.parse_listeners(lsof, {8080, 3000})
    assert [(x["addr"], x["loopback"]) for x in ls] == [("*:3000", False), ("127.0.0.1:8080", True)]


# -- sources and the collector --------------------------------------------------------------------


def test_source_ttl_errors_and_background():
    now = [0.0]
    calls = []

    def fn():
        calls.append(now[0])
        if len(calls) == 2:
            raise RuntimeError("podman down")
        return len(calls)

    s = dash.Source("x", 10, fn, clock=lambda: now[0])
    assert s.get() == 1 and s.get() == 1 and calls == [0.0]  # cached within the TTL
    now[0] = 11
    assert s.get() == 1 and s.error == "RuntimeError: podman down"  # failure keeps the last value
    now[0] = 22
    assert s.get() == 3 and s.error is None
    bg = dash.Source("bg", 10, lambda: "v", background=True, clock=lambda: now[0])
    assert bg.get() == "v"  # the first call waits for data


class FakeHost:
    """launchctl, ps, footprint, sysctl, vm_stat, ioreg, podman, tailscale and lsof, plus the
    llama-server HTTP API. Records every URL so tests can prove /slots is never touched."""

    def __init__(self, *, sleeping=False, metrics=True, loaded=True, serving=None):
        self.sleeping, self.metrics, self.loaded = sleeping, metrics, loaded
        self.serving = loaded if serving is None else serving  # something answers on :8080
        self.calls: list[list[str]] = []
        self.urls: list[str] = []

    def __call__(self, cmd: Sequence[str]) -> Result:
        c = list(cmd)
        self.calls.append(c)
        if c[:2] == ["launchctl", "print"]:
            return Result(0, LAUNCHCTL_PRINT, "") if self.loaded else Result(113, "", "")
        if c[0] == "launchctl":
            return Result(0, "", "")
        if c[0] == "ps":
            return Result(0, PS_LINE, "")
        if c[0] == "footprint":
            return Result(0, "llama-server [4242]: 64-bit    Footprint: 6.5 GB (16384 bytes per page)\n", "")
        if c[0] == "sysctl":
            return Result(0, SYSCTL, "")
        if c[0] == "vm_stat":
            return Result(0, VM_STAT, "")
        if c[0] == "ioreg":
            return Result(0, IOREG, "")
        if c[0] == "lsof" and "-t" in c:
            return Result(0, "68263\n", "") if self.serving else Result(1, "", "")
        if c[0] == "lsof":
            return Result(
                0,
                "COMMAND PID USER FD TYPE DEVICE SIZE/OFF NODE NAME\nl 1 u 3u IPv4 0 0t0 TCP 127.0.0.1:8080 (LISTEN)\n",
                "",
            )
        if c[0] == "ts":  # tailscale
            return Result(0, json.dumps(TS_STATUS if c[1] == "status" else TS_JSON), "")
        if c[1:3] == ["ps", "-a"]:
            return Result(0, PODMAN_PS, "")
        if c[1] == "stats":
            return Result(0, PODMAN_STATS, "")
        if c[1:3] == ["machine", "ssh"]:
            return Result(0, "MemTotal: 1988200 kB\nMemAvailable: 640828 kB\n", "")
        if c[1] == "logs":
            return Result(0, TOR_LOG if c[-1] == stack.GATEWAY else SEARX_LOG, "")
        return Result(1, "", f"unexpected {c}")

    def http(self, url: str, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
        self.urls.append(url)
        path = url.split(":8080", 1)[1]
        if not self.serving:
            return 0, b""
        if path == "/health":
            return 200, b'{"status":"ok"}'
        assert headers.get("Authorization", "").startswith("Bearer sk-lmtune-")
        if path == "/props":
            return 200, json.dumps(
                {"model_alias": "qwen3.8-27b", "build_info": "b11669-x", "is_sleeping": self.sleeping}
            ).encode()
        if path == "/metrics":
            if not self.metrics:
                return 501, b'{"error":{"code":501}}'
            return (
                200,
                b"llamacpp:requests_processing 0\nllamacpp:requests_deferred 2\nllamacpp:tokens_predicted_total 4242\n",
            )
        if path == "/slots":
            return 200, b'[{"id":0,"is_processing":true}]'
        return 404, b""


@pytest.fixture
def env(tmp_path):
    paths = Paths(tmp_path / "cfg", tmp_path / "state", tmp_path / "logs", tmp_path / "agents")
    paths.log_dir.mkdir(parents=True)
    paths.server_log.write_text(LOG)
    keys.write(paths.api_key, keys.generate())
    cfg = config.ServeConfig(model=tmp_path / "m.gguf", api_key_file=paths.api_key)
    return paths, cfg


PS_LINE = "21000000  3.0 Thu Oct 22 12:49:04 2026\n"
STARTED = dash.parse_ps(PS_LINE)["started"]  # local time, whatever the test machine's zone


def server(env, host: FakeHost) -> serve.Server:
    paths, cfg = env
    srv = serve.Server(cfg, paths, host, host.http)
    srv.tailscale.exe = "ts"
    return srv


def collector(env, host: FakeHost, **kw) -> dash.Collector:
    paths, _ = env
    ticks = iter([(0, 0, 0, 0), (30, 10, 60, 0), (60, 20, 120, 0)] * 10)
    return dash.Collector(
        server(env, host),
        stack.StackConfig(podman="pm"),
        dash.DashConfig(),
        paths,
        runner=host,
        clock=lambda: STARTED + 110 * 60,
        cpu_ticks=lambda: next(ticks),
        tailscale_exe="ts",
        **kw,
    )


def test_collector_state_end_to_end_without_waking_the_model(env):
    host = FakeHost()
    st = collector(env, host).state()
    assert not any(u.endswith("/slots") for u in host.urls)
    sv = st["server"]
    assert sv["state"] == "idle" and sv["model"] == "qwen3.8-27b" and sv["pid"] == 4242
    assert [r["task"] for r in sv["requests"]] == [315, 3]  # newest first
    assert sv["metrics"]["deferred"] == 2 and sv["metrics"]["gen_tokens"] == 4242
    assert sv["footprint_bytes"] == int(6.5 * 2**30) and sv["rss_bytes"] == 21000000 * 1024
    assert sv["started"] == STARTED and sv["requests"][0]["started"] == pytest.approx(STARTED + 107 * 60 + 26.144753)
    # last activity: request 315 ended at 109:06.006 on the log clock; now is 110:00 -> 546 s of 600 left
    assert sv["sleeps_in_s"] == pytest.approx(600 - (110 * 60 - (109 * 60 + 6.006489)))
    assert st["system"]["cpu_pct"] == pytest.approx(40.0) and st["system"]["gpu"]["device_pct"] == 20
    assert st["system"]["memory"]["pressure"] == "warning" and len(st["history"]) == 1
    k = st["stack"]
    assert k["vm"]["available_bytes"] == 640828 * 1024 and k["tor"]["bootstrap_pct"] == 100
    assert k["containers"][0]["name"] == "lmtune-tor" and k["searxng"][0]["name"] == "brave"
    t = st["tailnet"]
    assert t["self"] == "mac.tail1234.ts.net" and [p["name"] for p in t["peers"]] == ["phone", "pi"]
    assert st["links"]["tailnet"]["llama"] == "https://mac.tail1234.ts.net"
    assert st["links"]["local"]["analytics"] == "http://127.0.0.1:3000/admin/analytics"
    assert st["errors"] == {} and st["host"] == "mac.tail1234.ts.net"
    json.dumps(st)  # the payload is plain JSON


def test_collector_sleeping_and_no_metrics(env):
    host = FakeHost(sleeping=True, metrics=False)
    sv = collector(env, host).state()["server"]
    assert sv["state"] == "asleep" and sv["metrics"] is None and sv["sleeps_in_s"] is None
    assert not any(u.endswith("/slots") for u in host.urls)


def test_collector_reports_a_broken_source_and_keeps_going(env):
    host = FakeHost()
    real = host.__call__

    def broken(cmd):
        if cmd[0] == "pm" and cmd[1] == "ps":
            return Result(125, "", "Cannot connect to Podman")
        return real(cmd)

    c = collector(env, host)
    c.run = broken
    st = c.state()
    assert "Cannot connect to Podman" in st["errors"]["containers"] and st["stack"]["containers"] is None
    assert st["server"]["state"] == "idle"


def test_collector_stopped_server(env):
    host = FakeHost(loaded=False)
    sv = collector(env, host).state()["server"]
    assert sv["state"] == "stopped" and sv["pid"] is None and sv["managed"] is False


def test_collector_server_started_by_hand(env):
    """No launchd agent but llama-server answers: report it, found by its port, and don't map the
    log (a previous run's) onto it."""
    host = FakeHost(loaded=False, serving=True)
    st = collector(env, host).state()
    sv = st["server"]
    assert sv["state"] == "idle" and sv["managed"] is False and sv["pid"] == 68263
    assert sv["started"] is None and sv["requests"][0]["started"] is None and sv["requests"][0]["task"] == 315
    assert sv["active"] == [] and sv["sleeps_in_s"] is None and sv["footprint_bytes"] == int(6.5 * 2**30)
    assert any(c[:2] == ["footprint", "-p"] and c[2] == "68263" for c in host.calls)


def test_log_exit_ends_the_run():
    st = feed(
        LOG.splitlines()[0] + "\n0.01.000.000 I slot launch_slot_: id  0 | task 1 | processing task\n"
        "0.02.000.000 I srv    operator(): operator(): cleaning up before exit...\n"
    )
    assert st.lifecycle == "exited" and not st.active


# -- serve.Server.status never wakes a sleeping model ----------------------------------------------


def test_status_skips_slots_while_sleeping_and_prefers_metrics(env):
    host = FakeHost(sleeping=True)
    assert server(env, host).status().busy is None
    assert not any(u.endswith("/slots") for u in host.urls)
    host = FakeHost(sleeping=False, metrics=True)
    assert server(env, host).status().busy is False  # from requests_processing 0
    assert not any(u.endswith("/slots") for u in host.urls)
    host = FakeHost(sleeping=False, metrics=False)
    assert server(env, host).status().busy is True  # awake, no metrics: /slots
    assert host.urls[-1].endswith("/slots")


def test_parse_prometheus():
    m = serve.parse_prometheus(
        "# HELP llamacpp:requests_processing x\n# TYPE llamacpp:requests_processing gauge\n"
        "llamacpp:requests_processing 1\nllamacpp:prompt_seconds_total 12.5\n"
        'llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position="0"} 7\n'
    )
    assert m == {"requests_processing": 1.0, "prompt_seconds_total": 12.5}


def test_metrics_flag_exists_in_the_measured_build():
    known, _ = flags.parse_help((FIXTURES / "llama-server-help-b11669.txt").read_text())
    assert "--metrics" in known


# -- HTTP ----------------------------------------------------------------------------------------


class StubCollector:
    def __init__(self):
        self.n = 0

    def state(self):
        self.n += 1
        return {"generated": 1.0, "server": {"state": "idle"}}


@pytest.fixture
def served():
    stub = StubCollector()
    httpd = dash.DashServer(("127.0.0.1", 0), stub, frozenset())  # type: ignore[arg-type]
    port = httpd.server_address[1]
    httpd.hosts = dash.allowed_hosts(dash.DashConfig(port=port), "mac.tail1234.ts.net")
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield port, stub
    httpd.shutdown()
    httpd.server_close()


def get(port: int, path: str, host: str | None = None, method: str = "GET") -> http.client.HTTPResponse:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request(method, path, headers={"Host": host or f"127.0.0.1:{port}"})
    r = conn.getresponse()
    r.body = r.read()  # type: ignore[attr-defined]
    conn.close()
    return r


def test_http_page_has_a_hash_only_csp(served):
    port, _ = served
    r = get(port, "/")
    assert r.status == 200 and r.getheader("Content-Type").startswith("text/html")
    csp = r.getheader("Content-Security-Policy")
    html = r.body.decode()
    for tag in ("script", "style"):
        body = re.search(rf"<{tag}>(.*?)</{tag}>", html, re.S).group(1)
        digest = base64.b64encode(hashlib.sha256(body.encode()).digest()).decode()
        assert f"'sha256-{digest}'" in csp
    assert "default-src 'none'" in csp and "unsafe-inline" not in csp and "frame-ancestors 'none'" in csp
    assert r.getheader("X-Content-Type-Options") == "nosniff" and r.getheader("Cache-Control") == "no-store"
    assert "innerHTML" not in html  # everything is built with textContent


def test_http_state_and_hosts(served):
    port, stub = served
    r = get(port, "/api/state")
    assert r.status == 200 and json.loads(r.body)["server"]["state"] == "idle" and stub.n == 1
    assert get(port, "/api/state", host="mac.tail1234.ts.net:8444").status == 200  # via tailscale serve
    assert get(port, "/api/state", host=f"localhost:{port}").status == 200
    rebind = get(port, "/api/state", host=f"attacker.example:{port}")
    assert rebind.status == 421 and stub.n == 3  # DNS rebinding: refused before collecting anything
    assert get(port, "/nope").status == 404
    head = get(port, "/", method="HEAD")
    assert head.status == 200 and head.body == b""


def test_allowed_hosts():
    h = dash.allowed_hosts(dash.DashConfig(), "")
    assert h == {"127.0.0.1:8090", "localhost:8090", "[::1]:8090"}
    assert "x.ts.net:8444" in dash.allowed_hosts(dash.DashConfig(), "X.ts.net")


# -- config, login agent, CLI --------------------------------------------------------------------


def test_load_config(tmp_path):
    p = tmp_path / "dash.toml"
    assert dash.load(p) == dash.DashConfig()
    p.write_text("[dash]\nport = 9000\nrefresh_s = 5\n")
    assert dash.load(p).port == 9000
    p.write_text("[dash]\nhttps_port = 443\n")
    with pytest.raises(ValueError, match="443"):
        dash.load(p)
    p.write_text("[dash]\nprot = 1\n")
    with pytest.raises(ValueError, match="unknown"):
        dash.load(p)


def test_install_runs_at_login_with_background_priority(tmp_path):
    paths = Paths(tmp_path / "cfg", tmp_path / "state", tmp_path / "logs", tmp_path / "agents")
    host = FakeHost(loaded=False)
    path = dash.install(paths, host)
    d = plistlib.loads(path.read_bytes())
    assert d["Label"] == dash.DASH_LABEL and d["ProgramArguments"][1:] == ["dash"]
    assert d["ProcessType"] == "Background" and d["KeepAlive"] == {"SuccessfulExit": False}
    assert any(c[:2] == ["launchctl", "bootstrap"] for c in host.calls)
    dash.uninstall(paths, host)
    assert not path.exists()
    assert plistlib.loads(launchd.plist("x", ["y"], Path("/l")))["ProcessType"] == "Interactive"


def test_page_file_ships_with_the_package():
    html, csp = dash.page()
    assert b"api/state" in html and "script-src 'sha256-" in csp
    assert dash.PAGE_FILE.is_relative_to(Path(dash.__file__).resolve().parent)


@pytest.fixture
def cli_env(env, monkeypatch):
    paths, cfg = env
    paths.config_dir.mkdir(parents=True, exist_ok=True)
    paths.serve_config.write_text(config.to_toml(cfg))
    host = FakeHost()
    monkeypatch.setattr(cli, "_paths", lambda: paths)
    monkeypatch.setattr(cli, "_server", lambda p: server(env, host))
    return paths, host


def test_cli_expose_dash_uses_its_own_port(cli_env, capsys):
    _, host = cli_env
    assert cli.main(["expose", "on", "dash"]) == 0
    assert any(c[-3:] == ["--bg", "--https=8444", "8090"] for c in host.calls)
    assert "no login" in capsys.readouterr().out
    assert cli.main(["expose", "off", "dash"]) == 0
    assert any(c[-2:] == ["--https=8444", "off"] for c in host.calls)


def test_cli_dash_snapshot(cli_env, capsys):
    _, host = cli_env
    assert cli.main(["dash", "snapshot"]) == 0
    st = json.loads(capsys.readouterr().out)
    assert st["server"]["state"] == "idle" and not any(u.endswith("/slots") for u in host.urls)


def test_cli_parser_knows_dash():
    p = cli.build_parser()
    for argv in (["dash"], ["dash", "install"], ["dash", "snapshot"], ["expose", "on", "dash"]):
        assert p.parse_args(argv).fn


def test_tailscale_app_cli_is_forced_into_cli_mode():
    app = str(tailscale.APP_CLI)
    assert tailscale.argv(app, "status", "--json") == ["/usr/bin/env", "TAILSCALE_BE_CLI=1", app, "status", "--json"]
    assert tailscale.argv("/opt/homebrew/bin/tailscale", "status") == ["/opt/homebrew/bin/tailscale", "status"]


def test_collector_names_a_tailscale_failure_and_keeps_the_listeners(env):
    host = FakeHost()
    real = host.__call__

    def gui_instead_of_cli(cmd):
        if cmd[0] == "ts":
            return Result(0, "The Tailscale GUI failed to start: (Tailscale.CLIError error 3.)\n", "")
        return real(cmd)

    c = collector(env, host)
    c.run = gui_instead_of_cli
    st = c.state()
    assert "GUI failed to start" in st["errors"]["tailnet"]
    assert st["tailnet"]["listeners"][0]["addr"] == "127.0.0.1:8080"
