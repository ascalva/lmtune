"""`lmtune dash`: a live, read-only dashboard of this machine's AI stack, for the Mac and the phone.

One stdlib HTTP server on loopback serves one self-contained page that polls `/api/state`. Nothing
is collected while no page is open: each source has its own refresh interval and is refreshed only
when a request finds it stale (slow sources in the background, so a poll never waits on podman).
Remote viewing is `lmtune expose on dash`: tailnet-only HTTPS through `tailscale serve`.

It never wakes the model. llama-server's `/slots` waits for a sleeping server to reload its weights
(and, like every task but METRICS, restarts the idle timer), so model state comes from `/health`,
`/props` and `/metrics`, which bypass sleep, and from the server log, which also gives each
request's prompt and decode speed and MTP draft acceptance. The page shows no prompt, response or
search text: SearXNG and Tor log lines are classified and counted, never displayed.
"""

from __future__ import annotations

import base64
import ctypes
import functools
import hashlib
import json
import os
import re
import shutil
import statistics
import threading
import time
import tomllib
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, fields
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import stack, tailscale
from ._types import JSON
from .launchd import Launchd, plist
from .paths import Paths
from .proc import Runner, run
from .serve import Server, parse_prometheus

DASH_LABEL = "com.lmtune.dash"
MAX_REQUESTS = 40  # finished requests kept for the page
HISTORY = 120  # system samples kept for the sparklines (~6 min at the default refresh)
SUMMARY_N = 10  # requests the speed summary covers
MIN_SUMMARY_TOKENS = 16  # shorter generations are too noisy to count in the decode median
LOOPBACK_PREFIXES = ("127.0.0.1:", "[::1]:", "localhost:")
WAKE_MAX_S = 120  # a wake older than this has finished (measured reloads: 4-10 s)


@dataclass(frozen=True)
class DashConfig:
    port: int = 8090  # host loopback
    https_port: int = 8444  # tailnet (`lmtune expose on dash`)
    refresh_s: float = 3.0  # how often an open page polls
    tailnet_host: str = ""  # accepted Host for tailnet requests; default: the stack's tailnet_host

    def problems(self) -> list[str]:
        out = []
        if not 1024 <= self.port <= 65535:
            out.append(f"port {self.port} must be 1024-65535")
        if not 1 <= self.https_port <= 65535 or self.https_port == 443:
            out.append(f"https_port {self.https_port} must be 1-65535 and not 443 (the model server's)")
        if not 1 <= self.refresh_s <= 60:
            out.append("refresh_s must be 1-60")
        return out


def load(path: Path) -> DashConfig:
    raw = tomllib.loads(path.read_text()).get("dash", {}) if path.exists() else {}
    unknown = set(raw) - {f.name for f in fields(DashConfig)}
    if unknown:
        raise ValueError(f"unknown [dash] keys: {', '.join(sorted(unknown))}")
    cfg = DashConfig(**raw)
    if errs := cfg.problems():
        raise ValueError("; ".join(errs))
    return cfg


# -- server log ---------------------------------------------------------------------------------

# llama.cpp log lines: "<minutes>.<ss>.<ms>.<us> <level> <message>", the clock starting at process start
_LINE = re.compile(r"^(\d+)\.(\d{2})\.(\d{3})\.(\d{3}) [A-Z] (.*)$")
_TASK = re.compile(r"\bid +\d+ \| task (\d+) \| (.*)$")
_PREFILL = re.compile(r"prompt processing, n_tokens = +(\d+), progress = ([0-9.]+), .*?([0-9.]+) tokens per second")
_DECODE = re.compile(r"n_gen = +(\d+), tg = +[0-9.]+ t/s, tg_3s = +([0-9.]+) t/s")
_PROMPT_DONE = re.compile(r"^prompt eval time = +([0-9.]+) ms / +(\d+) tokens .*?([0-9.]+) tokens per second")
_EVAL_DONE = re.compile(r"^ *eval time = +([0-9.]+) ms / +(\d+) tokens .*?([0-9.]+) tokens per second")
_DRAFT = re.compile(r"draft acceptance = +[0-9.]+ \( *(\d+) accepted / +(\d+) generated\)")
_RELEASE = re.compile(r"stop processing: n_tokens = +(\d+)")


@dataclass
class Request:
    """One completion as the server log tells it. Times are seconds on the log's clock."""

    task: int
    started: float
    phase: str = "prefill"  # prefill | decode | done
    progress: float | None = None  # prefill fraction while prefilling
    live_tps: float | None = None  # current prefill or decode speed (decode: last 3 s)
    prompt_tokens: int | None = None  # evaluated, i.e. not reused from the cache
    prompt_tps: float | None = None
    prompt_s: float | None = None
    gen_tokens: int | None = None
    gen_tps: float | None = None
    gen_s: float | None = None
    draft_accepted: int | None = None
    draft_generated: int | None = None
    ctx_tokens: int | None = None  # context at the end: cached + new prompt + generated
    ended: float | None = None

    @property
    def acceptance(self) -> float | None:
        if not self.draft_generated or self.draft_accepted is None:
            return None
        return self.draft_accepted / self.draft_generated

    def to_json(self, t0: float | None, now_clock: float | None = None) -> JSON:
        def wall(t: float | None) -> float | None:
            return t0 + t if t0 is not None and t is not None else None

        end = self.ended if self.ended is not None else now_clock
        return {
            "task": self.task,
            "phase": self.phase,
            "started": wall(self.started),
            "ended": wall(self.ended),
            "elapsed_s": round(end - self.started, 1) if end is not None else None,
            "progress": self.progress,
            "live_tps": self.live_tps,
            "prompt_tokens": self.prompt_tokens,
            "prompt_tps": self.prompt_tps,
            "prompt_s": self.prompt_s,
            "gen_tokens": self.gen_tokens,
            "gen_tps": self.gen_tps,
            "gen_s": self.gen_s,
            "draft_accepted": self.draft_accepted,
            "draft_generated": self.draft_generated,
            "acceptance": self.acceptance,
            "ctx_tokens": self.ctx_tokens,
        }


@dataclass
class LogState:
    done: deque[Request] = field(default_factory=lambda: deque(maxlen=MAX_REQUESTS))
    active: dict[int, Request] = field(default_factory=dict)
    lifecycle: str | None = None  # last of: started | loaded | sleep | wake
    lifecycle_at: float | None = None
    last_activity: float | None = None  # last wake, start or request end: what the idle timer counts from
    clock: float | None = None  # newest timestamp seen

    def feed(self, line: str) -> None:
        m = _LINE.match(line)
        if not m:
            return
        t = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 1e3 + int(m.group(4)) / 1e6
        msg = m.group(5)
        if "llama_server: initializing" in msg:  # a new server process: its clock restarts at 0
            vars(self).update(vars(LogState()))
            self._event("started", t)
            self.last_activity = t
            return
        self.clock = t
        tm = _TASK.search(msg)
        if tm:
            self._task(int(tm.group(1)), tm.group(2), t)
        elif "server is entering sleeping state" in msg:
            self._event("sleep", t)
            self.active.clear()  # nothing survives the unload
        elif "server is exiting sleeping state" in msg:
            self._event("wake", t)
            self.last_activity = t
        elif "llama_server: model loaded" in msg:
            self._event("loaded", t)
            self.last_activity = t
        elif "cleaning up before exit" in msg:  # stopped (e.g. `lmtune serve stop`)
            self._event("exited", t)
            self.active.clear()

    def _event(self, name: str, t: float) -> None:
        self.lifecycle, self.lifecycle_at, self.clock = name, t, t

    def _task(self, task: int, rest: str, t: float) -> None:
        if "processing task" in rest:
            self.active[task] = Request(task, t)
            return
        r = self.active.get(task)
        if r is None:
            return
        if m := _PREFILL.search(rest):
            r.phase, r.progress, r.live_tps = "prefill", float(m.group(2)), float(m.group(3))
        elif m := _DECODE.search(rest):
            r.phase, r.gen_tokens, r.live_tps, r.progress = "decode", int(m.group(1)), float(m.group(2)), None
        elif m := _PROMPT_DONE.search(rest):
            r.prompt_s, r.prompt_tokens, r.prompt_tps = float(m.group(1)) / 1e3, int(m.group(2)), float(m.group(3))
        elif m := _EVAL_DONE.search(rest):
            r.gen_s, r.gen_tokens, r.gen_tps = float(m.group(1)) / 1e3, int(m.group(2)), float(m.group(3))
        elif m := _DRAFT.search(rest):
            r.draft_accepted, r.draft_generated = int(m.group(1)), int(m.group(2))
        elif m := _RELEASE.search(rest):
            r.ctx_tokens, r.ended, r.phase, r.progress, r.live_tps = int(m.group(1)), t, "done", None, None
            self.done.append(self.active.pop(task))
            self.last_activity = t


class LogFollower:
    """Parses only what was appended since the last call. A new or truncated file (rotation) starts
    over from its last `tail_bytes`; anything before the newest server start is discarded anyway."""

    def __init__(self, path: Path, tail_bytes: int = 4 * 2**20):
        self.path, self.tail_bytes = path, tail_bytes
        self.state = LogState()
        self._ino: int | None = None
        self._pos = 0
        self._partial = b""

    def update(self) -> LogState:
        try:
            st = self.path.stat()
        except OSError:
            return self.state
        skip_partial = False
        if st.st_ino != self._ino or st.st_size < self._pos or st.st_size - self._pos > self.tail_bytes:
            self.state, self._ino, self._partial = LogState(), st.st_ino, b""
            self._pos = max(0, st.st_size - self.tail_bytes)
            skip_partial = self._pos > 0
        if st.st_size == self._pos:
            return self.state
        with self.path.open("rb") as f:
            f.seek(self._pos)
            data = self._partial + f.read(st.st_size - self._pos)
        self._pos = st.st_size
        lines = data.split(b"\n")
        self._partial = lines.pop()  # unterminated: finish it next time
        if skip_partial and lines:
            lines.pop(0)  # began mid-line
        for raw in lines:
            self.state.feed(raw.decode("utf-8", "replace"))
        return self.state


def summarize(reqs: list[Request]) -> JSON:
    """Medians over the last SUMMARY_N finished requests; acceptance pooled over their drafts."""
    recent = [r for r in reqs if r.phase == "done"][-SUMMARY_N:]
    gen = [r.gen_tps for r in recent if r.gen_tps and (r.gen_tokens or 0) >= MIN_SUMMARY_TOKENS]
    prompt = [r.prompt_tps for r in recent if r.prompt_tps and (r.prompt_tokens or 0) >= MIN_SUMMARY_TOKENS]
    acc = sum(r.draft_accepted or 0 for r in recent)
    drafted = sum(r.draft_generated or 0 for r in recent)
    return {
        "requests": len(recent),
        "gen_tps": statistics.median(gen) if gen else None,
        "prompt_tps": statistics.median(prompt) if prompt else None,
        "acceptance": acc / drafted if drafted else None,
    }


def model_state(
    agent_loaded: bool,
    health: int,
    sleeping: bool | None,
    log: LogState,
    processing: float | None,
    *,
    now_clock: float | None = None,
) -> str:
    """stopped | down | loading | error | waking | asleep | busy | idle, from what answers on the
    port first: a server started by hand (no launchd agent) is still a server. While a sleeping
    server reloads, /props still says sleeping; a wake as the log's last lifecycle event tells waking
    from asleep, if it is recent (a reload takes seconds), so a missed log line can't leave it
    "waking". `log` must be this server's (pass an empty one otherwise)."""
    if health == 0:
        return "down" if agent_loaded else "stopped"
    if health == 503:
        return "loading"
    if health != 200:
        return "error"
    if sleeping:
        recent = now_clock is None or log.lifecycle_at is None or now_clock - log.lifecycle_at < WAKE_MAX_S
        return "waking" if log.lifecycle == "wake" and recent else "asleep"
    busy = processing > 0 if processing is not None else bool(log.active)
    return "busy" if busy else "idle"


# -- parsers for machine and stack sources ------------------------------------------------------


def parse_sysctl(text: str) -> JSON:
    """`sysctl kern.memorystatus_level kern.memorystatus_vm_pressure_level vm.swapusage hw.memsize`"""
    vals = dict(line.split(": ", 1) for line in text.splitlines() if ": " in line)

    def num(key: str) -> int | None:
        v = vals.get(key, "").strip()
        return int(v) if v.isdigit() else None

    swap = vals.get("vm.swapusage", "")
    total = re.search(r"total = ([0-9.]+)M", swap)
    used = re.search(r"used = ([0-9.]+)M", swap)
    level = num("kern.memorystatus_vm_pressure_level")
    return {
        "ram_bytes": num("hw.memsize"),
        "free_pct": num("kern.memorystatus_level"),  # what `memory_pressure` reports as free
        "pressure": {1: "normal", 2: "warning", 4: "critical"}.get(level or 0, "unknown"),
        "swap_total_bytes": int(float(total.group(1)) * 2**20) if total else None,
        "swap_used_bytes": int(float(used.group(1)) * 2**20) if used else None,
    }


def parse_vm_stat(text: str) -> JSON:
    """Activity Monitor's split: app (anonymous minus purgeable), wired, compressed, cached files."""
    m = re.search(r"page size of (\d+) bytes", text)
    page = int(m.group(1)) if m else 16384
    pages = {k.strip('"'): int(v) for k, v in re.findall(r"^([^:\n]+):\s+(\d+)\.?$", text, re.M)}

    def b(key: str) -> int:
        return pages.get(key, 0) * page

    purgeable = b("Pages purgeable")
    return {
        "app": max(0, b("Anonymous pages") - purgeable),
        "wired": b("Pages wired down"),
        "compressed": b("Pages occupied by compressor"),
        "cached": b("File-backed pages") + purgeable,
        "free": b("Pages free") + b("Pages speculative"),
    }


def parse_gpu(text: str) -> JSON:
    """IOAccelerator PerformanceStatistics from `ioreg -r -d 1 -c IOAccelerator` (no sudo)."""

    def num(key: str) -> int | None:
        m = re.search(rf'"{re.escape(key)}"=(\d+)', text)
        return int(m.group(1)) if m else None

    return {
        "device_pct": num("Device Utilization %"),
        "renderer_pct": num("Renderer Utilization %"),
        "in_use_bytes": num("In use system memory"),
    }


def parse_ps(text: str) -> JSON:
    """`ps -o rss=,%cpu=,lstart= -p PID` -> RSS, CPU %, start time (Unix)."""
    parts = text.split()
    if len(parts) < 3 or not parts[0].isdigit():
        return {}
    try:
        started = time.mktime(time.strptime(" ".join(parts[2:7]), "%a %b %d %H:%M:%S %Y"))
    except ValueError:
        started = None
    return {"rss_bytes": int(parts[0]) * 1024, "cpu_pct": float(parts[1]), "started": started}


def parse_footprint(text: str) -> int | None:
    m = re.search(r"Footprint: ([0-9.]+) ([KMG]B)", text)
    return int(float(m.group(1)) * {"KB": 2**10, "MB": 2**20, "GB": 2**30}[m.group(2)]) if m else None


_UNITS = {"b": 1, "kb": 1e3, "mb": 1e6, "gb": 1e9, "tb": 1e12, "kib": 2**10, "mib": 2**20, "gib": 2**30}


def parse_size(s: str) -> int | None:
    """podman's sizes: '688.2MB', '1.611GB', '512kB', '0B'."""
    m = re.fullmatch(r"\s*([0-9.]+)\s*([a-zA-Z]+)\s*", s)
    if not m or m.group(2).lower() not in _UNITS:
        return None
    return int(float(m.group(1)) * _UNITS[m.group(2).lower()])


def parse_containers(ps_json: str, stats_json: str) -> list[JSON]:
    try:
        ps = json.loads(ps_json or "[]")
    except json.JSONDecodeError:
        return []
    try:
        stats = {s.get("name"): s for s in json.loads(stats_json or "[]") if isinstance(s, dict)}
    except json.JSONDecodeError:
        stats = {}
    out = []
    for c in ps if isinstance(ps, list) else []:
        name = (c.get("Names") or ["?"])[0]
        s = stats.get(name, {})
        used, _, limit = str(s.get("mem_usage", "")).partition("/")
        cpu = str(s.get("cpu_percent", "")).rstrip("%")
        out.append(
            {
                "name": name,
                "ours": name.startswith("lmtune-"),
                "state": c.get("State", ""),
                "status": c.get("Status", ""),
                "started": c.get("StartedAt") or None,
                "mem_bytes": parse_size(used) if used else None,
                "mem_limit_bytes": parse_size(limit) if limit else None,
                "cpu_pct": float(cpu) if re.fullmatch(r"[0-9.]+", cpu) else None,
            }
        )
    return sorted(out, key=lambda c: (not c["ours"], c["name"]))


def parse_meminfo(text: str) -> JSON:
    kb = {k: int(v) * 1024 for k, v in re.findall(r"^(\w+):\s+(\d+) kB", text, re.M)}
    return {"total_bytes": kb.get("MemTotal"), "available_bytes": kb.get("MemAvailable")}


def _iso_time(s: str) -> float | None:
    """RFC 3339 (nanoseconds and all, which Safari's Date.parse rejects) -> Unix time."""
    try:
        return datetime.fromisoformat(s).timestamp()
    except ValueError:
        return None


def _log_time(line: str) -> float | None:
    """The timestamp `podman logs -t` puts in front of each line."""
    return _iso_time(line.split(" ", 1)[0])


def parse_tor(log: str, now: float, window_s: float = 3600) -> JSON:
    """Bootstrap progress and how many warnings Tor logged recently (counted, never shown)."""
    warns = [t for line in log.splitlines() if "[warn]" in line or "[err]" in line if (t := _log_time(line))]
    recent = [t for t in warns if now - t <= window_s]
    return {
        "bootstrap_pct": stack.tor_bootstrap(log),
        "warnings_1h": len(recent),
        "last_warning": max(recent) if recent else None,
    }


_ENGINE_LINE = re.compile(r"(?:ERROR|WARNING):searx\.(?:engines|network)\.([^:]+): (.*)$")


def classify_engine_error(msg: str) -> str:
    m = msg.lower()
    for kind, needles in (
        ("timeout", ("timeout", "timed out")),
        ("rate-limited", ("too many requests", "429")),
        ("captcha", ("captcha",)),
        ("blocked", ("access denied", "blocked", "403")),
        ("suspended", ("suspend",)),
        ("connection", ("connectionerror", "disconnected", "connection refused", "connection reset")),
    ):
        if any(n in m for n in needles):
            return kind
    return "error"


def parse_searxng(log: str, engines: tuple[str, ...]) -> list[JSON]:
    """Per configured engine, its errors in the log window by kind. SearXNG logs failures only
    (no access log, metrics off), so no errors means healthy or unused — not proof of success."""
    by_engine: dict[str, dict[str, int]] = {e: {} for e in engines}
    last: dict[str, float] = {}
    for line in log.splitlines():
        m = _ENGINE_LINE.search(line)
        if not m or m.group(1) not in by_engine:
            continue
        kind = classify_engine_error(m.group(2))
        by_engine[m.group(1)][kind] = by_engine[m.group(1)].get(kind, 0) + 1
        if (t := _log_time(line)) is not None:
            last[m.group(1)] = max(t, last.get(m.group(1), 0))
    return [{"name": e, "errors": by_engine[e], "last_error": last.get(e)} for e in engines]


def tailnet_peers(data: JSON) -> list[JSON]:
    """Devices in this tailnet (shared-in nodes such as Mullvad exit nodes are left out)."""
    suffix = str(data.get("MagicDNSSuffix") or "")
    out = []
    for p in (data.get("Peer") or {}).values():
        dns = str(p.get("DNSName") or "").rstrip(".")
        if suffix and not dns.endswith("." + suffix):
            continue
        hs = str(p.get("LastHandshake") or "")
        ips = [ip for ip in p.get("TailscaleIPs") or [] if ":" not in ip]
        out.append(
            {
                "name": dns.split(".")[0] or p.get("HostName", "?"),
                "os": p.get("OS") or "",
                "ip": ips[0] if ips else "",
                "online": bool(p.get("Online")),
                "active": bool(p.get("Active")),
                "direct": bool(p.get("CurAddr")),
                "relay": p.get("Relay") or "",
                "last_handshake": None if hs.startswith("0001-") else _iso_time(hs),
                "rx_bytes": p.get("RxBytes", 0),
                "tx_bytes": p.get("TxBytes", 0),
            }
        )
    return sorted(out, key=lambda p: (not p["active"], not p["online"], p["name"]))


def serve_exposures(data: JSON) -> list[JSON]:
    """Every `tailscale serve` handler, flagging any that Funnel puts on the public internet."""
    funnel = data.get("AllowFunnel") or {}
    out = []
    for hostport, cfg in (data.get("Web") or {}).items():
        host, _, port = str(hostport).rpartition(":")
        base = f"https://{host}" + ("" if port in ("443", "") else f":{port}")
        for path, h in ((cfg or {}).get("Handlers") or {}).items():
            target = (h or {}).get("Proxy") or (h or {}).get("Path") or ("text" if (h or {}).get("Text") else "?")
            out.append(
                {"url": base + ("" if path == "/" else path), "target": target, "funnel": bool(funnel.get(hostport))}
            )
    for port, cfg in (data.get("TCP") or {}).items():
        if (cfg or {}).get("TCPForward"):
            out.append({"url": f"tcp:{port}", "target": cfg["TCPForward"], "funnel": False})
    return sorted(out, key=lambda e: str(e["url"]))


def parse_listeners(lsof_text: str, ports: set[int]) -> list[JSON]:
    out = []
    for line in lsof_text.splitlines()[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        addr = parts[8]
        port = addr.rpartition(":")[2]
        if port.isdigit() and int(port) in ports:
            out.append(
                {"addr": addr, "port": int(port), "process": parts[0], "loopback": addr.startswith(LOOPBACK_PREFIXES)}
            )
    return sorted({(e["port"], e["addr"]): e for e in out}.values(), key=lambda e: (e["port"], e["addr"]))


# -- CPU load (Mach host statistics: no subprocess, no sudo) -------------------------------------

HOST_CPU_LOAD_INFO = 3  # user, system, idle, nice ticks summed over all CPUs


def _cpu_ticks_reader() -> Callable[[], tuple[int, int, int, int] | None]:
    try:
        lib = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
        lib.mach_host_self.restype = ctypes.c_uint32
        lib.host_statistics.argtypes = [
            ctypes.c_uint32,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]
        lib.host_statistics.restype = ctypes.c_int
        host = lib.mach_host_self()
    except OSError, AttributeError:
        return lambda: None

    def read() -> tuple[int, int, int, int] | None:
        arr = (ctypes.c_uint32 * 4)()
        n = ctypes.c_uint32(4)
        if lib.host_statistics(host, HOST_CPU_LOAD_INFO, arr, ctypes.byref(n)) != 0:
            return None
        return arr[0], arr[1], arr[2], arr[3]

    return read


def cpu_busy_pct(prev: tuple[int, ...], cur: tuple[int, ...]) -> float | None:
    d = [(b - a) % 2**32 for a, b in zip(prev, cur, strict=True)]  # 32-bit tick counters wrap
    total = sum(d)
    return 100 * (total - d[2]) / total if total else None


# -- collection ---------------------------------------------------------------------------------

_UNSET: Any = object()


class Source:
    """One data source with its own refresh interval. Refreshes happen when a request finds the
    value stale; `background` sources then refresh on a thread and the request gets the previous
    value, so a slow podman or tailscale call never delays the page (the first call still waits)."""

    def __init__(
        self,
        name: str,
        ttl: float,
        fn: Callable[[], Any],
        *,
        background: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.name, self.ttl, self.fn, self.background, self.clock = name, ttl, fn, background, clock
        self.value: Any = _UNSET
        self.error: str | None = None
        self.at = -float("inf")
        self._lock = threading.Lock()

    def refresh(self) -> None:
        with self._lock:
            if self.clock() - self.at < self.ttl:
                return  # another thread just did it
            try:
                self.value, self.error = self.fn(), None
            except Exception as e:  # noqa: BLE001 - one broken source must not take the page down
                self.error = f"{type(e).__name__}: {e}"[:300]
                if self.value is _UNSET:
                    self.value = None
            self.at = self.clock()

    def get(self) -> Any:
        if self.clock() - self.at >= self.ttl:
            if self.value is _UNSET or not self.background:
                self.refresh()
            elif not self._lock.locked():
                threading.Thread(target=self.refresh, name=f"dash-{self.name}", daemon=True).start()
        return None if self.value is _UNSET else self.value


class Collector:
    def __init__(
        self,
        srv: Server,
        sc: stack.StackConfig,
        dc: DashConfig,
        paths: Paths,
        *,
        runner: Runner = run,
        clock: Callable[[], float] = time.time,
        cpu_ticks: Callable[[], tuple[int, int, int, int] | None] | None = None,
        tailscale_exe: str | None = None,
    ):
        self.srv, self.sc, self.dc, self.paths, self.run, self.clock = srv, sc, dc, paths, runner, clock
        self.log = LogFollower(paths.server_log)
        self._cpu_ticks = cpu_ticks or _cpu_ticks_reader()
        self._cpu_prev = self._cpu_ticks()
        self.history: deque[JSON] = deque(maxlen=HISTORY)
        self.tailscale = tailscale_exe if tailscale_exe is not None else srv.tailscale.exe
        self._pid_start: tuple[int | None, float | None] = (None, None)
        self.server_pid: int | None = None
        self.sources = {
            s.name: s
            for s in (
                Source("server", 2.0, self._server),
                Source("system", 2.0, self._system),
                Source("footprint", 10.0, self._footprint, background=True),
                Source("containers", 10.0, self._containers, background=True),
                Source("tailnet", 15.0, self._tailnet, background=True),
                Source("listeners", 10.0, self._listeners, background=True),
                Source("vm", 30.0, self._vm, background=True),
                Source("tor", 30.0, self._tor, background=True),
                Source("searxng", 30.0, self._searxng, background=True),
            )
        }

    # each source returns plain JSON-able data

    def _server(self) -> JSON:
        agent = self.srv.launchd.state()
        code, _ = self.srv.get("/health", timeout=2.0)
        props: JSON = {}
        metrics: dict[str, float] | None = None
        if code == 200:
            pc, body = self.srv.get("/props", timeout=2.0)
            try:
                props = json.loads(body) if pc == 200 else {}
            except json.JSONDecodeError:
                props = {}
            mc, body = self.srv.get("/metrics", timeout=2.0)  # 501 unless llama-server runs with --metrics
            metrics = parse_prometheus(body.decode("utf-8", "replace")) if mc == 200 else None
        # the launchd agent's server writes server.log; one started by hand is found by its port
        pid = agent.pid or (self._listener_pid() if code else None)
        self.server_pid = pid
        proc: JSON = {}
        if pid:
            r = self.run(["ps", "-o", "rss=,%cpu=,lstart=", "-p", str(pid)])
            proc = parse_ps(r.out) if r.ok else {}
            if self._pid_start[0] != pid and proc.get("started"):
                self._pid_start = (pid, proc["started"])
        log = self.log.update()
        # the log's clock starts with the agent's process; another process's times can't be mapped
        t0 = self._pid_start[1] if agent.pid and self._pid_start[0] == agent.pid else None
        sleeping = props.get("is_sleeping") if isinstance(props.get("is_sleeping"), bool) else None
        processing = metrics.get("requests_processing") if metrics else None
        now = self.clock()
        now_clock = now - t0 if t0 is not None else log.clock
        ours = log if agent.loaded else LogState()
        state = model_state(agent.loaded, code, sleeping, ours, processing, now_clock=now_clock)
        sleeps_in = None
        if state == "idle" and self.srv.cfg.sleep_idle_s > 0 and t0 is not None and ours.last_activity is not None:
            sleeps_in = max(0.0, t0 + ours.last_activity + self.srv.cfg.sleep_idle_s - now)
        done = list(log.done)
        return {
            "state": state,
            "health": code,
            "managed": agent.loaded,  # False: not run by `lmtune serve` (the log is a previous run's)
            "pid": pid,
            "model": props.get("model_alias"),
            "build": props.get("build_info"),
            "sleep_idle_s": self.srv.cfg.sleep_idle_s,
            "sleeps_in_s": sleeps_in,
            "lifecycle": ours.lifecycle,
            "lifecycle_at": t0 + ours.lifecycle_at if t0 is not None and ours.lifecycle_at is not None else None,
            "rss_bytes": proc.get("rss_bytes"),
            "cpu_pct": proc.get("cpu_pct"),
            "started": t0,
            "active": [r.to_json(t0, now_clock) for r in ours.active.values()],
            "requests": [r.to_json(t0) for r in reversed(done)],
            "summary": summarize(done),
            "metrics": _metrics_view(metrics),
        }

    def _system(self) -> JSON:
        sysctl = self.run(
            ["sysctl", "kern.memorystatus_level", "kern.memorystatus_vm_pressure_level", "vm.swapusage", "hw.memsize"]
        )
        mem = parse_sysctl(sysctl.out) if sysctl.ok else {}
        vm = self.run(["vm_stat"])
        mem["breakdown"] = parse_vm_stat(vm.out) if vm.ok else None
        ticks = self._cpu_ticks()
        cpu = cpu_busy_pct(self._cpu_prev, ticks) if ticks and self._cpu_prev else None
        if ticks:
            self._cpu_prev = ticks
        gpu_r = self.run(["ioreg", "-r", "-d", "1", "-c", "IOAccelerator"])
        gpu = parse_gpu(gpu_r.out) if gpu_r.ok else {}
        load1 = os.getloadavg()[0]
        sample = {
            "t": self.clock(),
            "cpu_pct": cpu,
            "gpu_pct": gpu.get("device_pct"),
            "swap_used_bytes": mem.get("swap_used_bytes"),
            "free_pct": mem.get("free_pct"),
        }
        self.history.append(sample)
        return {"memory": mem, "cpu_pct": cpu, "load1": load1, "cpus": os.cpu_count(), "gpu": gpu}

    def _listener_pid(self) -> int | None:
        r = self.run(["lsof", "-nP", "-t", f"-iTCP:{self.srv.cfg.port}", "-sTCP:LISTEN"])
        pids = [int(x) for x in r.out.split() if x.isdigit()] if r.ok else []
        return pids[0] if pids else None

    def _footprint(self) -> JSON:
        pid = self.server_pid
        if not pid:
            return {"footprint_bytes": None}
        r = self.run(["footprint", "-p", str(pid)])
        return {"footprint_bytes": parse_footprint(r.out) if r.ok else None}

    def _podman(self, *args: str) -> str:
        r = self.run([self.sc.podman, *args])
        if not r.ok:
            raise RuntimeError((r.err or r.out).strip()[:200] or f"podman {args[0]} failed ({r.code})")
        return r.out

    def _containers(self) -> list[JSON]:
        ps = self._podman("ps", "-a", "--format", "json")
        stats = self.run([self.sc.podman, "stats", "--no-stream", "--format", "json"])
        return parse_containers(ps, stats.out if stats.ok else "[]")

    def _vm(self) -> JSON:
        return parse_meminfo(self._podman("machine", "ssh", "grep -E '^(MemTotal|MemAvailable):' /proc/meminfo"))

    def _tor(self) -> JSON:
        r = self.run([self.sc.podman, "logs", "-t", "--tail", "5000", stack.GATEWAY])
        if not r.ok:
            return {"bootstrap_pct": None, "warnings_1h": 0, "last_warning": None, "running": False}
        return {**parse_tor(r.out + r.err, self.clock()), "running": True}

    def _searxng(self) -> list[JSON]:
        r = self.run([self.sc.podman, "logs", "-t", "--since", "30m", stack.SEARXNG])
        return parse_searxng(r.out + r.err, stack.SEARCH_ENGINES) if r.ok else []

    def _tailscale_json(self, *args: str) -> JSON:
        r = self.run(tailscale.argv(str(self.tailscale), *args))
        try:
            data = json.loads(r.out) if r.ok else None
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data
        # e.g. the app's CLI printing why it couldn't run
        raise RuntimeError(f"tailscale {' '.join(args)}: {(r.err or r.out).strip()[:200] or f'exit {r.code}'}")

    def _tailnet(self) -> JSON:
        if not self.tailscale:
            raise RuntimeError("tailscale not installed")
        data = self._tailscale_json("status", "--json")
        return {
            "self": str((data.get("Self") or {}).get("DNSName", "")).rstrip("."),
            "backend": data.get("BackendState"),
            "peers": tailnet_peers(data),
            "exposures": serve_exposures(self._tailscale_json("serve", "status", "--json")),
        }

    def _listeners(self) -> list[JSON]:
        ports = {self.srv.cfg.port, self.sc.webui_port, self.dc.port}
        if extra := getattr(self.sc, "searxng_port", 0):
            ports.add(int(extra))
        r = self.run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"])
        return parse_listeners(r.out, ports) if r.ok else []

    # the page's payload

    def links(self, exposures: list[JSON]) -> JSON:
        def exposed(port: int) -> str | None:
            want = f":{port}"
            return next((e["url"] for e in exposures if str(e["target"]).rstrip("/").endswith(want)), None)

        webui_local = f"http://127.0.0.1:{self.sc.webui_port}"
        webui_tail = exposed(self.sc.webui_port)
        return {
            "local": {
                "webui": webui_local,
                "analytics": webui_local + "/admin/analytics",
                "llama": self.srv.cfg.url,
            },
            "tailnet": {
                "webui": webui_tail,
                "analytics": webui_tail + "/admin/analytics" if webui_tail else None,
                "llama": exposed(self.srv.cfg.port),
                "dash": exposed(self.dc.port),
            },
        }

    def state(self) -> JSON:
        vals = {name: s.get() for name, s in self.sources.items()}
        server = vals["server"] or {}
        if vals["footprint"]:
            server = {**server, **vals["footprint"]}
        tailnet = {**(vals["tailnet"] or {}), "listeners": vals["listeners"] or []}
        return {
            "generated": self.clock(),
            "host": tailnet.get("self") or os.uname().nodename,
            "refresh_s": self.dc.refresh_s,
            "server": server,
            "system": vals["system"],
            "history": list(self.history),
            "stack": {
                "containers": vals["containers"],
                "vm": vals["vm"],
                "tor": vals["tor"],
                "searxng": vals["searxng"],
            },
            "tailnet": tailnet,
            "links": self.links(tailnet.get("exposures") or []),
            "errors": {name: s.error for name, s in self.sources.items() if s.error},
        }


def _metrics_view(m: dict[str, float] | None) -> JSON | None:
    if m is None:
        return None
    drafted = m.get("spec_decode_num_draft_tokens_total") or 0
    return {
        "processing": m.get("requests_processing"),
        "deferred": m.get("requests_deferred"),
        "prompt_tokens": m.get("prompt_tokens_total"),
        "prompt_tokens_cached": m.get("prompt_tokens_cached_total"),
        "gen_tokens": m.get("tokens_predicted_total"),
        "acceptance": (m.get("spec_decode_num_accepted_tokens_total") or 0) / drafted if drafted else None,
    }


# -- HTTP ---------------------------------------------------------------------------------------


def allowed_hosts(dc: DashConfig, tailnet_host: str) -> frozenset[str]:
    """Host headers we answer: loopback, and the tailnet name `tailscale serve` forwards. Anything
    else is refused, which defeats DNS rebinding (a web page aiming its own name at 127.0.0.1)."""
    hosts = {f"127.0.0.1:{dc.port}", f"localhost:{dc.port}", f"[::1]:{dc.port}"}
    if tailnet_host:
        hosts |= {tailnet_host, f"{tailnet_host}:{dc.https_port}"}
    return frozenset(h.lower() for h in hosts)


class DashServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr: tuple[str, int], collector: Collector, hosts: frozenset[str]):
        self.collector, self.hosts = collector, hosts
        super().__init__(addr, DashHandler)


class DashHandler(BaseHTTPRequestHandler):
    server: DashServer
    server_version = "lmtune-dash"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet: a poll every few seconds would flood the launchd log."""

    def _send(self, code: int, ctype: str, body: bytes, extra: dict[str, str] | None = None) -> None:
        self.send_response(code)
        headers = {
            "Content-Type": ctype,
            "Content-Length": str(len(body)),
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "X-Frame-Options": "DENY",
            **(extra or {}),
        }
        for k, v in headers.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self) -> None:
        if (self.headers.get("Host") or "").lower() not in self.server.hosts:
            self._send(421, "text/plain; charset=utf-8", b"unknown host\n")
            return
        path = self.path.split("?", 1)[0]
        if path == "/":
            html, csp = page()
            self._send(200, "text/html; charset=utf-8", html, {"Content-Security-Policy": csp})
        elif path == "/api/state":
            body = json.dumps(self.server.collector.state(), separators=(",", ":"), default=str).encode()
            self._send(200, "application/json", body)
        else:
            self._send(404, "text/plain; charset=utf-8", b"not found\n")

    do_HEAD = do_GET


def serve_forever(collector: Collector, dc: DashConfig, tailnet_host: str) -> None:
    httpd = DashServer(("127.0.0.1", dc.port), collector, allowed_hosts(dc, tailnet_host))
    try:
        httpd.serve_forever(poll_interval=1.0)
    finally:
        httpd.server_close()


# -- login agent --------------------------------------------------------------------------------


def agent_plist(paths: Paths) -> Path:
    return paths.launch_agents / f"{DASH_LABEL}.plist"


def install(paths: Paths, runner: Runner = run) -> Path:
    """Run `lmtune dash` at login; launchd restarts it after a crash. Background priority: the
    dashboard must never compete with inference."""
    exe = shutil.which("lmtune") or str(Path.home() / ".local/bin/lmtune")
    path = agent_plist(paths)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(plist(DASH_LABEL, [exe, "dash"], paths.log_dir / "dash.log", process_type="Background"))
    ld = Launchd(DASH_LABEL, runner)
    if ld.state().loaded:
        ld.bootout()
    ld.bootstrap(path)
    return path


def uninstall(paths: Paths, runner: Runner = run) -> None:
    Launchd(DASH_LABEL, runner).bootout()
    agent_plist(paths).unlink(missing_ok=True)


# -- the page -----------------------------------------------------------------------------------

PAGE_FILE = Path(__file__).resolve().parent / "data" / "dash.html"


def _sha256(text: str) -> str:
    return "sha256-" + base64.b64encode(hashlib.sha256(text.encode()).digest()).decode()


@functools.cache
def page() -> tuple[bytes, str]:
    """The page and its Content-Security-Policy: its own inline style and script, allowed by hash,
    are all that can run, and it can only talk to this server."""
    html = PAGE_FILE.read_text()
    scripts = " ".join(f"'{_sha256(x)}'" for x in re.findall(r"<script>(.*?)</script>", html, re.S))
    styles = " ".join(f"'{_sha256(x)}'" for x in re.findall(r"<style>(.*?)</style>", html, re.S))
    csp = (
        f"default-src 'none'; script-src {scripts}; style-src {styles}; connect-src 'self'; img-src data:; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )
    return html.encode(), csp
