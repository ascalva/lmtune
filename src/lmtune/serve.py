"""Managed llama-server: launchd lifecycle and status.

`start` runs the server under launchd for this login session; `install` additionally places the
agent in ~/Library/LaunchAgents so it starts at login. Either way launchd restarts it after a
crash, and with `sleep_idle_s` the model unloads when idle and reloads on the next request.
"""

from __future__ import annotations

import json
import re
import shutil
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import keys
from .config import ServeConfig
from .launchd import AgentState, Launchd, plist
from .paths import LABEL, Paths
from .proc import Runner, run
from .tailscale import Exposure, Tailscale

CONTAINER = "lmtune-server"
LOG_ROTATE_BYTES = 20 * 2**20

HttpGet = Callable[[str, dict[str, str], float], tuple[int, bytes]]


def http_get(url: str, headers: dict[str, str], timeout: float) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except OSError:
        return 0, b""


@dataclass
class Status:
    installed: bool
    agent: AgentState
    health: str  # ok | loading | unreachable | error:<code>
    model: str | None = None
    build: str | None = None
    sleeping: bool | None = None
    busy: bool | None = None
    rss_gb: float | None = None
    footprint_gb: float | None = None
    listeners: list[str] = field(default_factory=list)
    exposure: Exposure | None = None
    key_problems: list[str] = field(default_factory=list)
    config_problems: list[str] = field(default_factory=list)

    @property
    def running(self) -> bool:
        return self.agent.pid is not None

    @property
    def loopback_only(self) -> bool:
        return all(a.startswith(("127.0.0.1:", "[::1]:", "localhost:")) for a in self.listeners)

    def to_json(self) -> str:
        d = asdict(self)
        d["running"] = self.running
        d["loopback_only"] = self.loopback_only
        return json.dumps(d, indent=1, default=str)


class Server:
    def __init__(
        self,
        cfg: ServeConfig,
        paths: Paths,
        runner: Runner = run,
        http: HttpGet = http_get,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.cfg, self.paths, self.run, self.http, self.sleep = cfg, paths, runner, http, sleep
        self.launchd = Launchd(LABEL, runner, sleep)
        self.tailscale = Tailscale(runner)

    # -- program -------------------------------------------------------------------------

    def program(self) -> list[str]:
        c = self.cfg
        if c.backend == "native":
            exe = shutil.which("llama-server") or "/opt/homebrew/bin/llama-server"
            return [exe, *c.argv()]
        podman = shutil.which("podman") or "/opt/podman/bin/podman"
        key_mount: list[str] = []
        # 0.0.0.0 is inside the container only; it is published on host loopback below
        mm = f"/models/{c.mmproj.name}" if c.mmproj else None  # problems(): next to the model
        argv = c.argv(model_path=f"/models/{c.model.name}", mmproj_path=mm, host="0.0.0.0", port=8080)  # noqa: S104
        if c.api_key_file is not None:
            key_mount = ["-v", f"{c.api_key_file.parent}:/run/lmtune:ro"]
            i = argv.index("--api-key-file")
            argv[i + 1] = f"/run/lmtune/{c.api_key_file.name}"
        # published on host loopback only; the container's 0.0.0.0 is inside the VM
        return [
            podman,
            "run",
            "--rm",
            "--name",
            CONTAINER,
            "--device",
            "/dev/dri",
            "-v",
            f"{c.model.parent}:/models:ro",
            *key_mount,
            "-p",
            f"127.0.0.1:{c.port}:8080",
            c.image,
            *argv,
        ]

    def _plist_path(self) -> Path:
        return self.paths.installed_plist if self.installed else self.paths.session_plist

    @property
    def installed(self) -> bool:
        return self.paths.installed_plist.exists()

    def _write_plist(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(plist(LABEL, self.program(), self.paths.server_log))

    def _rotate_log(self) -> None:
        log = self.paths.server_log
        if log.exists() and log.stat().st_size > LOG_ROTATE_BYTES:
            log.replace(log.with_suffix(".log.1"))

    # -- lifecycle ------------------------------------------------------------------------

    def start(self, wait_s: float = 300) -> Status:
        self.cfg.check()
        if self.cfg.api_key_file is not None:
            bad = keys.problems(self.cfg.api_key_file)
            if bad:
                raise RuntimeError("API key: " + "; ".join(bad))
        if self.launchd.state().loaded:  # already running (e.g. just installed): still wait for health
            return self.wait_healthy(wait_s)
        self._rotate_log()
        path = self._plist_path()
        self._write_plist(path)
        self.launchd.bootstrap(path)
        if not self.launchd.state().loaded:
            raise RuntimeError("launchd did not load the server agent; see `lmtune serve logs`")
        return self.wait_healthy(wait_s)

    def stop(self, wait_s: float = 30) -> None:
        self.launchd.bootout()
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and self._listeners():
            self.sleep(0.5)

    def restart(self, wait_s: float = 300) -> Status:
        self.stop()
        return self.start(wait_s)

    def install(self) -> Path:
        """Start at login. Replaces a session-only agent if one is running."""
        self.cfg.check()
        was_loaded = self.launchd.state().loaded
        if was_loaded:
            self.launchd.bootout()
        self._write_plist(self.paths.installed_plist)
        self.paths.session_plist.unlink(missing_ok=True)
        self.launchd.bootstrap(self.paths.installed_plist)
        return self.paths.installed_plist

    def uninstall(self) -> None:
        self.launchd.bootout()
        self.paths.installed_plist.unlink(missing_ok=True)

    def wait_healthy(self, wait_s: float) -> Status:
        deadline = time.monotonic() + wait_s
        st = self.status()
        while time.monotonic() < deadline and st.health != "ok":
            if not self.launchd.state().loaded:
                break
            self.sleep(2)
            st = self.status()
        return st

    # -- status ---------------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        f = self.cfg.api_key_file
        if f is not None and f.exists():
            return {"Authorization": f"Bearer {keys.read(f)}"}
        return {}

    def get(self, path: str, timeout: float = 3.0) -> tuple[int, bytes]:
        """GET with the API key. For monitoring, never /slots: it makes a sleeping server reload
        the model and restarts its idle timer (llama.cpp b11669); /health, /props and /metrics
        don't, and /metrics doesn't restart the timer either."""
        return self.http(self.cfg.url + path, self._headers(), timeout)

    def _busy(self) -> bool | None:
        """Only while awake. /metrics (with `--metrics`) leaves the idle timer alone; /slots is the
        fallback and postpones sleep by a full sleep_idle_s."""
        code, body = self.get("/metrics")
        if code == 200:
            processing = parse_prometheus(body.decode("utf-8", "replace")).get("requests_processing")
            if processing is not None:
                return processing > 0
        _, slots = self._get_json("/slots")
        if isinstance(slots, list):
            return any(s.get("is_processing") for s in slots if isinstance(s, dict))
        return None

    def _get_json(self, path: str) -> tuple[int, Any]:
        code, body = self.get(path)
        try:
            return code, json.loads(body) if body else None
        except json.JSONDecodeError:
            return code, None

    def _listeners(self) -> list[str]:
        r = self.run(["lsof", "-nP", f"-iTCP:{self.cfg.port}", "-sTCP:LISTEN"])
        return parse_lsof_listeners(r.out) if r.ok else []

    def _memory(self, pid: int) -> tuple[float | None, float | None]:
        rss = self.run(["ps", "-o", "rss=", "-p", str(pid)])
        fp = self.run(["footprint", "-p", str(pid)])
        rss_gb = int(rss.out.strip()) / 2**20 if rss.ok and rss.out.strip().isdigit() else None
        m = re.search(r"Footprint: ([0-9.]+) ([KMG]B)", fp.out)
        fp_gb = float(m.group(1)) * {"KB": 2**-20, "MB": 2**-10, "GB": 1.0}[m.group(2)] if m else None
        return rss_gb, fp_gb

    def status(self) -> Status:
        agent = self.launchd.state()
        st = Status(installed=self.installed, agent=agent, health="unreachable", config_problems=self.cfg.problems())
        if self.cfg.api_key_file is not None:
            st.key_problems = keys.problems(self.cfg.api_key_file)
        code, _ = self._get_json("/health")
        if code == 200:
            st.health = "ok"
        elif code == 503:
            st.health = "loading"
        elif code:
            st.health = f"error:{code}"
        if code == 200:
            _, props = self._get_json("/props")
            if isinstance(props, dict):
                st.model, st.build, st.sleeping = (
                    props.get("model_alias"),
                    props.get("build_info"),
                    props.get("is_sleeping"),
                )
            if st.sleeping is False:
                st.busy = self._busy()
        if agent.pid:
            st.rss_gb, st.footprint_gb = self._memory(agent.pid)
        st.listeners = self._listeners()
        st.exposure = self.tailscale.status(self.cfg.port)
        return st

    def logs(self, n: int) -> list[str]:
        log = self.paths.server_log
        if not log.exists():
            return []
        with log.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 256 * 1024))
            return f.read().decode("utf-8", "replace").splitlines()[-n:]


def parse_prometheus(text: str) -> dict[str, float]:
    """llama-server's /metrics: 'llamacpp:requests_processing 1' -> {'requests_processing': 1.0}.
    Labeled series (per draft position) are left out."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^llamacpp:(\w+) ([-+0-9.eE]+|nan|inf)$", line.strip())
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def parse_lsof_listeners(text: str) -> list[str]:
    """NAME column of `lsof -iTCP -sTCP:LISTEN`, e.g. ['127.0.0.1:8080']."""
    out = []
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 9:
            out.append(parts[8])
    return sorted(set(out))
