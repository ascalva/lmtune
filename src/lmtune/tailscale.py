"""Tailnet exposure via `tailscale serve`: HTTPS inside the tailnet, server stays on loopback.

Never uses `tailscale funnel` (public internet)."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from .proc import Runner, run

APP_CLI = Path("/Applications/Tailscale.app/Contents/MacOS/Tailscale")


def argv(exe: str, *args: str) -> list[str]:
    """The Tailscale app's bundled CLI, run from a launchd agent, tries to start the GUI instead
    (seen under the `lmtune dash` agent: "The Tailscale GUI failed to start", exit 0, no JSON);
    TAILSCALE_BE_CLI=1 keeps it a CLI. Set per command, not in our own environment."""
    if Path(exe) == APP_CLI:
        return ["/usr/bin/env", "TAILSCALE_BE_CLI=1", exe, *args]
    return [exe, *args]


def binary() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    return str(APP_CLI) if APP_CLI.exists() else None


@dataclass(frozen=True)
class Exposure:
    available: bool
    url: str | None = None  # https://host.tailnet.ts.net when serving our port
    target: str | None = None  # what it proxies to
    detail: str = ""


def parse_serve_status(data: dict[str, object], port: int) -> tuple[str | None, str | None]:
    """Find a web handler proxying to 127.0.0.1:<port> in `tailscale serve status --json`."""
    web = data.get("Web")
    if not isinstance(web, dict):
        return None, None
    for hostport, cfg in web.items():
        handlers = cfg.get("Handlers", {}) if isinstance(cfg, dict) else {}
        for _path, h in handlers.items():
            proxy = h.get("Proxy", "") if isinstance(h, dict) else ""
            if proxy.rstrip("/").endswith(f":{port}") and ("127.0.0.1" in proxy or "localhost" in proxy):
                host, _, hp = hostport.rpartition(":")
                return f"https://{host}" + ("" if hp in ("443", "") else f":{hp}"), proxy
    return None, None


class Tailscale:
    def __init__(self, runner: Runner = run, exe: str | None = None):
        self.run = runner
        self.exe = exe or binary()

    def status(self, port: int) -> Exposure:
        if not self.exe:
            return Exposure(False, detail="tailscale not installed")
        r = self.run(argv(self.exe, "serve", "status", "--json"))
        if not r.ok:
            return Exposure(False, detail=(r.err or r.out).strip()[:200])
        try:
            data = json.loads(r.out or "{}")
        except json.JSONDecodeError:
            return Exposure(True, detail="unparseable serve status")
        url, target = parse_serve_status(data if isinstance(data, dict) else {}, port)
        return Exposure(True, url, target, "" if url else "not exposed")

    def expose(self, port: int, https_port: int = 443) -> Exposure:
        if not self.exe:
            raise RuntimeError("tailscale not installed")
        r = self.run(argv(self.exe, "serve", "--bg", f"--https={https_port}", str(port)))
        if not r.ok:
            raise RuntimeError(f"tailscale serve failed: {(r.err or r.out).strip()}")
        return self.status(port)

    def unexpose(self, https_port: int = 443) -> None:
        if not self.exe:
            return
        r = self.run(argv(self.exe, "serve", f"--https={https_port}", "off"))
        if not r.ok and "not found" not in (r.err + r.out).lower():
            raise RuntimeError(f"tailscale serve off failed: {(r.err or r.out).strip()}")
