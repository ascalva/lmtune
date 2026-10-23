"""launchd user agent for the server: plist generation and launchctl wrappers."""

from __future__ import annotations

import os
import plistlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .proc import Runner, run


def domain() -> str:
    return f"gui/{os.getuid()}"


def plist(
    label: str, program: list[str], log: Path, *, keep_alive: bool = True, process_type: str = "Interactive"
) -> bytes:
    """Agent definition. KeepAlive restarts only on a crash (non-zero exit), throttled so a broken
    config can't spin; Interactive keeps macOS from treating inference as background work
    (helpers that must never compete with it pass "Background")."""
    d: dict[str, object] = {
        "Label": label,
        "ProgramArguments": program,
        "RunAtLoad": True,
        "KeepAlive": {"SuccessfulExit": False} if keep_alive else False,
        "ThrottleInterval": 30,
        "ProcessType": process_type,
        "StandardOutPath": str(log),
        "StandardErrorPath": str(log),
        "EnvironmentVariables": {"PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"},
    }
    return plistlib.dumps(d)


@dataclass(frozen=True)
class AgentState:
    loaded: bool
    state: str | None = None
    pid: int | None = None
    last_exit: str | None = None
    runs: int | None = None


def parse_print(text: str) -> AgentState:
    def grab(pattern: str) -> str | None:
        m = re.search(pattern, text, re.M)
        return m.group(1).strip() if m else None

    pid = grab(r"^\s*pid = (\d+)")
    runs = grab(r"^\s*runs = (\d+)")
    return AgentState(
        loaded=True,
        state=grab(r"^\s*state = (.+)$"),
        pid=int(pid) if pid else None,
        last_exit=grab(r"^\s*last exit code = (.+)$"),
        runs=int(runs) if runs else None,
    )


class Launchd:
    def __init__(self, label: str, runner: Runner = run, sleep: Callable[[float], None] = time.sleep):
        self.label = label
        self.run = runner
        self.sleep = sleep

    @property
    def target(self) -> str:
        return f"{domain()}/{self.label}"

    def state(self) -> AgentState:
        r = self.run(["launchctl", "print", self.target])
        return parse_print(r.out) if r.ok else AgentState(loaded=False)

    def bootstrap(self, plist_path: Path, attempts: int = 5) -> None:
        """Load the agent. launchd answers `5: Input/output error` while a just-booted-out
        instance is still being torn down (measured: fails immediately, succeeds ~2 s later),
        so error 5 is retried with backoff; anything else fails at once."""
        for i in range(attempts):
            r = self.run(["launchctl", "bootstrap", domain(), str(plist_path)])
            if r.ok or "already" in (r.err + r.out).lower():
                return
            if r.code != 5 or i == attempts - 1:
                raise RuntimeError(f"launchctl bootstrap failed ({r.code}): {r.err.strip() or r.out.strip()}")
            self.sleep(0.5 * 2**i)

    def bootout(self, wait_s: float = 15) -> None:
        r = self.run(["launchctl", "bootout", self.target])
        # 3/36/113: not loaded / no such process — already out, which is what we wanted
        if not r.ok and r.code not in (3, 36, 113) and "no such process" not in r.err.lower():
            raise RuntimeError(f"launchctl bootout failed ({r.code}): {r.err.strip()}")
        # bootout returns before launchd has finished unloading; wait until it really has
        deadline = time.monotonic() + wait_s
        while self.state().loaded and time.monotonic() < deadline:
            self.sleep(0.25)

    def kickstart(self, *, restart: bool = False) -> None:
        cmd = ["launchctl", "kickstart"] + (["-k"] if restart else []) + [self.target]
        r = self.run(cmd)
        if not r.ok:
            raise RuntimeError(f"launchctl kickstart failed ({r.code}): {r.err.strip()}")
