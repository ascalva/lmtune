"""Sample a process's memory and system swap while it runs; abort it if swap grows."""

from __future__ import annotations

import contextlib
import os
import re
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field

from .machine import swap_used_mb


def footprint_mb(pid: int) -> float | None:
    """macOS phys_footprint: the process's own allocations (excludes clean mmapped weights)."""
    out = subprocess.run(["footprint", "-p", str(pid)], capture_output=True, text=True, check=False).stdout
    m = re.search(r"Footprint: ([0-9.]+) ([KMG]B)", out)
    if not m:
        return None
    return float(m.group(1)) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[m.group(2)]


def rss_mb(pid: int) -> float | None:
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False
    ).stdout.strip()
    return int(out) / 1024 if out.isdigit() else None


@dataclass
class Sample:
    t: float
    footprint_mb: float | None
    rss_mb: float | None
    swap_mb: float | None


@dataclass
class Monitor:
    """Context manager: `with Monitor(pid) as mon: ...` then mon.peak_*/mon.tripped."""

    pid: int
    swap_limit_mb: float = 512
    interval: float = 2.0
    samples: list[Sample] = field(default_factory=list)
    tripped: str | None = None

    def __post_init__(self) -> None:
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._base = swap_used_mb()

    def _target(self) -> int:
        """The process to sample: the newest child if `pid` is a wrapper like /usr/bin/time."""
        out = subprocess.run(
            ["pgrep", "-n", "-P", str(self.pid)], capture_output=True, text=True, check=False
        ).stdout.strip()
        return int(out) if out.isdigit() else self.pid

    def _alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
            return True
        except OSError:
            return False

    def _run(self) -> None:
        while not self._stop.is_set() and self._alive():
            tgt = self._target()
            s = Sample(time.time(), footprint_mb(tgt), rss_mb(tgt), swap_used_mb())
            self.samples.append(s)
            if self._base is not None and s.swap_mb is not None and s.swap_mb - self._base > self.swap_limit_mb:
                self.tripped = f"swap grew {s.swap_mb - self._base:.0f} MB (> {self.swap_limit_mb} MB)"
                for victim in {tgt, self.pid}:  # the child too: killing a wrapper doesn't stop it
                    with contextlib.suppress(OSError):
                        os.kill(victim, signal.SIGTERM)
                return
            self._stop.wait(self.interval)

    def __enter__(self) -> Monitor:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=10)

    def _peak(self, attr: str) -> float | None:
        vals = [getattr(s, attr) for s in self.samples if getattr(s, attr) is not None]
        return max(vals) if vals else None

    @property
    def peak_footprint_mb(self) -> float | None:
        return self._peak("footprint_mb")

    @property
    def peak_rss_mb(self) -> float | None:
        return self._peak("rss_mb")

    @property
    def swap_delta_mb(self) -> float | None:
        vals = [s.swap_mb for s in self.samples if s.swap_mb is not None]
        return (max(vals) - self._base) if vals and self._base is not None else None
