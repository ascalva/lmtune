"""Subprocess seam: every external command goes through a Runner so tests can fake it."""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Result:
    code: int
    out: str
    err: str

    @property
    def ok(self) -> bool:
        return self.code == 0


Runner = Callable[[Sequence[str]], Result]


def run(cmd: Sequence[str]) -> Result:
    try:
        p = subprocess.run(list(cmd), capture_output=True, text=True, timeout=120, check=False)
    except FileNotFoundError:
        return Result(127, "", f"{cmd[0]}: not found")
    except subprocess.TimeoutExpired:
        return Result(124, "", f"{cmd[0]}: timed out")
    return Result(p.returncode, p.stdout, p.stderr)
