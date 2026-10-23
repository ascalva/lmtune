"""Discover which flags the installed llama.cpp build accepts.

Flags churn between builds (renames, removals, new spec types), so every flag a profile
or stage uses is checked against `<tool> --help` before anything runs.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass, field
from functools import cache

_OPT_LINE = re.compile(r"^\s{0,4}(-\S.*?)(?:\s{2,}|$)")
_CHOICE_BLOCK = re.compile(r"[\[{<]([^\]}>]+)[\]}>]")


class MissingFlags(RuntimeError):
    pass


@dataclass
class ToolFlags:
    tool: str
    build: int | None
    commit: str | None
    flags: set[str] = field(default_factory=set)
    args: dict[str, str] = field(default_factory=dict)  # flag -> raw argument spec

    def has(self, flag: str) -> bool:
        return flag in self.flags

    def choices(self, flag: str) -> list[str]:
        """Enumerated values for a flag, e.g. --spec-type or --flash-attn."""
        spec = self.args.get(flag, "")
        m = _CHOICE_BLOCK.search(spec)
        body = m.group(1) if m else spec
        if "|" in body:
            return [c for c in body.split("|") if c]
        if "," in body and " " not in body:
            return [c for c in body.split(",") if c and c != "..."]
        return []

    def missing(self, argv: list[str] | str) -> list[str]:
        """Flags in argv (a command line or list) that this build doesn't know."""
        if isinstance(argv, str):
            argv = shlex.split(argv)
        return [a for a in argv if a.startswith("-") and not _is_number(a) and a not in self.flags]

    def require(self, argv: list[str] | str) -> None:
        bad = self.missing(argv)
        if bad:
            raise MissingFlags(f"{self.tool} (build {self.build}) does not accept: {' '.join(bad)}")


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_help(text: str) -> tuple[set[str], dict[str, str]]:
    flags: set[str] = set()
    args: dict[str, str] = {}
    for line in text.splitlines():
        # aliases are padded for alignment ("-s,    --seed SEED"); don't mistake that for the description gap
        m = _OPT_LINE.match(re.sub(r",\s+(?=-)", ", ", line))
        if not m:
            continue
        head = m.group(1)
        names: list[str] = []
        rest: list[str] = []
        for tok in head.split():
            t = tok.rstrip(",")
            if t.startswith("-") and not rest and re.match(r"^--?[A-Za-z]", t):
                names.append(t)
            else:
                rest.append(tok)
        argspec = " ".join(rest)
        for n in names:
            flags.add(n)
            if argspec:
                args[n] = argspec
    return flags, args


def parse_version(text: str) -> tuple[int | None, str | None]:
    m = re.search(r"build (\d+), commit ([0-9a-f]+)", text)
    return (int(m.group(1)), m.group(2)) if m else (None, None)


@cache
def discover(tool: str) -> ToolFlags:
    def run(*a: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run([tool, *a], capture_output=True, text=True, timeout=120, check=False)

    help_out = run("--help")
    flags, args = parse_help(help_out.stdout + "\n" + help_out.stderr)
    ver = run("--version")
    build, commit = parse_version(ver.stdout + ver.stderr)
    return ToolFlags(tool=tool, build=build, commit=commit, flags=flags, args=args)
