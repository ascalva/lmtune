"""API key file: created 0600 in a 0700 directory; the key itself is never printed unless asked."""

from __future__ import annotations

import os
import secrets
import stat
from pathlib import Path

PREFIX = "sk-lmtune-"


def generate() -> str:
    return PREFIX + secrets.token_hex(32)


def write(path: Path, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key)  # no trailing newline: consumers that don't strip (podman secrets) get the exact key
    tmp.replace(path)  # atomic: readers never see a half-written key


def ensure(path: Path) -> bool:
    """Create the key if missing. Returns True if a new key was written."""
    if path.exists():
        return False
    write(path, generate())
    return True


def read(path: Path) -> str:
    return path.read_text().strip()


def problems(path: Path) -> list[str]:
    if not path.exists():
        return [f"{path} does not exist (run `lmtune key init`)"]
    out = []
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode & 0o077:
        out.append(f"{path} is mode {mode:o}; must not be readable by group/others (chmod 600)")
    dmode = stat.S_IMODE(path.parent.stat().st_mode)
    if dmode & 0o022:
        out.append(f"{path.parent} is writable by group/others (mode {dmode:o})")
    if len(read(path)) < 32:
        out.append(f"{path} holds a key shorter than 32 characters")
    return out
