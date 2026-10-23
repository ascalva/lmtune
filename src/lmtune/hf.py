"""Hugging Face repo listing, download planning against machine constraints, verified fetch."""

from __future__ import annotations

import difflib
import hashlib
import json
import subprocess
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._types import Log

API = "https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
RESOLVE = "https://huggingface.co/{repo}/resolve/main/{path}"


@dataclass
class RepoFile:
    path: str
    size: int
    sha256: str | None


def parse_tree(data: list[dict[str, Any]]) -> dict[str, RepoFile]:
    return {
        f["path"]: RepoFile(f["path"], int(f.get("size", 0)), (f.get("lfs") or {}).get("oid"))
        for f in data
        if f.get("type") == "file"
    }


def list_files(repo: str) -> dict[str, RepoFile]:
    with urllib.request.urlopen(API.format(repo=repo), timeout=60) as r:
        return parse_tree(json.load(r))


@dataclass
class PlanItem:
    file: RepoFile
    local: Path
    have: bool  # already on disk with the right size
    fits_memory: bool
    needs_approval: bool  # larger than the user's ask-before-download threshold
    note: str = ""


def plan(
    files: dict[str, RepoFile],
    wanted: list[str],
    dest: Path,
    *,
    budget_bytes: float,
    ask_bytes: float,
    disk_free: int,
    min_free: float,
    runtime_bytes: float = 0.0,
) -> tuple[list[PlanItem], list[str]]:
    """Decide what to fetch. `runtime_bytes` is the non-weight memory (KV + buffers) a config needs."""
    items, problems = [], []
    to_fetch = 0
    for name in wanted:
        f = files.get(name)
        if f is None:
            problems.append(f"{name}: not in repo (closest: {', '.join(_closest(name, files))})")
            continue
        local = dest / name
        have = local.exists() and local.stat().st_size == f.size
        fits = f.size + runtime_bytes <= budget_bytes
        item = PlanItem(f, local, have, fits, needs_approval=(not have and f.size > ask_bytes))
        if not fits:
            item.note = f"needs {(f.size + runtime_bytes) / 1e9:.1f} GB > budget {budget_bytes / 1e9:.1f} GB"
        if not have and fits:
            to_fetch += f.size - (local.stat().st_size if local.exists() else 0)
        items.append(item)
    if disk_free - to_fetch < min_free:
        problems.append(
            f"downloads need {to_fetch / 1e9:.1f} GB; only {disk_free / 1e9:.1f} GB free "
            f"and {min_free / 1e9:.0f} GB must stay free"
        )
    return items, problems


def _closest(name: str, files: dict[str, RepoFile], n: int = 3) -> list[str]:
    return difflib.get_close_matches(name, [p for p in files if p.endswith(".gguf")], n=n, cutoff=0.5)


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def fetch(repo: str, f: RepoFile, dest: Path, attempts: int = 20, log: Log = print) -> Path:
    """Resumable download with an outer retry loop, then sha256 verification.

    curl's own --retry doesn't cover connection resets mid-body without --retry-all-errors,
    and even that gives up; the outer loop resumes from the partial file (-C -).
    """
    out = dest / f.path
    out.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, attempts + 1):
        if out.exists() and out.stat().st_size == f.size:
            break
        log(f"fetch {f.path} attempt {attempt} ({out.stat().st_size if out.exists() else 0}/{f.size} bytes)")
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-all-errors",
                "--retry-delay",
                "5",
                "-C",
                "-",
                "-s",
                "-S",
                "-o",
                str(out),
                RESOLVE.format(repo=repo, path=f.path),
            ],
            check=False,
        )
        if out.exists() and out.stat().st_size > f.size:
            raise RuntimeError(f"{out} is larger than expected; refusing to continue")
        time.sleep(0 if out.exists() and out.stat().st_size == f.size else 10)
    else:
        raise RuntimeError(f"{f.path}: download incomplete after {attempts} attempts")
    if f.sha256:
        got = sha256(out)
        if got != f.sha256:
            raise RuntimeError(f"{f.path}: sha256 mismatch (got {got}, expected {f.sha256})")
    return out
