"""Model profiles (what to run, what to search, what must stay fixed) and user constraints."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, cast

DEFAULT_CONSTRAINTS = Path("~/.config/lmtune/constraints.toml").expanduser()


@dataclass
class Model:
    name: str
    hf_repo: str
    candidates: list[str]
    dir: Path
    extra_files: list[str] = field(default_factory=list)


@dataclass
class Workload:
    """Fixed across every config: changing these would change quality, not speed."""

    prompts: Path
    seeds: list[int] = field(default_factory=lambda: [1, 2])
    max_tokens: int = 1024
    greedy_tokens: int = 256
    sampling: dict[str, Any] = field(default_factory=dict)  # temp/top_p/top_k/min_p; empty -> GGUF defaults
    reasoning: dict[str, Any] = field(default_factory=dict)  # passed through as server flags, never searched


@dataclass
class Serve:
    ctx: int = 65536
    parallel: int = 1
    host: str = "127.0.0.1"
    port: int = 8080
    extra: list[str] = field(default_factory=lambda: ["--cors-origins", "localhost"])


@dataclass
class Search:
    fa: list[str] = field(default_factory=lambda: ["off", "on"])
    ubatch: list[int] = field(default_factory=lambda: [512, 1024])
    kv_types: list[str] = field(default_factory=lambda: ["f16", "q8_0"])
    depths: list[int] = field(default_factory=lambda: [0, 8192])
    spec_n_max: list[int] = field(default_factory=lambda: [1, 2])
    spec_n_max_banned: list[int] = field(default_factory=list)
    draft_sampling: list[str] = field(default_factory=lambda: ["greedy", "probabilistic"])
    draft_p_min: list[float] = field(default_factory=list)
    cache_ram_mib: list[int] = field(default_factory=lambda: [0, 2048])
    reps: int = 2

    def allowed_n_max(self) -> list[int]:
        return [n for n in self.spec_n_max if n not in self.spec_n_max_banned]


@dataclass
class Profile:
    model: Model
    workload: Workload
    serve: Serve
    search: Search
    source: Path


@dataclass
class Constraints:
    download_ask_gb: float = 10.0  # ask before any single download larger than this
    min_free_disk_gb: float = 10.0  # never plan downloads that leave less than this
    memory_headroom_gb: float = 6.0  # leave this much RAM for the OS and other apps
    swap_growth_limit_mb: int = 512  # abort a config if system swap grows more than this
    allow_sudo: bool = False  # the tool never runs sudo; it prints the command instead


def _kw[T](cls: type[T], d: dict[str, Any]) -> T:
    known = {f.name for f in fields(cast("Any", cls))}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__.lower()} keys: {', '.join(sorted(unknown))}")
    return cls(**d)


def load_profile(path: str | Path) -> Profile:
    path = Path(path)
    raw = tomllib.loads(path.read_text())
    m = dict(raw["model"])
    m["dir"] = Path(m["dir"]).expanduser()
    w = dict(raw.get("workload", {}))
    w["prompts"] = (path.parent / w.get("prompts", "../prompts/default.json")).resolve()
    w.setdefault("sampling", raw.get("sampling", {}))
    w.setdefault("reasoning", raw.get("reasoning", {}))
    prof = Profile(
        _kw(Model, m), _kw(Workload, w), _kw(Serve, raw.get("serve", {})), _kw(Search, raw.get("search", {})), path
    )
    if prof.serve.host not in ("127.0.0.1", "::1", "localhost"):
        raise ValueError(f"serve.host must be loopback, got {prof.serve.host!r}")
    return prof


def load_constraints(path: str | Path = DEFAULT_CONSTRAINTS) -> Constraints:
    path = Path(path)
    return _kw(Constraints, tomllib.loads(path.read_text())) if path.exists() else Constraints()
