"""Serve configuration (`~/.config/lmtune/serve.toml`): what the managed server runs.

Defaults are the configuration that won the Qwen3.8-27B tuning run on an M2 Pro / 32 GB
(qwen38-llamacpp/results.md); `lmtune serve init` writes them with comments.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal

LOOPBACK = ("127.0.0.1", "::1", "localhost")
KV_TYPES = ("f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0")
Backend = Literal["native", "podman"]


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Sampling:
    temp: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0


@dataclass(frozen=True)
class ServeConfig:
    model: Path
    alias: str = "qwen3.8-27b"
    host: str = "127.0.0.1"
    port: int = 8080
    ctx: int = 65536
    parallel: int = 1
    ubatch: int = 512
    flash_attn: str = "on"
    kv: str = "f16"
    cache_ram_mib: int = 2048
    sleep_idle_s: int = 600
    spec: tuple[str, ...] = (
        "--spec-type",
        "draft-mtp",
        "--spec-draft-n-max",
        "2",
        "--spec-draft-sampling",
        "probabilistic",
    )
    sampling: Sampling = field(default_factory=Sampling)
    # --metrics: Prometheus counters for `lmtune dash`, behind the API key like every endpoint but
    # /health; A-B-A-B-A with 1/s scraping measured no decode cost (docs/ARCHITECTURE.md)
    extra: tuple[str, ...] = ("--cors-origins", "localhost", "--metrics")
    api_key_file: Path | None = None
    # vision projector (mmproj GGUF) for image input; None = text only. Measured with Qwen3.8-27B:
    # +0.9 GB, no decode cost, MTP unaffected (docs/ARCHITECTURE.md "Media")
    mmproj: Path | None = None
    backend: Backend = "native"
    image: str = "ghcr.io/ggml-org/llama.cpp:server-vulkan-b11669"

    def problems(self) -> list[str]:
        """Everything wrong with this config; empty means valid."""
        out: list[str] = []
        if self.host not in LOOPBACK:
            out.append(f"host must be loopback ({', '.join(LOOPBACK)}); use `lmtune expose` for remote access")
        if not 1 <= self.port <= 65535:
            out.append(f"port {self.port} out of range")
        if self.kv not in KV_TYPES:
            out.append(f"kv {self.kv!r} not one of {', '.join(KV_TYPES)}")
        if self.flash_attn not in ("on", "off", "auto"):
            out.append("flash_attn must be on, off or auto")
        if self.kv != "f16" and self.flash_attn == "off":
            out.append("quantized KV cache needs flash_attn on")
        for name in ("ctx", "parallel", "ubatch"):
            if getattr(self, name) < 1:
                out.append(f"{name} must be >= 1")
        if self.cache_ram_mib < 0:
            out.append("cache_ram_mib must be >= 0")
        if self.sleep_idle_s < -1 or self.sleep_idle_s == 0:
            out.append("sleep_idle_s must be -1 (never) or > 0")
        if "--host" in self.extra or "--port" in self.extra:
            out.append("set host/port through their own keys, not extra")
        if "--api-key" in self.extra:
            out.append("never put an API key in extra; use api_key_file")
        if self.api_key_file is None:
            out.append("api_key_file is required (run `lmtune key init`)")
        if self.backend not in ("native", "podman"):
            out.append(f"backend {self.backend!r} must be native or podman")
        if self.mmproj is not None and self.backend == "podman" and self.mmproj.parent != self.model.parent:
            out.append("podman backend: mmproj must sit next to the model (one folder is mounted)")
        if any(a in self.extra for a in ("--mmproj", "-mm", "--mmproj-url", "-mmu")):
            out.append("set the vision projector through mmproj, not extra")
        return out

    def check(self) -> ServeConfig:
        p = self.problems()
        if p:
            raise ConfigError("invalid serve config:\n  - " + "\n  - ".join(p))
        return self

    def argv(
        self,
        *,
        model_path: str | None = None,
        mmproj_path: str | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> list[str]:
        """llama-server arguments (no binary). Overrides are for the container backend."""
        s = self.sampling
        args = [
            "-m",
            model_path or str(self.model),
            "-a",
            self.alias,
            "-ngl",
            "99",
            "-fa",
            self.flash_attn,
            "-ub",
            str(self.ubatch),
            "-np",
            str(self.parallel),
            "-c",
            str(self.ctx),
            "-ctk",
            self.kv,
            "-ctv",
            self.kv,
            "--cache-ram",
            str(self.cache_ram_mib),
            "--temp",
            _num(s.temp),
            "--top-p",
            _num(s.top_p),
            "--top-k",
            str(s.top_k),
            "--min-p",
            _num(s.min_p),
            "--host",
            host or self.host,
            "--port",
            str(port or self.port),
        ]
        if self.sleep_idle_s > 0:
            args += ["--sleep-idle-seconds", str(self.sleep_idle_s)]
        if self.api_key_file is not None:
            args += ["--api-key-file", str(self.api_key_file)]
        if self.mmproj is not None:
            args += ["--mmproj", mmproj_path or str(self.mmproj)]
        return args + list(self.extra) + list(self.spec)

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"


def _num(x: float) -> str:
    return f"{x:g}"


def load(path: Path) -> ServeConfig:
    try:
        raw = tomllib.loads(path.read_text())
    except FileNotFoundError:
        raise ConfigError(f"{path} not found; run `lmtune serve init`") from None
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: {e}") from None
    return from_dict(raw)


def from_dict(raw: dict[str, Any]) -> ServeConfig:
    srv = dict(raw.get("server", {}))
    known = {f.name for f in fields(ServeConfig)}
    unknown = set(srv) - known
    if unknown:
        raise ConfigError(f"unknown [server] keys: {', '.join(sorted(unknown))}")
    if "model" not in srv:
        raise ConfigError("[server] model is required")
    srv["model"] = Path(srv["model"]).expanduser()
    for key in ("api_key_file", "mmproj"):
        if key in srv:
            srv[key] = Path(srv[key]).expanduser() if srv[key] else None
    for key in ("spec", "extra"):
        if key in srv:
            if not isinstance(srv[key], list) or not all(isinstance(a, str) for a in srv[key]):
                raise ConfigError(f"[server] {key} must be a list of strings")
            srv[key] = tuple(srv[key])
    samp = raw.get("sampling", {})
    unknown = set(samp) - {f.name for f in fields(Sampling)}
    if unknown:
        raise ConfigError(f"unknown [sampling] keys: {', '.join(sorted(unknown))}")
    try:
        return ServeConfig(**srv, sampling=Sampling(**samp))
    except TypeError as e:
        raise ConfigError(str(e)) from None


def to_toml(cfg: ServeConfig) -> str:
    """Serialize with comments (stdlib has no TOML writer; the schema is small and flat)."""
    q = _toml_value
    s = cfg.sampling
    lines = [
        "# lmtune serve config. Defaults are the measured winner for Qwen3.8-27B on M2 Pro / 32 GB.",
        "# Edit, then `lmtune serve restart`. Validate with `lmtune doctor`.",
        "",
        "[server]",
        f"model = {q(str(cfg.model))}",
        f"alias = {q(cfg.alias)}            # model id clients use",
        f"host = {q(cfg.host)}       # loopback only; remote access goes through `lmtune expose`",
        f"port = {cfg.port}",
        f"ctx = {cfg.ctx}",
        f"parallel = {cfg.parallel}                 # slots; requests queue beyond this",
        f"ubatch = {cfg.ubatch}",
        f"flash_attn = {q(cfg.flash_attn)}",
        f"kv = {q(cfg.kv)}                  # f16 measured 20-32% faster than q8_0 at 8-16K context",
        f"cache_ram_mib = {cfg.cache_ram_mib}          # saved conversation states (~0.5-1 GB each); 0 = none",
        f"sleep_idle_s = {cfg.sleep_idle_s}            # unload the model after this many idle seconds; -1 = never",
        f"spec = {q(list(cfg.spec))}",
        f"extra = {q(list(cfg.extra))}",
        f"api_key_file = {q(str(cfg.api_key_file)) if cfg.api_key_file else q('')}",
        f'mmproj = {q(str(cfg.mmproj)) if cfg.mmproj else q("")}   # vision projector (image input); "" = text only',
        f"backend = {q(cfg.backend)}         # native (Metal) | podman (Vulkan in the podman VM)",
        f"image = {q(cfg.image)}",
        "",
        "# fixed workload settings (model card thinking-mode defaults); reasoning uses template defaults",
        "[sampling]",
        f"temp = {_num(s.temp)}",
        f"top_p = {_num(s.top_p)}",
        f"top_k = {s.top_k}",
        f"min_p = {_num(s.min_p)}",
        "",
    ]
    return "\n".join(lines)


def _toml_value(v: object) -> str:
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, list):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    raise TypeError(type(v))


def with_overrides(cfg: ServeConfig, **kw: Any) -> ServeConfig:
    return replace(cfg, **kw)
