"""Read GGUF headers (metadata + tensor names) without touching the weights.

Gives the facts the tuner needs: architecture, attention layout (for KV-cache sizing),
built-in MTP layers, publisher sampling defaults, and the chat template's reasoning defaults.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

# GGUF value types
_U8, _I8, _U16, _I16, _U32, _I32, _F32, _BOOL, _STR, _ARR, _U64, _I64, _F64 = range(13)
_SCALAR = {
    _U8: "<B",
    _I8: "<b",
    _U16: "<H",
    _I16: "<h",
    _U32: "<I",
    _I32: "<i",
    _F32: "<f",
    _BOOL: "<?",
    _U64: "<Q",
    _I64: "<q",
    _F64: "<d",
}
_MAX_ARRAY = 64  # longer arrays (vocab, merges) are summarized, not loaded

# bytes per element of KV cache types (block size 32 for the quantized ones)
KV_BYTES = {
    "f32": 4.0,
    "f16": 2.0,
    "bf16": 2.0,
    "q8_0": 34 / 32,
    "q5_1": 24 / 32,
    "q5_0": 22 / 32,
    "q4_1": 20 / 32,
    "q4_0": 18 / 32,
    "iq4_nl": 18 / 32,
}


class GGUFError(ValueError):
    pass


def _rd(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise GGUFError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _rint(f: BinaryIO, fmt: str) -> int:
    v = _rd(f, fmt)
    if not isinstance(v, int):
        raise GGUFError(f"expected an integer for {fmt}")
    return v


def _rstr(f: BinaryIO) -> str:
    return f.read(_rint(f, "<Q")).decode("utf-8", "replace")


def _rval(f: BinaryIO, t: int) -> Any:
    if t == _STR:
        return _rstr(f)
    if t == _ARR:
        et, n = _rint(f, "<I"), _rint(f, "<Q")
        if et == _STR:
            items = [_rstr(f) for _ in range(n)]
            return items if n <= _MAX_ARRAY else f"<{n} strings>"
        if et not in _SCALAR:
            raise GGUFError(f"unsupported array element type {et}")
        fmt = _SCALAR[et]
        data = f.read(struct.calcsize(fmt) * n)
        return list(struct.unpack("<" + fmt[1] * n, data)) if n <= _MAX_ARRAY else f"<{n} values>"
    if t not in _SCALAR:
        raise GGUFError(f"unsupported value type {t}")
    return _rd(f, _SCALAR[t])


def read_header(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    """Return (metadata dict, tensor names)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise GGUFError(f"{path}: not a GGUF file")
        version = _rint(f, "<I")
        if version < 2:
            raise GGUFError(f"{path}: GGUF v{version} not supported")
        n_tensors, n_kv = _rint(f, "<Q"), _rint(f, "<Q")
        meta: dict[str, Any] = {"gguf.version": version}
        for _ in range(n_kv):
            key = _rstr(f)
            meta[key] = _rval(f, _rint(f, "<I"))
        names = []
        for _ in range(n_tensors):
            names.append(_rstr(f))
            n_dims = _rint(f, "<I")
            f.read(8 * n_dims + 4 + 8)  # dims, type, offset
    return meta, names


@dataclass
class ReasoningDefaults:
    """What the chat template does when the client sends nothing."""

    thinking_default_on: bool | None = None
    effort_default: str | None = None
    effort_levels: list[str] = field(default_factory=list)
    preserve_thinking_default_on: bool | None = None


@dataclass
class ModelInfo:
    path: str
    arch: str
    name: str
    file_type: int | None
    block_count: int
    mtp_layers: int
    attention_layers: int
    head_count_kv: int
    key_length: int
    value_length: int
    context_length: int | None
    sampling: dict[str, Any]
    chat_template: str | None
    reasoning: ReasoningDefaults

    def kv_bytes_per_token(self, cache_type: str = "f16") -> float:
        """KV-cache bytes per token of context for one sequence (attention layers only)."""
        return self.attention_layers * self.head_count_kv * (self.key_length + self.value_length) * KV_BYTES[cache_type]

    def kv_bytes(self, n_ctx: int, cache_type: str = "f16") -> float:
        return self.kv_bytes_per_token(cache_type) * n_ctx


def parse_reasoning(template: str | None) -> ReasoningDefaults:
    r = ReasoningDefaults()
    if not template:
        return r
    # e.g. {%- if enable_thinking is undefined or enable_thinking is true %}
    if re.search(r"enable_thinking is undefined or enable_thinking is true", template):
        r.thinking_default_on = True
    elif re.search(r"enable_thinking is defined and enable_thinking is true", template):
        r.thinking_default_on = False
    m = re.search(r"reasoning_effort\s*\|\s*default\(\s*['\"](\w+)['\"]\s*\)", template)
    if m:
        r.effort_default = m.group(1)
    m = re.search(r"resolved_reasoning_effort not in \(([^)]*)\)", template)
    if m:
        r.effort_levels = re.findall(r"['\"](\w+)['\"]", m.group(1))
    if re.search(r"preserve_thinking is undefined or preserve_thinking is true", template):
        r.preserve_thinking_default_on = True
    return r


def inspect(path: str | Path) -> ModelInfo:
    meta, names = read_header(path)
    arch = meta.get("general.architecture", "")

    def a(key: str, default: Any = None) -> Any:
        return meta.get(f"{arch}.{key}", default)

    blocks = int(a("block_count", 0) or 0)
    mtp = int(a("nextn_predict_layers", 0) or 0)
    if not mtp:  # some converters only leave tensor names
        mtp = len({n.split(".")[1] for n in names if ".nextn." in n})
    main_layers = blocks - mtp
    interval = a("full_attention_interval")
    # hybrid models: every Nth layer is full attention (has KV), the rest are recurrent
    attn_layers = main_layers // int(interval) if interval else main_layers
    heads = int(a("attention.head_count", 0) or 0)
    kv_heads = a("attention.head_count_kv", heads)
    kv_heads = int(max(kv_heads) if isinstance(kv_heads, list) else kv_heads or 0)
    head_dim = int(a("embedding_length", 0) or 0) // heads if heads else 0
    sampling = {k.removeprefix("general.sampling."): v for k, v in meta.items() if k.startswith("general.sampling.")}
    template = meta.get("tokenizer.chat_template")
    return ModelInfo(
        path=str(path),
        arch=arch,
        name=meta.get("general.name", Path(path).stem),
        file_type=meta.get("general.file_type"),
        block_count=blocks,
        mtp_layers=mtp,
        attention_layers=attn_layers,
        head_count_kv=kv_heads,
        key_length=int(a("attention.key_length", head_dim) or 0),
        value_length=int(a("attention.value_length", head_dim) or 0),
        context_length=a("context_length"),
        sampling=sampling,
        chat_template=template,
        reasoning=parse_reasoning(template),
    )
