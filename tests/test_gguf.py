import struct
from pathlib import Path

import pytest

from lmtune import gguf

QWEN = Path.home() / "models/qwen38/Qwen3.8-27B-Q4_0.gguf"

TEMPLATE = """
{%- if enable_thinking is undefined or enable_thinking is true %}
    {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    {%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}
        {{- raise_exception('bad') }}
    {%- endif %}
{%- endif %}
{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}
"""


def _s(x: str) -> bytes:
    b = x.encode()
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, value) -> bytes:
    out = _s(key)
    if isinstance(value, bool):
        return out + struct.pack("<I", 7) + struct.pack("<?", value)
    if isinstance(value, int):
        return out + struct.pack("<I", 4) + struct.pack("<I", value)
    if isinstance(value, float):
        return out + struct.pack("<I", 6) + struct.pack("<f", value)
    if isinstance(value, str):
        return out + struct.pack("<I", 8) + _s(value)
    if isinstance(value, list):  # u32 array
        return (
            out
            + struct.pack("<I", 9)
            + struct.pack("<I", 4)
            + struct.pack("<Q", len(value))
            + b"".join(struct.pack("<I", v) for v in value)
        )
    raise TypeError(value)


def write_gguf(path: Path, meta: dict, tensors: list[str]) -> Path:
    body = b"".join(_kv(k, v) for k, v in meta.items())
    for name in tensors:
        body += _s(name) + struct.pack("<I", 1) + struct.pack("<Q", 32) + struct.pack("<I", 0) + struct.pack("<Q", 0)
    path.write_bytes(
        b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors)) + struct.pack("<Q", len(meta)) + body
    )
    return path


def hybrid_meta(**over):
    meta = {
        "general.architecture": "qwen35",
        "general.name": "Tiny",
        "general.file_type": 2,
        "general.sampling.top_k": 20,
        "general.sampling.temp": 1.0,
        "qwen35.block_count": 65,
        "qwen35.nextn_predict_layers": 1,
        "qwen35.full_attention_interval": 4,
        "qwen35.embedding_length": 5120,
        "qwen35.attention.head_count": 24,
        "qwen35.attention.head_count_kv": 4,
        "qwen35.attention.key_length": 256,
        "qwen35.attention.value_length": 256,
        "qwen35.context_length": 262144,
        "tokenizer.chat_template": TEMPLATE,
    }
    meta.update(over)
    return meta


def test_hybrid_layout_and_kv_size(tmp_path):
    info = gguf.inspect(write_gguf(tmp_path / "m.gguf", hybrid_meta(), ["blk.0.attn_q.weight"]))
    assert info.mtp_layers == 1
    assert info.attention_layers == 16  # 64 main layers, every 4th is attention
    assert info.kv_bytes_per_token("f16") == 16 * 4 * 512 * 2
    # q8_0 is 34 bytes per 32 elements
    assert info.kv_bytes_per_token("q8_0") == pytest.approx(16 * 4 * 512 * 34 / 32)
    assert info.sampling == {"top_k": 20, "temp": 1.0}


def test_dense_model_defaults_head_dim(tmp_path):
    meta = {
        "general.architecture": "llama",
        "llama.block_count": 32,
        "llama.embedding_length": 4096,
        "llama.attention.head_count": 32,
        "llama.attention.head_count_kv": 8,
    }
    info = gguf.inspect(write_gguf(tmp_path / "d.gguf", meta, []))
    assert (info.attention_layers, info.mtp_layers) == (32, 0)
    assert info.key_length == info.value_length == 128
    assert info.kv_bytes(1000, "f16") == 32 * 8 * 256 * 2 * 1000


def test_mtp_detected_from_tensor_names(tmp_path):
    meta = hybrid_meta()
    del meta["qwen35.nextn_predict_layers"]
    info = gguf.inspect(write_gguf(tmp_path / "m.gguf", meta, ["blk.64.nextn.eh_proj.weight"]))
    assert info.mtp_layers == 1


def test_reasoning_defaults_from_template():
    r = gguf.parse_reasoning(TEMPLATE)
    assert r.thinking_default_on is True
    assert r.effort_default == "xhigh"
    assert r.effort_levels == ["xhigh", "medium", "low"]
    assert r.preserve_thinking_default_on is True


def test_not_gguf(tmp_path):
    p = tmp_path / "x.gguf"
    p.write_bytes(b"nope")
    with pytest.raises(gguf.GGUFError):
        gguf.read_header(p)


@pytest.mark.skipif(not QWEN.exists(), reason="real model not downloaded")
def test_real_qwen_header():
    info = gguf.inspect(QWEN)
    assert info.arch == "qwen35" and info.mtp_layers == 1 and info.attention_layers == 16
    assert info.reasoning.effort_default == "xhigh"
    assert info.sampling["top_k"] == 20
