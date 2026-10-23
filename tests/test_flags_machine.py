from pathlib import Path

import pytest

from lmtune import flags, machine

HELP = (Path(__file__).parent / "fixtures/llama-server-help-b11669.txt").read_text()

BENCH_HELP = """
test parameters:
  -m, --model <filename>                            (default: models/7B/ggml-model-q4_0.gguf)
  -fa, --flash-attn <on|off|auto>                   (default: auto)
  -ub, --ubatch-size <n>                            (default: 512)
  -lm, --load-mode <auto|none|mmap|mlock|mmap+mlock|dio> (default: auto)
"""


def test_server_help_flags():
    f, args = flags.parse_help(HELP)
    for flag in [
        "-ngl",
        "-fa",
        "--flash-attn",
        "-ctk",
        "--spec-type",
        "--spec-draft-n-max",
        "--spec-draft-sampling",
        "--cache-ram",
        "--cors-origins",
        "--reasoning-preserve",
        "--no-reasoning-preserve",
        "-np",
        "--host",
    ]:
        assert flag in f, flag
    assert "--draft-max" in f  # removed upstream but still listed (as an error) -- presence only
    tf = flags.ToolFlags("llama-server", 11669, "620d2225c", f, args)
    assert "draft-mtp" in tf.choices("--spec-type")
    assert tf.choices("-fa") == ["on", "off", "auto"]
    assert tf.choices("--spec-draft-sampling") == ["greedy", "probabilistic"]


def test_require_reports_unknown_flags():
    f, args = flags.parse_help(HELP)
    tf = flags.ToolFlags("llama-server", 11669, None, f, args)
    tf.require("-ngl 99 -fa on --spec-type draft-mtp --spec-draft-n-max 2 --temp 1.0 --top-p 0.95")
    with pytest.raises(flags.MissingFlags, match="--made-up"):
        tf.require(["-ngl", "99", "--made-up", "1"])
    assert tf.missing("--temp -1 --seed -1") == []  # negative numbers are values, not flags


def test_bench_help_indented():
    f, args = flags.parse_help(BENCH_HELP)
    assert {"-m", "--model", "-fa", "-ub", "-lm"} <= f
    tf = flags.ToolFlags("llama-bench", None, None, f, args)
    assert tf.choices("-lm") == ["auto", "none", "mmap", "mlock", "mmap+mlock", "dio"]


def test_version():
    assert flags.parse_version("version: 0.7.0 (build 11669, commit 620d2225c)") == (11669, "620d2225c")


def test_power_and_swap_parsing():
    p = machine.parse_pmset_batt(
        "Now drawing from 'AC Power'\n -InternalBattery-0 (id=1)\t100%; charged; 0:00 remaining present: true"
    )
    assert (p.source, p.percent, p.state) == ("AC", 100, "charged")
    p = machine.parse_pmset_batt(
        "Now drawing from 'Battery Power'\n -InternalBattery-0 (id=1)\t72%; discharging; 6:45 remaining"
    )
    assert (p.source, p.percent, p.state) == ("Battery", 72, "discharging")
    assert machine.parse_swap_mb("total = 6144.00M  used = 4886.81M  free = 1257.19M  (encrypted)") == 4886.81


def test_gpu_devices():
    d = machine.parse_gpu_devices(
        "Available devices:\n  BLAS: Accelerate (0 MiB, 0 MiB free)\n  MTL0: Apple M2 Pro (25559 MiB, 25558 MiB free)\n"
    )
    assert d == {"BLAS": (0, 0), "MTL0": (25559, 25558)}


def test_contention_scan():
    ps = """
  101 /Applications/Ollama.app/Contents/Resources/llama-server --model /x --port 5
  102 /opt/homebrew/bin/llama-server -m q.gguf --port 8080
  103 /usr/bin/python3 -m mlx_lm.server --model foo
  104 /Applications/LM Studio.app/Contents/MacOS/LM Studio
  105 /bin/zsh -c grep llama-server-help
  106 /opt/homebrew/bin/llama-bench -m q.gguf
"""
    found = machine.scan_processes(ps, ignore_pids={102})
    assert found == [(101, "ollama-runner"), (103, "mlx"), (104, "lm-studio")]


def test_gpu_utilization_parse():
    text = (
        '  | |   "PerformanceStatistics" = '
        '{"Tiler Utilization %"=24,"Device Utilization %"=17,"Renderer Utilization %"=16}'
    )
    assert machine.parse_gpu_utilization(text) == 17
    assert machine.parse_gpu_utilization("nothing") is None
