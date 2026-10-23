import subprocess
import sys
import time
from pathlib import Path

import pytest

from lmtune import hf, monitor, profile

ROOT = Path(__file__).parents[1]


def test_qwen_profile_loads():
    p = profile.load_profile(ROOT / "profiles/qwen3.8-27b.toml")
    assert p.model.hf_repo == "unsloth/Qwen3.8-27B-GGUF"
    assert p.search.allowed_n_max() == [1, 2]  # 3 is banned
    assert p.workload.prompts.exists()
    assert p.workload.sampling["top_k"] == 20
    assert p.serve.host == "127.0.0.1"


def test_profile_rejects_unknown_keys_and_public_host(tmp_path):
    base = (ROOT / "profiles/qwen3.8-27b.toml").read_text()
    (tmp_path / "a.toml").write_text(base.replace("reps = 2", "reps = 2\nmystery = 1"))
    with pytest.raises(ValueError, match="mystery"):
        profile.load_profile(tmp_path / "a.toml")
    (tmp_path / "b.toml").write_text(base.replace('host = "127.0.0.1"', 'host = "0.0.0.0"'))
    with pytest.raises(ValueError, match="loopback"):
        profile.load_profile(tmp_path / "b.toml")


def test_constraints_default_when_missing(tmp_path):
    c = profile.load_constraints(tmp_path / "none.toml")
    assert c.download_ask_gb == 10 and c.allow_sudo is False


TREE = [
    {"type": "file", "path": "M-Q4_0.gguf", "size": 16_000_000_000, "lfs": {"oid": "aa"}},
    {"type": "file", "path": "M-Q8_0.gguf", "size": 29_000_000_000, "lfs": {"oid": "bb"}},
    {"type": "file", "path": "M-UD-Q4_K_M.gguf", "size": 16_500_000_000, "lfs": {"oid": "cc"}},
    {"type": "directory", "path": "MTP"},
]


def test_plan_memory_budget_disk_and_approval(tmp_path):
    files = hf.parse_tree(TREE)
    items, problems = hf.plan(
        files,
        ["M-Q4_0.gguf", "M-Q8_0.gguf", "M-Q4_K_M.gguf"],
        tmp_path,
        budget_bytes=25e9,
        ask_bytes=10e9,
        disk_free=40_000_000_000,
        min_free=10e9,
        runtime_bytes=4e9,
    )
    by = {i.file.path: i for i in items}
    assert by["M-Q4_0.gguf"].fits_memory and by["M-Q4_0.gguf"].needs_approval
    assert not by["M-Q8_0.gguf"].fits_memory and "budget" in by["M-Q8_0.gguf"].note
    assert any("M-Q4_K_M.gguf: not in repo" in p and "M-UD-Q4_K_M.gguf" in p for p in problems)
    # only the 16 GB file is fetched; 40 - 16 = 24 GB left >= 10 GB
    assert not any("only" in p for p in problems)


def test_plan_flags_insufficient_disk(tmp_path):
    files = hf.parse_tree(TREE)
    _, problems = hf.plan(
        files, ["M-Q4_0.gguf"], tmp_path, budget_bytes=25e9, ask_bytes=10e9, disk_free=20_000_000_000, min_free=10e9
    )
    assert any("must stay free" in p for p in problems)


def test_sha256(tmp_path):
    p = tmp_path / "f"
    p.write_bytes(b"abc")
    assert hf.sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


@pytest.mark.skipif(sys.platform != "darwin", reason="Monitor samples with macOS `footprint`")
def test_monitor_samples_process():
    proc = subprocess.Popen(["sleep", "2"])
    with monitor.Monitor(proc.pid, interval=0.2) as mon:
        proc.wait()
    assert mon.samples and mon.tripped is None
    assert mon.peak_rss_mb is not None


@pytest.mark.skipif(sys.platform != "darwin", reason="Monitor samples with macOS `footprint`")
def test_monitor_swap_guard_kills(monkeypatch):
    vals = iter([1000.0] + [1000.0 + 300 * i for i in range(1, 50)])
    monkeypatch.setattr(monitor, "swap_used_mb", lambda: next(vals))
    proc = subprocess.Popen(["sleep", "30"])
    with monitor.Monitor(proc.pid, swap_limit_mb=512, interval=0.1) as mon:
        proc.wait(timeout=10)
    assert mon.tripped and "swap grew" in mon.tripped
    assert proc.returncode != 0  # SIGTERM'd


@pytest.mark.skipif(sys.platform != "darwin", reason="Monitor samples with macOS `footprint`")
def test_monitor_follows_and_kills_child_of_wrapper(monkeypatch):
    vals = iter([1000.0] + [1000.0 + 300 * i for i in range(1, 50)])
    monkeypatch.setattr(monitor, "swap_used_mb", lambda: next(vals))
    # sh is the wrapper (like /usr/bin/time); sleep is the child that must die too
    proc = subprocess.Popen(["/bin/sh", "-c", "sleep 30; true"])

    time.sleep(0.3)
    child = int(
        subprocess.run(["pgrep", "-n", "-P", str(proc.pid)], capture_output=True, text=True, check=False).stdout
    )
    with monitor.Monitor(proc.pid, swap_limit_mb=512, interval=0.1) as mon:
        proc.wait(timeout=10)
    assert mon.tripped
    time.sleep(0.2)
    assert subprocess.run(["kill", "-0", str(child)], capture_output=True, check=False).returncode != 0
