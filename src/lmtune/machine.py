"""Machine facts and anything that would make a measurement dishonest.

`contention()` is the lesson from the first Qwen run: a background scheduler loaded an 18 GB
model into Ollama mid-benchmark and the numbers were garbage. Scan before and after every config.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from collections.abc import Set as AbstractSet
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _sh(*cmd: str, timeout: int = 30) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return r.stdout + r.stderr
    except OSError, subprocess.TimeoutExpired:
        return ""


@dataclass
class Power:
    source: str  # "AC" | "Battery" | "unknown"
    percent: int | None
    state: str | None  # charging | discharging | charged | ...
    low_power_mode: bool | None


def parse_pmset_batt(text: str) -> Power:
    src = "AC" if "AC Power" in text else "Battery" if "Battery Power" in text else "unknown"
    m = re.search(r"(\d+)%;\s*([a-zA-Z ]+?);", text)
    return Power(src, int(m.group(1)) if m else None, m.group(2).strip() if m else None, None)


def power() -> Power:
    p = parse_pmset_batt(_sh("pmset", "-g", "batt"))
    m = re.search(r"lowpowermode\s+(\d)", _sh("pmset", "-g"))
    p.low_power_mode = (m.group(1) == "1") if m else None
    return p


def parse_swap_mb(text: str) -> float | None:
    m = re.search(r"used = ([0-9.]+)M", text)
    return float(m.group(1)) if m else None


def swap_used_mb() -> float | None:
    return parse_swap_mb(_sh("sysctl", "-n", "vm.swapusage"))


def parse_gpu_devices(text: str) -> dict[str, tuple[int, int]]:
    """'MTL0: Apple M2 Pro (25559 MiB, 25558 MiB free)' -> {'MTL0': (25559, 25558)}"""
    return {
        m.group(1): (int(m.group(2)), int(m.group(3)))
        for m in re.finditer(r"^\s*(\w+): .*\((\d+) MiB, (\d+) MiB free\)", text, re.M)
    }


@dataclass
class Machine:
    chip: str
    ram_bytes: int
    gpu_working_set_mib: int | None  # what the GPU backend may allocate (Metal recommendedMaxWorkingSetSize)
    wired_limit_mb: int | None  # iogpu.wired_limit_mb (0 = OS default)
    disk_free_bytes: int
    power: Power

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect(models_dir: str = "~/models", llama_tool: str = "llama-server") -> Machine:
    devs = parse_gpu_devices(_sh(llama_tool, "--list-devices", timeout=120))
    gpu = next((total for name, (total, _) in devs.items() if not name.startswith(("BLAS", "CPU"))), None)
    wl = _sh("sysctl", "-n", "iogpu.wired_limit_mb").strip()
    models = Path(models_dir).expanduser()
    models.mkdir(parents=True, exist_ok=True)
    return Machine(
        chip=_sh("sysctl", "-n", "machdep.cpu.brand_string").strip(),
        ram_bytes=int(_sh("sysctl", "-n", "hw.memsize").strip() or 0),
        gpu_working_set_mib=gpu,
        wired_limit_mb=int(wl) if wl.isdigit() else None,
        disk_free_bytes=shutil.disk_usage(models).free,
        power=power(),
    )


def parse_gpu_utilization(text: str) -> int | None:
    m = re.search(r'"Device Utilization %"=(\d+)', text)
    return int(m.group(1)) if m else None


def gpu_idle_utilization(samples: int = 5, interval: float = 0.5) -> float | None:
    """Mean GPU 'Device Utilization %' while nothing of ours runs (desktop compositing, video,
    other apps). Readable without sudo via IOAccelerator; > ~5% means decode numbers will drift."""
    vals = []
    for _ in range(samples):
        v = parse_gpu_utilization(_sh("ioreg", "-r", "-d", "1", "-c", "IOAccelerator"))
        if v is not None:
            vals.append(v)
        time.sleep(interval)
    return sum(vals) / len(vals) if vals else None


# --- contention -------------------------------------------------------------------------

# command-line patterns of other local-LLM runtimes that compete for memory bandwidth/RAM
COMPETITORS = {
    "llama-server": re.compile(r"(^|/)llama-server(\s|$)"),
    "ollama-runner": re.compile(r"Ollama\.app/Contents/Resources/(llama-server|ollama runner)"),
    "lm-studio": re.compile(r"LM Studio\.app|Bionic\.app|\.lmstudio/"),
    "mlx": re.compile(r"mlx_lm|mlx-lm"),
}


@dataclass
class Contention:
    processes: list[tuple[int, str]] = field(default_factory=list)  # (pid, kind)
    ollama_models: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return not self.processes and not self.ollama_models

    def describe(self) -> str:
        parts = [f"{kind} pid {pid}" for pid, kind in self.processes]
        if self.ollama_models:
            parts.append(
                "ollama models loaded: "
                + ", ".join(self.ollama_models)
                + " (unload with `ollama stop <model>`; keep_alive=0 via the API can respawn it)"
            )
        return "; ".join(parts) or "none"


def scan_processes(ps_text: str, ignore_pids: AbstractSet[int] = frozenset()) -> list[tuple[int, str]]:
    found = []
    for line in ps_text.splitlines():
        m = re.match(r"\s*(\d+)\s+(.*)", line)
        if not m or int(m.group(1)) in ignore_pids:
            continue
        pid, args = int(m.group(1)), m.group(2)
        for kind, rx in COMPETITORS.items():
            if rx.search(args):
                # an Ollama runner also matches the generic llama-server pattern; report it once
                if kind == "llama-server" and COMPETITORS["ollama-runner"].search(args):
                    continue
                found.append((pid, kind))
                break
    return found


def ollama_loaded(url: str = "http://127.0.0.1:11434") -> list[str]:
    try:
        with urllib.request.urlopen(url + "/api/ps", timeout=2) as r:
            return [m.get("name", "?") for m in json.load(r).get("models", [])]
    except OSError, ValueError:
        return []


def contention(ignore_pids: AbstractSet[int] = frozenset()) -> Contention:
    ignore = set(ignore_pids) | {os.getpid()}
    return Contention(scan_processes(_sh("ps", "-axo", "pid=,args="), ignore), ollama_loaded())
