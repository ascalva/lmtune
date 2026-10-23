"""llama-bench stages. Every config runs in its own process so peak memory is per config."""

from __future__ import annotations

import json
import re
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from . import flags, machine
from ._types import JSON, Log
from .monitor import Monitor


@dataclass
class BenchConfig:
    model: Path
    fa: str = "on"
    ub: int = 512
    ctk: str = "f16"
    ctv: str = "f16"
    n_prompt: int = 2048
    n_gen: int = 128
    depths: list[int] = field(default_factory=lambda: [0])
    reps: int = 2
    extra: list[str] = field(default_factory=list)

    def argv(self) -> list[str]:
        return [
            "-m",
            str(self.model),
            "-ngl",
            "99",
            "-fa",
            self.fa,
            "-ub",
            str(self.ub),
            "-ctk",
            self.ctk,
            "-ctv",
            self.ctv,
            "-p",
            str(self.n_prompt),
            "-n",
            str(self.n_gen),
            "-d",
            ",".join(map(str, self.depths)),
            "-r",
            str(self.reps),
            "-o",
            "jsonl",
            *self.extra,
        ]

    def tag(self) -> str:
        t = f"{self.model.stem}_fa{self.fa}_ub{self.ub}_k{self.ctk}_v{self.ctv}"
        if self.depths != [0]:
            t += "_d" + "-".join(map(str, self.depths))
        if self.extra:
            t += "_" + re.sub(r"[^A-Za-z0-9.+-]+", "", "".join(self.extra))
        return t


@dataclass
class BenchResult:
    config: BenchConfig
    records: list[JSON]  # raw llama-bench jsonl records
    max_rss_gb: float | None
    peak_footprint_gb: float | None
    contended: str | None  # description of competing processes, if any
    aborted: str | None  # swap guard reason, if tripped

    def ts(self, kind: str, depth: int = 0) -> tuple[float, float] | None:
        """(avg, stddev) tokens/s for 'pp' or 'tg' at a depth."""
        for r in self.records:
            is_pp = r["n_prompt"] > 0 and r["n_gen"] == 0
            if (kind == "pp") == is_pp and r.get("n_depth", 0) == depth:
                return r["avg_ts"], r["stddev_ts"]
        return None

    @property
    def ok(self) -> bool:
        return not self.aborted and bool(self.records)


def parse_time_l(stderr: str) -> tuple[float | None, float | None]:
    """(max RSS GB, peak footprint GB) from macOS `/usr/bin/time -l`."""
    rss = re.search(r"(\d+)\s+maximum resident set size", stderr)
    fp = re.search(r"(\d+)\s+peak memory footprint", stderr)
    return (int(rss.group(1)) / 1e9 if rss else None, int(fp.group(1)) / 1e9 if fp else None)


def parse_jsonl(stdout: str) -> list[JSON]:
    return [json.loads(line) for line in stdout.splitlines() if line.strip().startswith("{")]


def run(cfg: BenchConfig, out_dir: Path, swap_limit_mb: float = 512, log: Log = print) -> BenchResult:
    argv = cfg.argv()
    flags.discover("llama-bench").require(argv)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = cfg.tag()
    before = machine.contention()
    proc = subprocess.Popen(
        ["/usr/bin/time", "-l", "llama-bench", *argv], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    with Monitor(proc.pid, swap_limit_mb=swap_limit_mb) as mon:
        stdout, stderr = proc.communicate()
    after = machine.contention()
    (out_dir / f"{tag}.jsonl").write_text(stdout)
    (out_dir / f"{tag}.err").write_text(stderr)
    rss, fp = parse_time_l(stderr)
    cont = "; ".join(c.describe() for c in (before, after) if not c.clean) or None
    res = BenchResult(cfg, parse_jsonl(stdout), rss, fp, cont, mon.tripped)
    pp, tg = res.ts("pp"), res.ts("tg")
    log(
        f"  {tag}: pp={pp and round(pp[0], 1)} tg={tg and round(tg[0], 2)}"
        + (f"  CONTENDED: {cont}" if cont else "")
        + (f"  ABORTED: {mon.tripped}" if mon.tripped else "")
    )
    return res


def noise_pct(samples: list[float]) -> float:
    """Spread of repeated measurements of one config, as % of the mean (max - min)."""
    return (max(samples) - min(samples)) / statistics.mean(samples) * 100 if len(samples) > 1 else 0.0


def pick_winner(results: list[BenchResult], tie_pct: float) -> BenchResult:
    """Highest tg; anything within tie_pct of the best is a tie, broken by pp. Contended/aborted
    results never win."""
    ok = [r for r in results if r.ok and not r.contended and r.ts("tg") and r.ts("pp")]
    if not ok:
        raise RuntimeError("no clean results to pick from")
    best = max(rate(r, "tg") for r in ok)
    tied = [r for r in ok if rate(r, "tg") >= best * (1 - tie_pct / 100)]
    return max(tied, key=lambda r: rate(r, "pp"))


def rate(r: BenchResult, kind: str) -> float:
    ts = r.ts(kind)
    if ts is None:
        raise ValueError(f"{r.config.tag()} has no {kind} measurement")
    return ts[0]


# --- stages -----------------------------------------------------------------------------


def matrix(
    models: list[Path],
    fa: list[str],
    ubatch: list[int],
    out: Path,
    reps: int,
    *,
    swap_limit_mb: float = 512,
    log: Log = print,
) -> list[BenchResult]:
    return [
        run(BenchConfig(m, fa=f, ub=u, reps=reps), out, swap_limit_mb, log) for m in models for f in fa for u in ubatch
    ]


def noise(cfg: BenchConfig, out: Path, runs: int = 3, *, swap_limit_mb: float = 512, log: Log = print) -> float:
    """Re-run tg-only for the config several times in separate processes; return spread %."""
    tgs: list[float] = []
    for i in range(runs):
        r = run(
            BenchConfig(cfg.model, cfg.fa, cfg.ub, cfg.ctk, cfg.ctv, n_prompt=0, reps=5, extra=[]),
            out / f"noise{i}",
            swap_limit_mb,
            log,
        )
        if r.ts("tg"):
            tgs += [float(x) for x in r.records[0].get("samples_ts", [rate(r, "tg")])]
    return noise_pct(tgs)


def kv_depth(
    cfg: BenchConfig,
    kv_types: list[str],
    depths: list[int],
    out: Path,
    *,
    swap_limit_mb: float = 512,
    log: Log = print,
) -> list[BenchResult]:
    """tg at each depth for each KV type (K and V the same: mixed types measured pathological)."""
    return [
        run(
            BenchConfig(cfg.model, cfg.fa, cfg.ub, t, t, n_prompt=0, depths=depths, reps=cfg.reps),
            out,
            swap_limit_mb,
            log,
        )
        for t in kv_types
    ]


KNOBS = {
    "t1": ["-t", "1"],
    "t4": ["-t", "4"],
    "poll0": ["--poll", "0"],
    "poll100": ["--poll", "100"],
    "lm-none": ["-lm", "none"],
    "lm-mlock": ["-lm", "mmap+mlock"],
}


def knobs(cfg: BenchConfig, out: Path, *, swap_limit_mb: float = 512, log: Log = print) -> dict[str, BenchResult]:
    base = {
        "baseline": run(
            BenchConfig(cfg.model, cfg.fa, cfg.ub, cfg.ctk, cfg.ctv, n_prompt=0, reps=3), out, swap_limit_mb, log
        )
    }
    return base | {
        k: run(
            BenchConfig(cfg.model, cfg.fa, cfg.ub, cfg.ctk, cfg.ctv, n_prompt=0, reps=3, extra=v),
            out,
            swap_limit_mb,
            log,
        )
        for k, v in KNOBS.items()
    }
