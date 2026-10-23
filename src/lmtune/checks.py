"""Correctness gates: a faster config is only a winner if it doesn't change what the model writes.

- greedy_equivalence: with top_k=1, an exact speedup (e.g. speculative decoding) must reproduce
  the reference output token for token. Numerics changes (KV precision) legitimately diverge,
  so those are compared against a reference with the same numerics.
- nll_distribution: at temperature > 0, score every sampled output with the target model alone.
  An exact sampler gives the same expected per-token NLL; a biased one shifts it. Compared
  within prompt (stratified Welch) because prompt-to-prompt variance dwarfs the effect.
"""

from __future__ import annotations

import math
import re
import statistics
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ._types import Log
from .server import Generation


@dataclass
class EquivRow:
    prompt: str
    identical: bool
    first_diff: int | None
    length: int


def greedy_equivalence(ref: list[Generation], cand: list[Generation]) -> list[EquivRow]:
    by = {(g.prompt, g.seed): g.text for g in ref}
    rows = []
    for g in cand:
        a = by.get((g.prompt, g.seed))
        if a is None:
            continue
        b = g.text
        n = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
        if n is None and len(a) != len(b):
            n = min(len(a), len(b))
        rows.append(EquivRow(g.prompt, a == b, n, len(a)))
    return rows


def sampled_identity(ref: list[Generation], cand: list[Generation]) -> float:
    """Fraction of sampled (temperature > 0) outputs byte-identical to the reference with the same
    prompt and seed. An exact speedup that consumes randomness like the reference (e.g. greedy-draft
    speculative decoding) reproduces nearly all of them; that is the strongest evidence available.
    Measured: Qwen3.8-27B MTP n=2 reproduced 11/12 1,024-token samples; probabilistic drafting 0/12."""
    by = {(g.prompt, g.seed): g.text for g in ref}
    pairs = [(by[(g.prompt, g.seed)], g.text) for g in cand if (g.prompt, g.seed) in by]
    return sum(a == b for a, b in pairs) / len(pairs) if pairs else 0.0


def nll(model: Path, text: str, ctx: int = 256) -> float | None:
    """Mean per-token NLL (nats) of text under the model, via llama-perplexity."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(text)
    try:
        out = subprocess.run(
            ["llama-perplexity", "-m", str(model), "-ngl", "99", "-fa", "on", "-c", str(ctx), "-f", f.name],
            capture_output=True,
            check=False,
            text=True,
        ).stderr
    finally:
        Path(f.name).unlink(missing_ok=True)
    m = re.search(r"Final estimate: PPL = ([0-9.]+)", out)
    return math.log(float(m.group(1))) if m else None


def stratified(x: dict[str, list[float]], ref: dict[str, list[float]]) -> tuple[float, float]:
    """Mean over prompts of (mean_x - mean_ref) and its t statistic (within-prompt variances)."""
    ps = [p for p in ref if len(x.get(p, [])) > 1 and len(ref[p]) > 1]
    if not ps:
        return float("nan"), float("nan")
    d = sum(statistics.mean(x[p]) - statistics.mean(ref[p]) for p in ps) / len(ps)
    var = sum(statistics.variance(x[p]) / len(x[p]) + statistics.variance(ref[p]) / len(ref[p]) for p in ps)
    se = math.sqrt(var) / len(ps)
    return d, (d / se if se else float("nan"))


@dataclass
class DistResult:
    """Informational only: with ~12 long samples the 95% CI is about +-0.2 nats/token, too wide to
    certify a sampler. Gating uses sampled_identity or the exact.pooled_sharpening test."""

    label: str
    n: int
    mean_nll: float
    delta: float
    t: float

    @property
    def ci95(self) -> tuple[float, float]:
        se = abs(self.delta / self.t) if self.t and not math.isnan(self.t) else float("nan")
        return self.delta - 1.96 * se, self.delta + 1.96 * se

    def passed(self, t_limit: float = 2.5) -> bool:
        return math.isnan(self.t) or abs(self.t) <= t_limit


def nll_distribution(
    model: Path, ref: list[Generation], cands: dict[str, list[Generation]], log: Log = print
) -> list[DistResult]:
    def score(gens: list[Generation]) -> dict[str, list[float]]:
        byp: dict[str, list[float]] = {}
        for g in gens:
            v = nll(model, g.text)
            if v is not None:
                byp.setdefault(g.prompt, []).append(v)
                log(f"  nll {g.label:16s} {g.prompt:10s} s{g.seed} {v:.4f}")
        return byp

    rp = score(ref)
    out = []
    for label, gens in cands.items():
        cp = score(gens)
        d, t = stratified(cp, rp)
        vals = [v for vs in cp.values() for v in vs]
        out.append(DistResult(label, len(vals), statistics.mean(vals) if vals else float("nan"), d, t))
    return out
