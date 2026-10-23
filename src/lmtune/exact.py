"""Direct exactness test for samplers that change the random stream (e.g. probabilistic drafting).

Long generations are a weak test: per-sample NLL varies so much between trajectories that a
real bias hides in the noise (12 x 1024-token samples gave a 95% CI of +-0.2 nats/token).
Instead, sample many short continuations from fixed high-entropy prefixes and compare the
token distribution at each position with a permutation chi-square test. Positions after the
first are the ones speculative drafting touches. Calibrate with reference-vs-reference on
disjoint seeds: that comparison must pass, or the test itself is broken.
"""

from __future__ import annotations

import json
import random
import urllib.request
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

from ._types import JSON, Log, Samples

Stat = Callable[[list[int], list[int]], float]

PREFIXES = [
    "My favorite fruits, in no particular order, are",
    "Here is a list of random words: lantern, quarry, velvet,",
    "The three most underrated programming languages are",
    "Some good names for a pet goldfish are",
]
# (replaced "Once upon a time, in a small village by the": near-deterministic, no signal)


def sample(url: str, prompt: str, seed: int, n: int, sampling: JSON) -> list[int]:
    body = {
        "prompt": prompt,
        "n_predict": n,
        "seed": seed,
        "cache_prompt": True,
        "return_tokens": True,
        "temperature": sampling.get("temp", 1.0),
        "top_k": sampling.get("top_k", 40),
        "top_p": sampling.get("top_p", 0.95),
        "min_p": sampling.get("min_p", 0.0),
    }
    req = urllib.request.Request(
        url + "/completion", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        tokens = json.load(r)["tokens"]
    if not isinstance(tokens, list) or not all(isinstance(t, int) for t in tokens):
        raise ValueError("server returned no token ids (needs return_tokens support)")
    return tokens


def collect(
    url: str, seeds: range, n_tokens: int, sampling: JSON, *, prefixes: list[str] = PREFIXES, log: Log = print
) -> Samples:
    """{prefix_index: [[tok, tok, ...], ...]} — one token list per seed."""
    out = {}
    for i, p in enumerate(prefixes):
        out[i] = [sample(url, p, s, n_tokens, sampling) for s in seeds]
        log(f"  prefix {i}: {len(out[i])} samples")
    return out


def chi2(a: list[int], b: list[int], min_expected: float = 5.0) -> float:
    """Chi-square homogeneity statistic for two samples of categories; rare categories pooled."""
    ca, cb = Counter(a), Counter(b)
    na, nb = len(a), len(b)
    n = na + nb
    cats = set(ca) | set(cb)
    big = [c for c in cats if (ca[c] + cb[c]) * min(na, nb) / n >= min_expected]
    rows = [(ca[c], cb[c]) for c in big]
    other = (na - sum(r[0] for r in rows), nb - sum(r[1] for r in rows))
    if sum(other):
        rows.append(other)
    stat = 0.0
    for xa, xb in rows:
        tot = xa + xb
        for obs, nn in ((xa, na), (xb, nb)):
            exp = tot * nn / n
            stat += (obs - exp) ** 2 / exp if exp else 0.0
    return stat


def collision(x: list[int]) -> float:
    """Unbiased estimate of sum(p_i^2): probability two independent draws coincide. A sampler
    that sharpens toward likely tokens (the plausible failure of a biased drafter) raises it."""
    n = len(x)
    return sum(c * (c - 1) for c in Counter(x).values()) / (n * (n - 1)) if n > 1 else 0.0


def collision_diff(a: list[int], b: list[int]) -> float:
    return abs(collision(a) - collision(b))


def permutation_p(
    a: list[int], b: list[int], iters: int = 2000, rng: random.Random | None = None, stat: Stat = chi2
) -> float:
    """Two-sample permutation p-value for any statistic where larger = more different."""
    rng = rng or random.Random(0)
    obs = stat(a, b)
    pool, na = a + b, len(a)
    hits = 0
    for _ in range(iters):
        rng.shuffle(pool)
        if stat(pool[:na], pool[na:]) >= obs:
            hits += 1
    return (hits + 1) / (iters + 1)


@dataclass
class PositionTest:
    prefix: int
    position: int
    p: float
    stat: str = "chi2"
    ref_collision: float | None = None
    cand_collision: float | None = None


def compare(ref: Samples, cand: Samples, positions: range, iters: int = 2000) -> list[PositionTest]:
    """Omnibus chi-square and collision-probability tests at every (prefix, position)."""
    out = []
    for i, ref_samples in ref.items():
        for pos in positions:
            a = [s[pos] for s in ref_samples if len(s) > pos]
            b = [s[pos] for s in cand[i] if len(s) > pos]
            ca, cb = collision(a), collision(b)
            out.append(PositionTest(i, pos, permutation_p(a, b, iters), "chi2", ca, cb))
            out.append(PositionTest(i, pos, permutation_p(a, b, iters, stat=collision_diff), "collision", ca, cb))
    return out


def pooled_sharpening(
    ref: Samples, cand: Samples, positions: range, *, iters: int = 2000, rng: random.Random | None = None
) -> tuple[float, float]:
    """One test across all (prefix, position) cells: S = sum of (collision(cand) - collision(ref)).

    A real sampler bias is systematic (same direction everywhere), so pooling multiplies power by
    ~sqrt(#cells) and needs no multiple-comparison penalty. Labels are permuted within each cell.
    Returns (S, two-sided p). S > 0 means the candidate is sharper (less diverse) than the reference.
    """
    rng = rng or random.Random(0)
    cells = []
    for i, ref_samples in ref.items():
        for pos in positions:
            a = [s[pos] for s in ref_samples if len(s) > pos]
            b = [s[pos] for s in cand[i] if len(s) > pos]
            if len(a) > 1 and len(b) > 1:
                cells.append((a, b))
    obs = sum(collision(b) - collision(a) for a, b in cells)
    hits = 0
    pools = [(a + b, len(a)) for a, b in cells]
    for _ in range(iters):
        tot = 0.0
        for pool, na in pools:
            rng.shuffle(pool)
            tot += collision(pool[na:]) - collision(pool[:na])
        if abs(tot) >= abs(obs):
            hits += 1
    return obs, (hits + 1) / (iters + 1)


def conditional_cells(
    ref: Samples, cand: Samples, *, max_pos: int = 3, min_group: int = 40
) -> list[tuple[list[int], list[int]]]:
    """(ref, cand) samples of the token at position k given an identical history (positions < k).

    Marginals at later positions mix many histories, and sharpening each conditional can flatten
    the mixture, so the sharpening statistic is only monotone within a fixed history. Groups need
    >= min_group samples on both sides. Position 0 is included (its history is the prompt)."""
    cells = []
    for i, ref_samples in ref.items():
        for k in range(max_pos + 1):
            ga: dict[tuple[int, ...], list[int]] = {}
            gb: dict[tuple[int, ...], list[int]] = {}
            for s in ref_samples:
                if len(s) > k:
                    ga.setdefault(tuple(s[:k]), []).append(s[k])
            for s in cand[i]:
                if len(s) > k:
                    gb.setdefault(tuple(s[:k]), []).append(s[k])
            for h, group in ga.items():
                if len(group) >= min_group and len(gb.get(h, [])) >= min_group:
                    cells.append((group, gb[h]))
    return cells


def pooled_conditional_sharpening(
    ref: Samples,
    cand: Samples,
    *,
    max_pos: int = 3,
    min_group: int = 40,
    iters: int = 2000,
    rng: random.Random | None = None,
) -> tuple[float, float, int]:
    """pooled_sharpening over history-conditioned cells. Returns (S, two-sided p, #cells)."""
    rng = rng or random.Random(0)
    cells = conditional_cells(ref, cand, max_pos=max_pos, min_group=min_group)
    obs = sum(collision(b) - collision(a) for a, b in cells)
    pools = [(a + b, len(a)) for a, b in cells]
    hits = 0
    for _ in range(iters):
        tot = 0.0
        for pool, na in pools:
            rng.shuffle(pool)
            tot += collision(pool[na:]) - collision(pool[:na])
        if abs(tot) >= abs(obs):
            hits += 1
    return obs, (hits + 1) / (iters + 1), len(cells)


def verdict(tests: list[PositionTest], alpha: float = 0.05) -> tuple[bool, float]:
    """Pass unless some position is significant after Bonferroni correction."""
    m = min(t.p for t in tests)
    return m > alpha / len(tests), m
