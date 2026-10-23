import random

from lmtune import exact


def draw(rng, weights, n):
    cats = list(range(len(weights)))
    return rng.choices(cats, weights=weights, k=n)


def test_same_distribution_passes():
    rng = random.Random(1)
    w = [30, 20, 15, 10, 10, 5, 5, 3, 1, 1]
    ps = [exact.permutation_p(draw(rng, w, 300), draw(rng, w, 300), iters=500, rng=random.Random(i)) for i in range(10)]
    assert min(ps) > 0.01  # 10 null comparisons; none should look significant


def _cells(rng, sharpen_t, n=300, cells=16):
    """16 synthetic (prefix, position) cells; candidate optionally sharpened like temperature t."""
    shapes = [[rng.paretovariate(1.2) for _ in range(rng.randint(5, 30))] for _ in range(cells)]

    def sharp(w):
        v = [x ** (1 / sharpen_t) for x in w]
        return [x / sum(v) for x in v]

    ref = {c: [[x] for x in rng.choices(range(len(w)), weights=w, k=n)] for c, w in enumerate(shapes)}
    cand = {c: [[x] for x in rng.choices(range(len(w)), weights=sharp(w), k=n)] for c, w in enumerate(shapes)}
    return ref, cand


def test_pooled_detects_systematic_sharpening():
    # per-cell tests miss this (power ~2-10% after Bonferroni); pooling across cells catches it
    ref, cand = _cells(random.Random(5), 0.9)
    S, p = exact.pooled_sharpening(ref, cand, range(1), iters=500)
    assert S > 0 and p < 0.01


def test_pooled_null_passes():
    ref, cand = _cells(random.Random(6), 1.0)
    _, p = exact.pooled_sharpening(ref, cand, range(1), iters=500)
    assert p > 0.05


def test_collision_null_passes():
    rng = random.Random(3)
    w = [30, 20, 15, 10, 10, 5, 5, 3, 1, 1]
    ps = [
        exact.permutation_p(
            draw(rng, w, 300), draw(rng, w, 300), iters=500, rng=random.Random(i), stat=exact.collision_diff
        )
        for i in range(10)
    ]
    assert min(ps) > 0.01


def test_rare_categories_pooled():
    assert exact.chi2([1] * 50 + [2], [1] * 50 + [3]) < 1.0


def test_verdict_bonferroni():
    tests = [exact.PositionTest(0, i, p) for i, p in enumerate([0.02, 0.5, 0.6, 0.9])]
    assert exact.verdict(tests)[0] is True  # 0.02 > 0.05/4 is false? 0.02 > 0.0125 -> passes
    tests[0].p = 0.001
    assert exact.verdict(tests)[0] is False


def test_conditional_cells_group_by_history():
    ref = {0: [[1, 7], [1, 7], [1, 8], [2, 9]] * 20}
    cand = {0: [[1, 7], [1, 8], [1, 8], [2, 9]] * 20}
    cells = exact.conditional_cells(ref, cand, max_pos=1, min_group=20)
    # position 0 (history = prompt), position 1 given token 1, position 1 given token 2
    assert len(cells) == 3
    given1 = next((a, b) for a, b in cells if set(a) == {7, 8})
    assert given1[0].count(7) == 40 and given1[1].count(7) == 20


def test_conditional_detects_sharpening_hidden_by_mixture():
    # sharpen each conditional at position 1; the marginal mixture barely changes
    rng = random.Random(9)

    def gen(t, n=600):
        out = []
        for _ in range(n):
            h = rng.choice([0, 1, 2])
            w = [[50, 30, 20], [20, 30, 50], [33, 33, 34]][h]
            v = [x ** (1 / t) for x in w]
            out.append([h, rng.choices([10 + h * 3 + i for i in range(3)], weights=v)[0]])
        return {0: out}

    ref, cand = gen(1.0), gen(0.6)
    S, p, n = exact.pooled_conditional_sharpening(ref, cand, max_pos=1, min_group=40, iters=500)
    assert n >= 3 and S > 0 and p < 0.05
