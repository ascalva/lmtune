from pathlib import Path

import pytest

from lmtune import bench

FX = Path(__file__).parent / "fixtures"


def load(name):
    return bench.parse_jsonl((FX / name).read_text())


def result(tg, pp, contended=None, aborted=None, name="m"):
    recs = [
        {"n_prompt": 2048, "n_gen": 0, "n_depth": 0, "avg_ts": pp, "stddev_ts": 0.1},
        {"n_prompt": 0, "n_gen": 128, "n_depth": 0, "avg_ts": tg, "stddev_ts": 0.01},
    ]
    return bench.BenchResult(bench.BenchConfig(Path(f"/m/{name}.gguf")), recs, 16.0, 0.5, contended, aborted)


def test_parse_real_output():
    recs = load("bench-q4_0-faon-ub512.jsonl")
    r = bench.BenchResult(bench.BenchConfig(Path("/x.gguf")), recs, None, None, None, None)
    pp, tg = r.ts("pp"), r.ts("tg")
    assert 90 < pp[0] < 100 and 9.5 < tg[0] < 10.5
    assert recs[0]["build_number"] == 11669


def test_time_l():
    rss, fp = bench.parse_time_l((FX / "bench-q4_0-faon-ub512.err").read_text())
    assert 15 < rss < 18 and 0.3 < fp < 1.0


def test_depth_lookup():
    recs = load("bench-depth-q8_0.jsonl")
    r = bench.BenchResult(bench.BenchConfig(Path("/x.gguf")), recs, None, None, None, None)
    assert r.ts("tg", 8192)[0] < r.ts("tg", 0)[0]  # q8_0 KV slows decode at depth


def test_winner_tie_broken_by_pp_and_dirty_results_excluded():
    a = result(9.89, 96.5, name="a")
    b = result(9.87, 97.0, name="b")  # within 0.5%: tie -> higher pp wins
    c = result(9.80, 99.0, name="c")  # outside the tie band
    d = result(12.0, 120.0, contended="ollama models loaded: x", name="d")
    e = result(13.0, 130.0, aborted="swap grew", name="e")
    assert bench.pick_winner([a, b, c, d, e], tie_pct=0.5) is b
    assert bench.pick_winner([a, b, c], tie_pct=0.1) is a


def test_no_clean_results():
    with pytest.raises(RuntimeError):
        bench.pick_winner([result(10, 90, contended="x")], 0.5)


def test_noise_pct():
    assert bench.noise_pct([9.88, 9.89, 9.83, 9.84, 9.89]) == pytest.approx(0.609, abs=0.01)


def test_config_argv_and_tag():
    c = bench.BenchConfig(
        Path("/m/Q4_0.gguf"),
        fa="on",
        ub=512,
        ctk="q8_0",
        ctv="q8_0",
        n_prompt=0,
        depths=[0, 8192],
        extra=["-lm", "none"],
    )
    argv = c.argv()
    assert argv[argv.index("-d") + 1] == "0,8192" and argv[-2:] == ["-lm", "none"]
    assert c.tag() == "Q4_0_faon_ub512_kq8_0_vq8_0_d0-8192_-lmnone"
