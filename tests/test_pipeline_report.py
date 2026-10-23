from pathlib import Path

from lmtune import pipeline, profile, report

ROOT = Path(__file__).parents[1]


def make_run(tmp_path, state):
    r = pipeline.Run(
        profile.load_profile(ROOT / "profiles/qwen3.8-27b.toml"), profile.Constraints(), tmp_path, log=lambda *_: None
    )
    r.state = state
    return r


STATE = {
    "env": {
        "machine": {
            "chip": "Apple M2 Pro",
            "ram_bytes": 32 * 2**30,
            "gpu_working_set_mib": 25559,
            "power": {"source": "AC", "percent": 100, "state": "charged"},
        },
        "build": 11669,
        "commit": "620d2225c",
        "contention": "none",
        "started": "x",
    },
    "model": {"mtp_layers": 1},
    "noise_pct": 0.4,
    "tie_pct": 0.4,
    "winner": {"model": "/m/Q4_0.gguf", "fa": "on", "ub": 512},
    "matrix": [
        {
            "model": "/m/Q4_0.gguf",
            "fa": "on",
            "ub": 512,
            "pp": 96.5,
            "pp_sd": 0,
            "tg": 9.89,
            "tg_sd": 0,
            "max_rss_gb": 16.4,
            "peak_footprint_gb": 0.5,
            "contended": None,
            "ok": True,
            "aborted": None,
        }
    ],
    "kv": {"f16": {"0": 9.88, "8192": 9.42}, "q8_0": {"0": 9.72, "8192": 7.52}},
    "kv_choice": "f16",
    "server": {
        "off": {"args": [], "gen_tps": 9.83, "prompt_tps": 81, "acceptance": None, "peak_rss_gb": 20},
        "mtp2": {
            "args": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"],
            "gen_tps": 12.6,
            "prompt_tps": 79,
            "acceptance": 0.51,
            "greedy_identical": True,
            "sampled_identity": 11 / 12,
            "dist": {"method": "sampled-identity", "passed": True},
        },
        "mtp2-prob": {
            "args": ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2", "--spec-draft-sampling", "probabilistic"],
            "gen_tps": 14.8,
            "prompt_tps": 79,
            "acceptance": 0.65,
            "greedy_identical": True,
            "sampled_identity": 0.0,
            "dist": {
                "method": "pooled-sharpening",
                "S": 0.05,
                "p": 0.001,
                "null_p": 0.4,
                "positive_p": 0.001,
                "controls_ok": True,
                "passed": False,
                "verdict": "FAIL (distribution differs)",
            },
        },
        "mtp3": {
            "args": [],
            "gen_tps": 20.0,
            "prompt_tps": 79,
            "acceptance": 0.5,
            "greedy_identical": False,
            "dist": {"method": "sampled-identity", "passed": True},
        },
    },
    "cache_ram_choice": 2048,
}


def test_best_valid_config_requires_both_gates(tmp_path):
    r = make_run(tmp_path, STATE)
    # mtp2-prob is faster but fails the distribution test; mtp3 is fastest but not greedy-identical
    assert r.best_server_label() == "mtp2"


def test_server_configs_respect_banned_n_max(tmp_path):
    r = make_run(tmp_path, STATE)
    labels = [c.label for c in r.server_configs()]
    assert labels[0] == "off" and "mtp1" in labels and "mtp2-prob" in labels and "mtp2-pmin0.5" in labels
    assert not any(lab.startswith("mtp3") for lab in labels)


def test_report_and_script(tmp_path):
    r = make_run(tmp_path, STATE)
    md = report.results_md(r)
    assert (
        "**(best valid)**" in md
        and "FAIL (distribution differs)" in md
        and "92% of samples identical" in md
        and "| f16 **(chosen)** | 9.88 | 9.42 |" in md
    )
    sh = report.run_script(r)
    assert "--spec-draft-n-max 2" in sh and "probabilistic" not in sh
    assert "-ctk f16" in sh and "--cache-ram 2048" in sh and "--port 8080" in sh
    assert "--cors-origins localhost" in sh and "--host 127.0.0.1" in sh
