"""Render a run's state.json into results.md and a run script."""

from __future__ import annotations

import shlex
from pathlib import Path

from .pipeline import Run


def _f(x: float | None, nd: int = 2) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def results_md(run: Run) -> str:
    s, out = run.state, []
    env = s.get("env", {})
    m = env.get("machine", {})
    out += [
        f"# {run.p.model.name} on {m.get('chip', '?')}",
        "",
        f"- llama.cpp build {env.get('build')} ({env.get('commit')}), run started {env.get('started')}",
        f"- RAM {m.get('ram_bytes', 0) / 2**30:.0f} GiB, GPU working set {m.get('gpu_working_set_mib')} MiB, "
        f"power {m.get('power', {}).get('source')} {m.get('power', {}).get('percent')}% "
        f"{m.get('power', {}).get('state')}",
        f"- contention at start: {env.get('contention')}",
        f"- repeat noise {_f(s.get('noise_pct'))}% → tie threshold {_f(s.get('tie_pct'))}%",
        "",
    ]
    if s.get("matrix"):
        w = s["winner"]
        out += [
            "## Quant × FA × ubatch (llama-bench, pp2048 / tg128)",
            "",
            "| model | fa | ub | pp t/s | tg t/s | max RSS GB | flags |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in sorted(s["matrix"], key=lambda r: -(r["tg"] or 0)):
            win = r["model"] == w["model"] and r["fa"] == w["fa"] and int(r["ub"]) == int(w["ub"])
            note = (
                ("**winner** " if win else "")
                + ("contended " if r["contended"] else "")
                + ("aborted" if r["aborted"] else "")
            )
            out.append(
                f"| {Path(r['model']).name} | {r['fa']} | {r['ub']} | {_f(r['pp'], 1)} | {_f(r['tg'])} "
                f"| {_f(r['max_rss_gb'])} | {note} |"
            )
        out.append("")
    if s.get("kv"):
        depths = list(next(iter(s["kv"].values())).keys())
        out += [
            "## KV cache type vs context depth (tg t/s)",
            "",
            "| KV | " + " | ".join(depths) + " |",
            "|---|" + "---|" * len(depths),
        ]
        for t, row in s["kv"].items():
            out.append(
                f"| {t}{' **(chosen)**' if t == s.get('kv_choice') else ''} | "
                + " | ".join(_f(row[d]) for d in depths)
                + " |"
            )
        out.append("")
    if s.get("knobs"):
        out += ["## Runtime knobs (tg t/s; adopted only if better than noise)", "", "| knob | tg t/s |", "|---|---|"]
        out += [f"| {k} | {_f(v['tg'])} |" for k, v in s["knobs"].items()]
        out += ["", f"adopted: {', '.join(s.get('knobs_adopted', {})) or 'none'}", ""]
    if s.get("server"):
        base = s["server"].get("off", {}).get("gen_tps")
        best = run.best_server_label()
        out += [
            "## llama-server configs (fixed prompts, seeds, sampling, reasoning)",
            "",
            "| config | gen t/s | vs off | prompt t/s | acceptance | greedy identical | distribution | peak RSS GB |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for lab, r in s["server"].items():
            rel = f"{(r['gen_tps'] / base - 1) * 100:+.1f}%" if base and lab != "off" else "-"
            d = r.get("dist")
            if not d:
                dist = "-"
            elif d.get("method") == "sampled-identity":
                dist = f"pass ({r.get('sampled_identity', 0):.0%} of samples identical)"
            elif d.get("method") == "pooled-sharpening":
                dist = f"{d['verdict']} (p={d['p']:.3f}; null p={d['null_p']:.2f}, positive p={d['positive_p']:.3f})"
            else:
                dist = "pass" if d.get("passed") else "FAIL"
            gi = "-" if lab == "off" else ("yes" if r.get("greedy_identical") else "NO")
            acc = "-" if r.get("acceptance") is None else f"{r['acceptance'] * 100:.0f}%"
            out.append(
                f"| {lab}{' **(best valid)**' if lab == best else ''} | {_f(r['gen_tps'])} | {rel} "
                f"| {_f(r['prompt_tps'], 1)} | {acc} | {gi} | {dist} | {_f(r.get('peak_rss_gb'))} |"
            )
        out.append("")
    if s.get("memory"):
        out += ["## Prompt cache (--cache-ram) and multi-turn reuse", ""]
        for cr, r in s["memory"].items():
            g = [x for x in r["growth_mb"] if x]
            turns = ", ".join(
                f"{t['step']} reused {t['reused']}/{t['prompt_tokens']} ({t['prefill_s']:.1f}s)" for t in r["turns"]
            )
            out.append(
                f"- cache-ram {cr}: footprint {g[0]:.0f} → {g[-1]:.0f} MB over {len(g) - 1} conversations; {turns}"
            )
        out += ["", f"chosen: --cache-ram {s.get('cache_ram_choice')}", ""]
    return "\n".join(out)


def run_script(run: Run) -> str:
    s = run.state
    argv = run._server_argv(cache_ram=s.get("cache_ram_choice", 0)) + run.best_server_args()
    port_i = argv.index("--port") + 1
    argv[port_i] = str(run.p.serve.port)
    return (
        "#!/bin/zsh\n"
        f"# Generated by lmtune from {run.dir} (llama.cpp build {s.get('env', {}).get('build')}).\n"
        "# Reasoning settings are the model template's defaults; sampling is fixed by the profile.\n"
        "set -euo pipefail\n"
        f'exec llama-server {shlex.join(argv)} "$@"\n'
    )


def write(run: Run) -> tuple[Path, Path]:
    md = run.dir / "results.md"
    md.write_text(results_md(run))
    sh = run.dir / f"run-{Path(run.state['winner']['model']).stem}.sh"
    sh.write_text(run_script(run))
    sh.chmod(0o755)
    return md, sh
