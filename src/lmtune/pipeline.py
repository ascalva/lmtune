"""The tuning pipeline: stages share a run directory and a state.json of decisions."""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from . import bench, checks, exact, flags, gguf, machine, server
from ._types import JSON, Log, Samples
from .profile import Constraints, Profile

STAGES = ["preflight", "matrix", "noise", "kv", "knobs", "server", "dist", "memory", "report"]


class Run:
    def __init__(self, profile: Profile, constraints: Constraints, run_dir: Path, log: Log = print) -> None:
        self.p, self.c, self.dir, self.log = profile, constraints, run_dir, log
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.state: JSON = json.loads(self.state_path.read_text()) if self.state_path.exists() else {}

    def save(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=1, default=str))

    @property
    def swap_limit(self) -> float:
        return float(self.c.swap_growth_limit_mb)

    # -- stages ---------------------------------------------------------------------------

    def preflight(self, allow_contention: bool = False) -> None:
        m = machine.detect(str(self.p.model.dir))
        tools = {t: flags.discover(t) for t in ("llama-server", "llama-bench", "llama-perplexity")}
        cont = machine.contention()
        gpu_idle = machine.gpu_idle_utilization()
        if gpu_idle and gpu_idle > 5:
            self.log(
                f"WARNING: GPU is {gpu_idle:.0f}% busy before we start (desktop/video/other apps); "
                "absolute numbers will drift, keep comparisons back to back"
            )
        self.state["env"] = {
            "machine": m.to_dict(),
            "gpu_idle_pct": gpu_idle,
            "build": tools["llama-server"].build,
            "commit": tools["llama-server"].commit,
            "contention": cont.describe(),
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        models = [self.p.model.dir / f for f in self.p.model.candidates if (self.p.model.dir / f).exists()]
        if not models:
            raise SystemExit("no candidate models on disk; run `lmtune plan` / `lmtune fetch` first")
        info = gguf.inspect(models[0])
        self.state["model"] = {
            "files": [str(x) for x in models],
            "arch": info.arch,
            "mtp_layers": info.mtp_layers,
            "attention_layers": info.attention_layers,
            "kv_mib_per_1k_tokens": {t: info.kv_bytes(1024, t) / 2**20 for t in ("f16", "q8_0")},
            "reasoning_defaults": dataclasses.asdict(info.reasoning),
        }
        self.save()
        if m.power.low_power_mode:
            self.log("WARNING: Low Power Mode is on; results will not reflect normal performance")
        if not cont.clean:
            msg = f"other LLM processes are running: {cont.describe()}"
            if not allow_contention:
                raise SystemExit(msg + "\nstop them for clean numbers, or pass --allow-contention")
            self.log("WARNING: " + msg)

    def matrix(self) -> None:
        s = self.p.search
        res = bench.matrix(
            [Path(f) for f in self.state["model"]["files"]],
            s.fa,
            s.ubatch,
            self.dir / "matrix",
            s.reps,
            swap_limit_mb=self.swap_limit,
            log=self.log,
        )
        self.state["matrix"] = [_bench_row(r) for r in res]
        w = bench.pick_winner(res, tie_pct=self.state.get("tie_pct", 0.5))
        self.state["winner"] = {"model": str(w.config.model), "fa": w.config.fa, "ub": w.config.ub}
        self.save()

    def noise(self) -> None:
        w = self._winner_cfg()
        pct = bench.noise(w, self.dir / "noise", swap_limit_mb=self.swap_limit, log=self.log)
        self.state["noise_pct"] = pct
        self.state["tie_pct"] = max(pct, 0.3)
        self.log(f"repeat noise {pct:.2f}% -> tie threshold {self.state['tie_pct']:.2f}%")
        # re-pick with the measured threshold from the stored matrix rows
        rows = [r for r in self.state["matrix"] if r["ok"] and not r["contended"] and r["tg"] and r["pp"]]
        best = max(r["tg"] for r in rows)
        tied = [r for r in rows if r["tg"] >= best * (1 - self.state["tie_pct"] / 100)]
        w = max(tied, key=lambda r: r["pp"])
        self.state["winner"] = {"model": w["model"], "fa": w["fa"], "ub": w["ub"]}
        self.save()

    def kv(self) -> None:
        res = bench.kv_depth(
            self._winner_cfg(),
            self.p.search.kv_types,
            self.p.search.depths,
            self.dir / "kv",
            swap_limit_mb=self.swap_limit,
            log=self.log,
        )
        deep = max(self.p.search.depths)
        rows = {
            r.config.ctk: {str(d): (r.ts("tg", d) or (None,))[0] for d in self.p.search.depths} for r in res if r.ok
        }
        self.state["kv"] = rows
        self.state["kv_choice"] = max(rows, key=lambda t: rows[t][str(deep)] or 0)
        self.log(f"KV choice: {self.state['kv_choice']} (best tg at depth {deep})")
        self.save()

    def knobs(self) -> None:
        res = bench.knobs(self._winner_cfg(kv=True), self.dir / "knobs", swap_limit_mb=self.swap_limit, log=self.log)
        base = bench.rate(res["baseline"], "tg")
        tie = self.state.get("tie_pct", 0.5)
        self.state["knobs"] = {
            k: {"tg": bench.rate(r, "tg") if r.ts("tg") else None, "error": None if r.ok else "failed"}
            for k, r in res.items()
        }
        wins = [
            k
            for k, r in res.items()
            if k != "baseline" and r.ok and r.ts("tg") and bench.rate(r, "tg") > base * (1 + tie / 100)
        ]
        self.state["knobs_adopted"] = {k: bench.KNOBS[k] for k in wins}
        self.save()

    def server_configs(self) -> list[server.ServerConfig]:
        cfgs = [server.ServerConfig("off")]
        if self.state["model"]["mtp_layers"]:
            for n in self.p.search.allowed_n_max():
                for samp in self.p.search.draft_sampling:
                    lab = f"mtp{n}" + ("-prob" if samp == "probabilistic" else "")
                    cfgs.append(
                        server.ServerConfig(
                            lab,
                            ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n), "--spec-draft-sampling", samp],
                        )
                    )
                for pm in self.p.search.draft_p_min:
                    cfgs.append(
                        server.ServerConfig(
                            f"mtp{n}-pmin{pm:g}",
                            ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(n), "--spec-draft-p-min", str(pm)],
                        )
                    )
        return cfgs

    def server(self) -> None:
        prompts = json.loads(self.p.workload.prompts.read_text())
        results = self.state.setdefault("server", {})
        for cfg in self.server_configs():
            if cfg.label in results:
                continue
            argv = self._server_argv(cache_ram=min(self.p.search.cache_ram_mib or [0])) + cfg.args
            self.log(f"server {cfg.label}: {' '.join(cfg.args) or '(no speculation)'}")
            try:
                with server.serve(argv, self.dir / "server" / f"{cfg.label}.log", self.c.swap_growth_limit_mb) as srv:
                    mon = srv.monitor
                    gens = server.run_workload(
                        srv.url, cfg.label, prompts, self.p.workload.seeds, self.p.workload.max_tokens, log=self.log
                    )
                    greedy = server.run_workload(
                        srv.url,
                        cfg.label,
                        prompts,
                        [self.p.workload.seeds[0]],
                        self.p.workload.greedy_tokens,
                        greedy=True,
                        log=self.log,
                    )
                    tripped = mon.tripped
            except (OSError, RuntimeError, TimeoutError) as e:  # URLError is an OSError
                reason = (mon.tripped if "mon" in locals() and mon.tripped else None) or f"{type(e).__name__}: {e}"
                self.log(f"  ABORTED {cfg.label}: {reason}")
                results[cfg.label] = {"args": cfg.args, "aborted": reason, "gen_tps": 0}
                self.save()
                continue
            _dump(self.dir / "server" / f"{cfg.label}.gens.jsonl", gens + greedy)
            results[cfg.label] = {
                "args": cfg.args,
                **server.aggregate(gens),
                "greedy_tps": server.aggregate(greedy)["gen_tps"],
                "peak_rss_gb": (mon.peak_rss_mb or 0) / 1e3,
                "peak_footprint_gb": (mon.peak_footprint_mb or 0) / 1e3,
                "swap_delta_mb": mon.swap_delta_mb,
                "aborted": tripped,
                "contended": srv.contended_at_start,
            }
            self.save()
        if results.get("off", {}).get("aborted"):
            raise SystemExit("baseline 'off' config aborted; no reference for the correctness gates")
        ref = _load(self.dir / "server" / "off.gens.jsonl")
        for label in results:
            if label == "off" or results[label].get("aborted"):
                continue
            rows = checks.greedy_equivalence(
                [g for g in ref if g.greedy],
                [g for g in _load(self.dir / "server" / f"{label}.gens.jsonl") if g.greedy],
            )
            results[label]["greedy_identical"] = all(r.identical for r in rows) and bool(rows)
            results[label]["greedy_rows"] = [dataclasses.asdict(r) for r in rows]
        self.save()

    def dist(self, n: int = 500, identity_pass: float = 0.75) -> None:
        """Sampled-output correctness gate for every speculative config.

        1. sampled identity vs 'off' (same seeds): >= identity_pass -> pass.
        2. otherwise the pooled sharpening test (exact.py) on short continuations, with a null
           control (off vs off, disjoint seeds) that must pass and a positive control
           (temperature x0.9) that must fail; a broken control makes the result inconclusive.
        The NLL comparison is recorded for information only.
        """
        model = Path(self.state["winner"]["model"])
        ref = [g for g in _load(self.dir / "server" / "off.gens.jsonl") if not g.greedy]
        need = []
        for lab, r in self.state["server"].items():
            if lab == "off" or r.get("aborted"):
                continue
            gens = [g for g in _load(self.dir / "server" / f"{lab}.gens.jsonl") if not g.greedy]
            ident = checks.sampled_identity(ref, gens)
            r["sampled_identity"] = ident
            if ident >= identity_pass:
                r["dist"] = {"method": "sampled-identity", "passed": True}
            else:
                need.append(lab)
            for d in checks.nll_distribution(model, ref, {lab: gens}, log=self.log):
                r["nll"] = dataclasses.asdict(d) | {"ci95": d.ci95}
            self.save()
        if not need:
            return
        samp = {"temp": 0.8, "top_p": 0.95, "top_k": 40, "min_p": 0.05} | self.p.workload.sampling
        base = self._server_argv(cache_ram=0)
        base[base.index("-c") + 1] = "8192"
        out = self.dir / "exact"
        out.mkdir(exist_ok=True)

        def collect(label: str, args: list[str], seeds: range, sampling: JSON) -> Samples:
            f = out / f"{label}.json"
            if f.exists():
                return {int(k): v for k, v in json.loads(f.read_text()).items()}
            with server.serve(base + args, out / f"{label}.log", self.c.swap_growth_limit_mb) as srv:
                data = exact.collect(srv.url, seeds, 4, sampling, log=self.log)
            f.write_text(json.dumps(data))
            return data

        A = collect("ref", [], range(1, n + 1), samp)
        B = collect("null", [], range(10_001, 10_001 + n), samp)
        C = collect("positive", [], range(20_001, 20_001 + n), samp | {"temp": samp["temp"] * 0.9})
        # history-conditioned cells: marginals at drafted positions mix histories, and sharpening can
        # flatten a mixture (Qwen run: the marginal test missed a T=0.9 positive control, p=0.07)
        _, p_null, _ = exact.pooled_conditional_sharpening(A, B)
        _, p_pos, _ = exact.pooled_conditional_sharpening(A, C)
        controls_ok = p_null >= 0.05 and p_pos < 0.05
        for lab in need:
            P = collect(lab, self.state["server"][lab]["args"], range(1, n + 1), samp)
            S, p, _ = exact.pooled_conditional_sharpening(A, P)
            self.state["server"][lab]["dist"] = {
                "method": "pooled-sharpening",
                "S": S,
                "p": p,
                "null_p": p_null,
                "positive_p": p_pos,
                "controls_ok": controls_ok,
                "passed": controls_ok and p >= 0.05,
                "verdict": (
                    "inconclusive (controls failed)"
                    if not controls_ok
                    else "pass"
                    if p >= 0.05
                    else "FAIL (distribution differs)"
                ),
            }
            self.save()

    def memory(self) -> None:
        doc = Path(__file__).parent / "data" / "turn_doc.txt"
        text = doc.read_text() if doc.exists() else ("The quick brown fox. " * 1500)
        out = {}
        for cr in self.p.search.cache_ram_mib:
            argv = self._server_argv(cache_ram=cr) + self.best_server_args()
            with server.serve(argv, self.dir / "memory" / f"cache{cr}.log", self.c.swap_growth_limit_mb) as srv:
                out[str(cr)] = {
                    "growth_mb": server.cache_growth(srv.url, srv.proc.pid),
                    "turns": server.turn_reuse(srv.url, text),
                }
        self.state["memory"] = out
        budget_mb = (self.state["env"]["machine"]["ram_bytes"] / 2**20) - self.c.memory_headroom_gb * 1024
        peak = max((r.get("peak_rss_gb", 0) for r in self.state["server"].values()), default=0) * 1e3
        fits = [cr for cr in self.p.search.cache_ram_mib if peak + cr <= budget_mb]
        self.state["cache_ram_choice"] = max(fits) if fits else 0
        self.save()

    # -- helpers --------------------------------------------------------------------------

    def best_server_label(self) -> str:
        srv = self.state["server"]
        ok = {
            k: v
            for k, v in srv.items()
            if not v.get("aborted")
            and not v.get("contended")
            and (k == "off" or (v.get("greedy_identical") and v.get("dist", {}).get("passed", False)))
        }
        return str(max(ok, key=lambda k: float(ok[k]["gen_tps"])))

    def best_server_args(self) -> list[str]:
        if not self.state.get("server"):
            return []
        return [str(a) for a in self.state["server"][self.best_server_label()]["args"]]

    def _winner_cfg(self, kv: bool = False) -> bench.BenchConfig:
        w = self.state["winner"]
        t = self.state.get("kv_choice", "f16") if kv else "f16"
        return bench.BenchConfig(Path(w["model"]), w["fa"], int(w["ub"]), t, t, reps=self.p.search.reps)

    def _server_argv(self, cache_ram: int) -> list[str]:
        w, sv = self.state["winner"], self.p.serve
        extra = list(sv.extra) + [a for v in self.state.get("knobs_adopted", {}).values() for a in v]
        return server.base_args(
            Path(w["model"]),
            ub=int(w["ub"]),
            ctx=sv.ctx,
            parallel=sv.parallel,
            kv=self.state.get("kv_choice", "f16"),
            cache_ram=cache_ram,
            sampling=self.p.workload.sampling,
            host=sv.host,
            port=sv.port + 9,
            extra=extra,
        )


def _bench_row(r: bench.BenchResult) -> JSON:
    pp, tg = r.ts("pp"), r.ts("tg")
    return {
        "model": str(r.config.model),
        "fa": r.config.fa,
        "ub": r.config.ub,
        "pp": pp and pp[0],
        "pp_sd": pp and pp[1],
        "tg": tg and tg[0],
        "tg_sd": tg and tg[1],
        "max_rss_gb": r.max_rss_gb,
        "peak_footprint_gb": r.peak_footprint_gb,
        "contended": r.contended,
        "ok": r.ok,
        "aborted": r.aborted,
    }


def _dump(path: Path, gens: list[server.Generation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dataclasses.asdict(g)) + "\n" for g in gens))


def _load(path: Path) -> list[server.Generation]:
    return [server.Generation(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
