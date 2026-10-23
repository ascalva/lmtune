"""llama-server lifecycle + workload client. Timings come from the server itself."""

from __future__ import annotations

import contextlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from . import flags, machine
from ._types import JSON, Log
from .monitor import Monitor, footprint_mb


@dataclass
class ServerConfig:
    label: str
    args: list[str] = field(default_factory=list)  # variant flags (spec decoding, KV type, ...)


def base_args(
    model: Path,
    *,
    ub: int,
    ctx: int,
    parallel: int,
    kv: str,
    cache_ram: int,
    sampling: JSON,
    host: str,
    port: int,
    extra: list[str],
) -> list[str]:
    s = {"temp": 0.8, "top_p": 0.95, "top_k": 40, "min_p": 0.05} | sampling
    return [
        "-m",
        str(model),
        "-ngl",
        "99",
        "-fa",
        "on",
        "-ub",
        str(ub),
        "-np",
        str(parallel),
        "-c",
        str(ctx),
        "-ctk",
        kv,
        "-ctv",
        kv,
        "--cache-ram",
        str(cache_ram),
        "--temp",
        str(s["temp"]),
        "--top-p",
        str(s["top_p"]),
        "--top-k",
        str(s["top_k"]),
        "--min-p",
        str(s["min_p"]),
        "--host",
        host,
        "--port",
        str(port),
        *extra,
    ]


@dataclass
class Running:
    url: str
    proc: subprocess.Popen[bytes]
    monitor: Monitor
    contended_at_start: str | None


@contextlib.contextmanager
def serve(argv: list[str], log_path: Path, swap_limit_mb: float = 512, startup_s: int = 300) -> Iterator[Running]:
    """Start llama-server, wait for /health, monitor it, always stop it."""
    flags.discover("llama-server").require(argv)
    host = argv[argv.index("--host") + 1]
    port = argv[argv.index("--port") + 1]
    if host not in ("127.0.0.1", "::1", "localhost"):
        raise ValueError("refusing to bind a non-loopback host")
    url = f"http://{host}:{port}"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(["llama-server", *argv], stdout=logf, stderr=subprocess.STDOUT)
    try:
        deadline = time.time() + startup_s
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited during startup (see {log_path})")
            with contextlib.suppress(OSError), urllib.request.urlopen(url + "/health", timeout=1) as r:
                if json.load(r).get("status") == "ok":
                    break
            time.sleep(1)
        else:
            raise TimeoutError(f"llama-server not healthy after {startup_s}s")
        cont = machine.contention(ignore_pids={proc.pid})
        with Monitor(proc.pid, swap_limit_mb=swap_limit_mb) as mon:
            yield Running(url, proc, mon, None if cont.clean else cont.describe())
    finally:
        proc.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=30)
        if proc.poll() is None:
            proc.kill()


def chat(
    url: str,
    messages: list[JSON],
    *,
    max_tokens: int,
    seed: int | None = None,
    greedy: bool = False,
    cache_prompt: bool = False,
    timeout: int = 1800,
    api_key: str | None = None,
) -> JSON:
    body: JSON = {"messages": messages, "max_tokens": max_tokens, "cache_prompt": cache_prompt, "stream": False}
    if seed is not None:
        body["seed"] = seed
    if greedy:
        body["top_k"] = 1
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        res = json.load(r)
    msg = res["choices"][0]["message"]
    return {
        "timings": res.get("timings", {}),
        "finish": res["choices"][0].get("finish_reason"),
        "reasoning": msg.get("reasoning_content") or "",
        "content": msg.get("content") or "",
    }


@dataclass
class Generation:
    label: str
    prompt: str
    seed: int
    greedy: bool
    timings: JSON
    finish: str | None
    text: str  # reasoning + answer, for equivalence/distribution checks


def run_workload(
    url: str,
    label: str,
    prompts: list[JSON],
    seeds: list[int],
    max_tokens: int,
    *,
    greedy: bool = False,
    log: Log = print,
    api_key: str | None = None,
) -> list[Generation]:
    chat(url, [{"role": "user", "content": "Say OK."}], max_tokens=8, api_key=api_key)  # warmup
    out = []
    for seed in seeds:
        for p in prompts:
            r = chat(url, p["messages"], max_tokens=max_tokens, seed=seed, greedy=greedy, api_key=api_key)
            g = Generation(
                label,
                p["id"],
                seed,
                greedy,
                r["timings"],
                r["finish"],
                "=== reasoning ===\n" + r["reasoning"] + "\n=== content ===\n" + r["content"] + "\n",
            )
            t = g.timings
            acc = f" acc {t['draft_n_accepted']}/{t['draft_n']}" if t.get("draft_n") else ""
            log(
                f"  [{label}{'-greedy' if greedy else ''}] {p['id']:10s} seed={seed} "
                f"tg={t.get('predicted_per_second', 0):.2f}{acc}"
            )
            out.append(g)
    return out


def aggregate(gens: list[Generation]) -> JSON:
    """Token-weighted rates (total tokens / total time), not a mean of per-request rates."""
    t = [g.timings for g in gens]
    gn, gms = sum(x["predicted_n"] for x in t), sum(x["predicted_ms"] for x in t)
    pn, pms = sum(x["prompt_n"] for x in t), sum(x["prompt_ms"] for x in t)
    dn, da = sum(x.get("draft_n", 0) for x in t), sum(x.get("draft_n_accepted", 0) for x in t)
    per = [x["predicted_per_second"] for x in t]
    return {
        "requests": len(t),
        "gen_tps": gn / gms * 1000 if gms else 0,
        "prompt_tps": pn / pms * 1000 if pms else 0,
        "acceptance": da / dn if dn else None,
        "gen_tps_min": min(per, default=0),
        "gen_tps_max": max(per, default=0),
    }


def cache_growth(url: str, pid: int, n: int = 8) -> list[float | None]:
    """Server footprint (MB) after each of n distinct short conversations (prompt-cache growth)."""
    out = [footprint_mb(pid)]
    for i in range(n):
        chat(
            url,
            [{"role": "user", "content": f"Conversation {i}: list three facts about {i * 37 + 11}."}],
            max_tokens=64,
            cache_prompt=True,
        )
        out.append(footprint_mb(pid))
    return out


def turn_reuse(url: str, document: str) -> list[JSON]:
    """A1 -> A2 (same conversation) -> B1 (other) -> A3: tokens reused and prefill seconds per step."""

    def step(tag: str, messages: list[JSON]) -> tuple[str, JSON, JSON]:
        r = chat(url, messages, max_tokens=96, seed=1, cache_prompt=True)
        t = r["timings"]
        return tag, t, {"role": "assistant", "content": r["content"], "reasoning_content": r["reasoning"]}

    A = [{"role": "user", "content": "Here is a document:\n\n" + document + "\n\nSummarize its first section."}]
    B = [{"role": "user", "content": "Write a haiku about autumn."}]
    rows: list[tuple[str, JSON]] = []
    tag, t, a1 = step("A1", A)
    rows.append((tag, t))
    A += [a1, {"role": "user", "content": "Now the second section."}]
    tag, t, a2 = step("A2", A)
    rows.append((tag, t))
    A += [a2, {"role": "user", "content": "And the third."}]
    tag, t, _ = step("B1", B)
    rows.append((tag, t))
    tag, t, _ = step("A3", A)
    rows.append((tag, t))
    return [
        {
            "step": s,
            "prompt_tokens": t["prompt_n"] + t.get("cache_n", 0),
            "reused": t.get("cache_n", 0),
            "prefill_s": t["prompt_ms"] / 1000,
        }
        for s, t in rows
    ]
