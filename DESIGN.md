# lmtune — design

Find the fastest *correct* way to run a given local model on *this* machine, measure every
change, keep only what wins, and emit a run script plus a results report. Re-runnable after
every runtime update, model release, or hardware change.

Origin: hand-tuning Qwen3.8-27B on an M2 Pro / 32 GB with llama.cpp (Oct 2026). The method
and every lesson from that run are encoded below so the tool doesn't have to rediscover them.

## Principles

1. **Measure, don't assume.** No expected speedups or "theoretical ceilings" drive decisions.
   (The bandwidth "ceiling" for a 16 GB model was beaten by 50% once MTP verified several
   tokens per weight pass; q8_0 KV — commonly recommended — turned out 20–32% slower at depth.)
2. **Correctness gates before speed.** A config that changes outputs is rejected, however fast.
   - Greedy (top_k=1) token-equality vs the reference config.
   - Sampled-output distribution check (target-model NLL of outputs) for any change that
     touches sampling (e.g. probabilistic drafting). Greedy equality can't catch sampler bias.
3. **A noisy measurement is not a measurement.** Detect contention (other LLM processes,
   swap growth) and flag or abort; derive the tie threshold from measured repeat noise.
4. **Verify every runtime flag against `--help`** of the installed build before use; record the
   build number with every result. Flags churn.
5. **Respect user guardrails** as data in the profile: max download size without asking, sudo
   never run by the tool, loopback-only servers, swap-growth abort.

## Pipeline (stages)

| stage | what | output |
|---|---|---|
| `preflight` | build no., chip, RAM, Metal working-set limit, disk, power, swap, contention scan | `env.json` |
| `plan` | list repo files (HF API: size + sha256), filter candidates that fit memory budget, estimate disk, flag downloads needing approval | `plan.md` |
| `fetch` | resumable download (retry-all-errors loop), sha256 verify vs HF LFS oid | `hashes.txt` |
| `inspect` | GGUF header: arch, layers, attention layout, KV heads/dims → KV bytes/token, MTP/nextn layers, chat template + reasoning defaults | `model.json` |
| `matrix` | llama-bench: models × FA × ubatch (each config its own process; `time -l` peak mem) | winner |
| `noise` | repeat the winner N times → tie threshold | `noise.json` |
| `depth` / `kv` | winner at depths (0 / 8K / 16K) × KV types | KV choice |
| `knobs` | threads, poll, load mode, mlock … (expect nothing; prove it) | |
| `server` | llama-server configs (spec types × n-max × draft sampling × p-min) on fixed prompts/seeds; timings from the server itself | |
| `equiv` | greedy equality vs baseline per config | pass/fail |
| `dist` | NLL distribution check (stratified by prompt) per sampling-affecting config | pass/fail |
| `memory` | server footprint growth across N conversations (prompt cache), multi-turn prefix reuse, conversation switching cost | cache-ram choice |
| `report` | results.md + run script with the winning, validated settings | |

## Lessons encoded as defaults/checks

- **Contention:** a background scheduler (mind-palace) loaded an 18 GB model into Ollama mid-run
  → swap → garbage numbers. Scan before/after every config: Ollama `/api/ps`, foreign
  `llama-server`, LM Studio, MLX. `keep_alive: 0` via the API *re-spawned* an Ollama model;
  `ollama stop <model>` is the reliable unload.
- **Swap guard:** abort a config if system swap grows > 512 MB over its start value.
- **Prompt cache growth:** llama-server's `--cache-ram` default (8192 MiB) grows ~460 MB per
  conversation for hybrid (recurrent-state) models → swap on 32 GB. Measure growth; pick a
  bounded value. 2048 kept conversation switching instant (0.6 s vs 59 s re-prefill).
- **KV type is a speed knob, not just memory:** f16 vs q8_0 on Metal: −1.6% at 0, −20% at 8K,
  −32% at 16K for q8_0. Mixed K/V types: pathological (±1 tok/s noise). Always sweep at depth.
- **Speculative decoding:** MTP built into the GGUF (`<arch>.nextn_predict_layers`) needs no
  draft file. Sweep n-max (respect user-banned values), greedy vs probabilistic drafting, p-min.
  n-gram drafting only helps greedy, copy-heavy output.
- **Quant choice is empirical:** Q4_0 beat UD-Q4_K_M by 15% and IQ4_XS by 5% on M2 decode,
  despite IQ4_XS being 1.8 GB smaller (dequant cost).
- **Environment drifts by hours:** the same config read 10.3 tok/s at 6 PM and 11 AM but 9.9 from
  7 PM to 4 AM — not thermal, not power, not the wired limit. Never compare numbers taken in
  different sessions; run comparisons back to back (A-B-A when a setting needs the user, e.g.
  sudo), and record idle GPU utilization (desktop compositing on a 5K/144 Hz display spiked it
  to 17–24%).
- **Wired limit:** raising `iogpu.wired_limit_mb` changed nothing when peak use was below the
  default working set (A-B-A: 10.33 vs 10.31). Only test it when a config doesn't fit.
- **Reasoning settings are part of the workload:** read the chat template's defaults
  (e.g. reasoning_effort default, preserve_thinking) and keep them fixed across configs;
  generation caps keep runs comparable.
- **Security:** llama-server allows all CORS origins by default; any web page could call a
  loopback server. Emit `--cors-origins localhost`.

## Inputs

- `machine` (auto-detected): chip, RAM, Metal recommendedMaxWorkingSetSize, disk free.
- `profiles/<model>.toml`: HF repo, candidate files, MTP info, sampling params, reasoning
  settings, context size, banned values (e.g. n-max 3), prompts, generation caps.
- `constraints.toml` (user): download-approval threshold, memory headroom to leave for other
  apps, swap limit, ports/host.

## Backends

llama.cpp first (llama-bench + llama-server). Interface kept narrow so MLX (mlx_lm.server) or
Ollama can be added: `bench(config) -> metrics`, `serve(config) -> url`, `flags() -> set`.

## Non-goals

- Running sudo (the tool prints the command; the user runs it).
- Changing model quality settings to gain speed (sampling/reasoning are fixed inputs).
