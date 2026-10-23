# lmtune

Find the fastest **correct** way to run a local LLM on *this* machine. Every change is
measured, every speedup must pass correctness gates, and the output is a results report plus a
ready-to-run server script. Re-run it after `brew upgrade llama.cpp`, a new model release, or a
hardware change. Design and lessons learned: [DESIGN.md](DESIGN.md).

Backend today: llama.cpp (`llama-bench`, `llama-server`, `llama-perplexity`) on macOS.

## Serve (production)

```bash
lmtune serve init                 # ~/.config/lmtune/serve.toml (measured winner) + 0600 API key
lmtune serve install              # launchd: starts at login, restarts after a crash, unloads when idle
lmtune serve status | logs -f | restart | stop
lmtune expose on [webui|searxng|dash]  # tailnet-only HTTPS (443: llama.cpp UI, 8443: Open WebUI, 9443: SearXNG, 8444: dashboard)
lmtune dash [install]             # live dashboard (model, requests, memory, stack, tailnet) on 127.0.0.1:8090
lmtune stack up | status | install   # Open WebUI + private SearXNG in podman, web only via Tor
lmtune stack verify --fail-closed    # prove research can't bypass Tor
lmtune harness opencode --write   # local-only OpenCode config + `oc` wrapper
lmtune media install | verify       # ffmpeg/exiftool/whisper.cpp as MCP tools: sandboxed, no network, one folder
lmtune bench                      # decode speed of the running server
lmtune doctor                     # config, flags vs build, memory, keys, listeners, stack, contention
```

Install with `make install` (a copied install: launchd-run `lmtune` cannot read ~/Documents
without a macOS privacy prompt, so editable installs hang under launchd). Architecture, threat
model and measurements: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Tune (measurement)

```bash
uv sync
uv run lmtune tune info ~/models/qwen38/Qwen3.8-27B-Q4_0.gguf     # layout, KV cost per token, MTP, template defaults
uv run lmtune tune plan profiles/qwen3.8-27b.toml                  # what fits, what would download, contention
uv run lmtune tune fetch profiles/qwen3.8-27b.toml --approve Qwen3.8-27B-Q4_0.gguf
uv run lmtune tune run profiles/qwen3.8-27b.toml                   # all stages (hours); --stages to pick
uv run lmtune tune report profiles/qwen3.8-27b.toml runs/<model>/<timestamp>
```

## What a run does

| stage | decision it makes |
|---|---|
| preflight | refuses to start if other LLM processes are loaded (Ollama, LM Studio, MLX, llama-server) |
| matrix | quant × flash-attention × ubatch → winner on decode speed, prompt speed breaks ties |
| noise | repeats the winner to measure run-to-run spread → the tie threshold |
| kv | KV cache type at several context depths → chosen at the deepest depth |
| knobs | threads, polling, load mode — adopted only if they beat the noise |
| server | speculative-decoding variants on fixed prompts/seeds/sampling (reasoning left at template defaults) |
| dist | sampled-output distribution check (target-model NLL, stratified by prompt) |
| memory | prompt-cache growth and multi-turn reuse → bounded `--cache-ram` that fits the RAM budget |
| report | `results.md` + `run-<model>.sh` using the best config that passed **both** gates |

A speculative config is only eligible if it is greedy token-identical to the baseline **and**
passes the distribution check. Every config runs under a swap guard and a contention scan;
contended or aborted results never win.

## Your constraints

`~/.config/lmtune/constraints.toml` (all optional):

```toml
download_ask_gb = 10        # never download a file bigger than this without --approve
min_free_disk_gb = 10
memory_headroom_gb = 6      # RAM left for macOS and your apps
swap_growth_limit_mb = 512  # abort a config if system swap grows more than this
```

The tool never runs `sudo`; servers bind to loopback only and restrict CORS to localhost.

## New model

Copy `profiles/qwen3.8-27b.toml`, set the repo, candidate files, the model card's sampling
parameters, and anything that must never be tried (`spec_n_max_banned`).
