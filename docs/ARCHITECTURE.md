# Private local AI on a Mac — architecture

One tuned model server on this Mac, used for coding (laptop), general questions and research
(laptop + iPhone). Nothing leaves devices you own unless you explicitly ask for web research.

## Goals

| goal | meaning here |
|---|---|
| Private | prompts, files, outputs and chat history stay on your devices; no telemetry, no cloud fallbacks |
| Secure | smallest attack surface that still works; least privilege; defense in depth; secrets never in argv or chat |
| Reliable | always reachable, self-healing, no silent degradation; every claim checkable (`lmtune doctor`) |
| Fast | the measured winner: Qwen3.8-27B Q4_0, f16 KV, MTP n=2 probabilistic — ~14 tok/s |

## Topology

```
 iPhone (Safari web apps) ─┐  WireGuard (Tailscale), device-authenticated
                           ▼
   tailscale serve :8443 ──► 127.0.0.1:3000 ─► Open WebUI ─┬─ lmtune-app (internal) ─► relay ─► host:8080
   tailscale serve :443  ──┐                  (podman VM)   └─ lmtune-web (internal) ─► SearXNG
   (HTTPS, tailnet-only)   │                                        │  socks5h / CONNECT / DNS
   laptop: `oc` (OpenCode) ┴────────► 127.0.0.1:8080                 ▼
                                      llama-server (launchd, Metal)  Tor gateway ─► obfs4 bridges ─► Tor ─► web

 your browsers (Mac, iPhone) ─► tailscale serve :9443 ─► 127.0.0.1:8888 ─► SearXNG   (opt-in; same Tor path)
 your browsers (Mac, iPhone) ─► tailscale serve :8444 ─► 127.0.0.1:8090 ─► lmtune dash (read-only)
```

The `:9443`/`:8888` path exists only when `searxng_port` is set in `stack.toml` (deployed:
8888): SearXNG is then also published on the Mac's loopback for your own browsers, and
`lmtune expose on searxng` puts it on the tailnet. It stays on `lmtune-web` only. Measured with a
throwaway container: podman publishes from an internal network without giving the VM an address
on it (inbound connections arrive from the container's own address), the container still has no
route out (`Network is unreachable`), and on the Mac gvproxy binds only `127.0.0.1`.

| network | kind | members | purpose |
|---|---|---|---|
| `lmtune-app` 10.153.1.0/24 | internal, no DNS, no IPv6 | Open WebUI, relay | the model, nothing else |
| `lmtune-web` 10.153.2.0/24 | internal, no DNS, no IPv6 | Open WebUI, SearXNG, Tor gateway | the web, only through Tor |
| `lmtune` | routed | relay | relay → `host.containers.internal:8080` |
| `lmtune-egress` | routed | Tor gateway | gateway → bridges |

Open WebUI and SearXNG have **no route** out of their internal networks (no default route, no
VM address, IPv6 off), so if Tor is down research fails instead of leaking. SearXNG uses Tor's
SOCKS port with remote DNS and refuses to search unless check.torproject.org says the proxy is
Tor; Open WebUI fetches pages through Tor's HTTP CONNECT tunnel and resolves names only through
Tor's DNSPort (its SSRF guard resolves before fetching). The gateway runs the sandbox's
`anon-tor` image with lmtune's own torrc, data volume and entrypoint, reading the sandbox's
bridge lines read-only; it is not attached to any sandbox network.

Verified on this machine: the only host listener is `127.0.0.1:8080` (no LAN exposure); requests
without the key get 401 locally and via the tailnet URL; the tailnet cert validates.
`lmtune stack verify --fail-closed` (17 probes, each "must be blocked" with a "must work" twin;
19 with `searxng_port` set) passes: direct internet/host blocked from Open WebUI and SearXNG,
page fetch exits via Tor, search via Tor works — from Open WebUI and, from the Mac, through the
published port; with the gateway stopped search returns 0 results on 3 of 3 tries on both paths,
page fetch and DNS fail, and search works again once Tor is back (19/19, 2026-10-23).

## Threat model

| threat | control | status |
|---|---|---|
| LAN / internet scanning | bind 127.0.0.1; remote only via Tailscale | ✓ verified |
| Malicious web page calling localhost from your browser | CORS restricted to localhost + API key | ✓ verified (no ACAO header; 401) |
| Another tailnet device (e.g. the GL.iNet router node) | API key; **recommended**: tailnet ACL allowing only iPhone + Mac to :443 (below) | key ✓, ACL pending you |
| Secrets leaking via process list / shell history / logs | key passed as `--api-key-file`, never `--api-key`; 0600 file in 0700 dir; `lmtune key show` is the only printer | ✓ |
| Harness phoning home (telemetry, sharing, auto-update, model catalogs) | OpenCode: one local provider, sharing/updates off, catalog/LSP/skill downloads off via `oc` wrapper; Open WebUI: `OFFLINE_MODE`, no telemetry, no update checks, env-authoritative config (`RESET_CONFIG_ON_START` re-seeds it from env at every start) | ✓ verified (`opencode debug config`, image source) |
| Other containers in the podman VM reaching the server | they can reach it via `host.containers.internal`; the API key is required | ✓ |
| Open WebUI account takeover | first account = admin (created while localhost-only); signup closed afterwards; **TOTP at every login** (`ENABLE_MFA`; codes single-use, 10 recovery codes, 10 tries per 15 min) | ✓ verified (signup 403, API 401 over the tailnet; MFA flow on a throwaway instance of the same image, below); enrollment pending you |
| Your SearXNG used by someone else (it has no login) | opt-in (`searxng_port`); loopback only; tailnet only via `lmtune expose on searxng`, so **the tailnet ACL is its access control** (below); no limiter/`public_instance` (they're for internet-facing instances and need Valkey, which isn't here). Any process on the Mac (or a page using DNS rebinding) can query it: worst case, a search through your Tor | design; ACL pending you |
| Browser leaking around SearXNG | `image_proxy: true`: result images come through SearXNG, so over Tor; `Referrer-Policy: no-referrer`. A result you *open* loads from your own IP: SearXNG hides searches, not visits | ✓ (settings) |
| Code execution through the chat UI | Open WebUI code execution/interpreter, image generation, webhooks, direct connections off; containers drop all capabilities, no-new-privileges, memory/pid limits | ✓ |
| Prompt injection from fetched web pages | research has fetch/search only; shell/edit tools require approval in coding sessions; web tools "ask" | design |
| Research linking queries/pages to your IP | all SearXNG and Open WebUI web traffic exits via Tor (obfs4 bridges); separate circuits per site for page fetches (`IsolateDestAddr`) | ✓ verified (`stack verify`) |
| Tor down → silent fallback to clearnet | Open WebUI and SearXNG sit only on internal networks with no resolver but Tor | ✓ verified (`stack verify --fail-closed`) |
| Components bypassing the proxy | anything that ignores proxy settings has no route: observed SearXNG fetching ClearURLs lists and a radio-browser DNS lookup directly — both blocked (radio browser removed) | ✓ observed |
| Container escape into the podman VM | the VM has **no** folders of the Mac (podman's default shares `/Users`, `/private`, `/var/folders` and `~/.config/containers` read-write — removed; configs reach containers as podman secrets); containers drop all caps, no-new-privileges, IPv6 off | ✓ verified (`doctor` vm-shares; no virtiofs mounts in the VM) |
| Hostile media files (ffmpeg/exiftool parser bugs) | media worker in `sandbox-exec`: no network at all (DNS, outbound, loopback), home unreadable except its folder, writes only there; ffmpeg limited to local protocols; worker refuses to run unsandboxed | ✓ verified (`lmtune media verify`, 18/18) |
| Supply chain | Homebrew llama.cpp build recorded and every flag verified per build (`doctor`); GGUFs sha256-verified; registry images pinned by digest, the local Tor image by ID | ✓ |
| Silent performance/correctness regressions after updates | `lmtune doctor`; `lmtune tune run` re-measures with greedy-equivalence + distribution gates | ✓ |
| Memory exhaustion / swap | idle unload frees ~21 GB; doctor budgets weights + KV + cache (+ vision projector); swap guard in benchmarks; measured: projector + 2 GB prompt cache, or whisper during decode, swap (see Media) | ✓ |
| Monitoring keeping the model loaded | the dashboard never calls `/slots` (it reloads a sleeping model and restarts the idle timer); `serve status`/`doctor` skip it while sleeping and prefer `/metrics` | ✓ verified (unloaded on schedule with the dashboard polling) |
| Dashboard seen by another tailnet device or a web page | read-only, no prompt/reply/search text, no secrets; loopback bind; tailnet only via `lmtune expose on dash` (**the ACL decides who sees it**); Host allowlist refuses DNS rebinding; hash-only CSP | ✓ verified (421 for a foreign Host); ACL pending you |
| `/metrics` revealing usage | needs the API key like every endpoint but `/health` | ✓ verified (401 locally and over the tailnet) |
| Losing `~/.config/lmtune/webui.secret` | TOTP secrets are encrypted with a key derived from it: without it Open WebUI refuses to start ("Stored MFA secrets cannot be decrypted"). lmtune never regenerates an existing file; keep a copy in your password manager | design |

### Recommended tailnet ACL (you apply it in the Tailscale admin console)

Tailscale's default policy lets every device reach every other. Narrow the served ports on this
Mac — 443 (model), 8443 (Open WebUI), 9443 (SearXNG) and 8444 (dashboard), the last two with no
login of their own — to your phone and the Mac itself (keep your existing rules for SSH etc.):

```jsonc
"hosts": { "mac": "100.64.0.1", "iphone": "100.64.0.2" },
"grants": [
  { "src": ["iphone", "mac"], "dst": ["mac"], "ip": ["tcp:443", "tcp:8443", "tcp:9443", "tcp:8444"] }
  // ...your other grants; remove any blanket allow-all that would still cover these ports
]
```

## Daily operation

```bash
lmtune serve start | stop | restart     # launchd-supervised, session only
lmtune serve install                    # start at login; restarts after a crash
lmtune serve status [--json]            # health, sleeping/busy, memory, listeners, tailnet URL
lmtune serve logs -f
lmtune expose on | off | status [webui|searxng|dash]   # tailnet-only HTTPS (:443, :8443, :9443, :8444)
lmtune dash [install | uninstall | snapshot]   # live dashboard on 127.0.0.1:8090 (below)
lmtune key rotate                       # then `lmtune serve restart`, update clients
lmtune doctor                           # every check that has bitten a real run
lmtune stack verify [--fail-closed]     # probe isolation + Tor from inside the containers
```

After `brew upgrade llama.cpp`: `lmtune doctor` (catches renamed/removed flags), then re-measure
with `lmtune tune run profiles/qwen3.8-27b.toml --stages preflight,server,dist,report`.

## Dashboard (`lmtune dash`)

A live, read-only page for the Mac and the phone: `http://127.0.0.1:8090` (login agent
`com.lmtune.dash`, installed with `lmtune dash install`) and, on the tailnet,
`https://mac.tail1234.ts.net:8444` (`lmtune expose on dash`; on the iPhone, Share → Add to
Home Screen). Stdlib only: one `http.server` and one self-contained page
(`src/lmtune/data/dash.html`) that polls `/api/state` every 3 s and stops while hidden. No
Prometheus or Grafana: the podman VM has ~600 MB free.

| shows | from | refresh |
|---|---|---|
| model state: stopped, not answering, loading, asleep, waking, idle (≈ time until it unloads), busy (prefill % or live decode tok/s) | `/health`, `/props` (`is_sleeping`), `/metrics` (`requests_processing`, queue), server log | 2 s |
| recent requests: decode and prefill tok/s, MTP draft acceptance, new (uncached) prompt tokens, output tokens, duration; medians of the last 10 | `server.log` `print_timing` lines, read incrementally | 2 s |
| memory pressure, app/wired/compressed/cached split, swap, CPU, GPU | `sysctl`, `vm_stat`, Mach host statistics (ctypes), `ioreg` IOAccelerator | 2 s |
| llama-server footprint, RSS, CPU | `footprint`, `ps` | 10 s, 2 s |
| containers (state, memory vs limit), podman VM memory available | `podman ps`/`stats`, `/proc/meminfo` via `podman machine ssh` | 10 s, 30 s |
| Tor bootstrap and warnings in the last hour; SearXNG errors per engine in the last 30 min, by kind | `podman logs`, classified and counted (no log text is shown) | 30 s |
| tailnet devices (connected = active WireGuard session), `tailscale serve` exposures (Funnel flagged), local listeners (non-loopback flagged) | `tailscale status`/`serve status --json`, `lsof` | 15 s, 10 s |

Usage by user and model is Open WebUI's: the page links to Admin → Analytics (`/admin/analytics`,
present in the pinned image) rather than duplicating it.

Nothing is collected while no page is open: a source refreshes only when a request finds it stale,
and the slow ones (podman, tailscale) refresh on a background thread so a poll never waits for them.

**It never wakes the model.** In llama.cpp b11669 every HTTP handler first waits for a sleeping
server to reload unless it passes `bypass_sleep` (`tools/server/server-context.cpp`): `/health`,
`/props` and `/metrics` do, `/slots` does not, and its task also restarts the idle timer, which
only METRICS tasks leave alone (`server-queue.cpp`, `task_resets_idle_timer`). The server log
showed a wake with no request after it, the signature of `lmtune serve status`, which called
`/slots` whenever the server answered. `Server.status` (`serve status`, `doctor`, `bench`) now skips
`/slots` while sleeping and asks `/metrics` first when awake. Verified: with the page polling every
3 s the model unloaded on schedule and stayed unloaded.

`--metrics` (present in b11669's `--help`; `doctor` checks every configured flag against the
build) is on. Like every endpoint but `/health` it needs the API key: 401 without it, locally and
over the tailnet. Measured before enabling it, `lmtune bench` (4 prompts × 512 tokens, same seeds),
back to back on 2026-10-23:

| run | config | decode tok/s |
|---|---|---|
| A | production | 14.05 |
| B | + `--metrics`, dashboard open, `/metrics` also scraped every second | 13.94 (1 GB of swap written during this run) |
| A | production | 14.13 |
| B | as above | 14.13 |
| A | production | 14.00 |

B averages 14.04 and A 14.06 (−0.2%, inside the 0.9% spread of the A runs): no measurable cost.

Security: binds 127.0.0.1 only and answers only `Host: 127.0.0.1:8090`, `localhost:8090` and the
tailnet name, so a web page can't reach it through DNS rebinding (421). The page runs only its own
inline script and style, allowed by hash (`default-src 'none'`), and builds every element with
`textContent`. It has no login: the tailnet ACL decides who sees it (port 8444 in the grant above).
It shows no prompt, reply or search text and no secrets.

Caveats: the time until the model unloads is estimated from the log, so anything that restarts the
idle timer without a log line (`/slots` from another client) makes the real unload later (seen:
44 s). A server started outside lmtune (no launchd agent, e.g. by hand for a benchmark) still shows
its state and memory, found by its port, but the request history is then the last managed run's.
Run from a launchd agent, the Tailscale app's bundled CLI tries to start the GUI ("The Tailscale
GUI failed to start", exit 0, no JSON) unless `TAILSCALE_BE_CLI=1` is set; lmtune sets it for
every `tailscale` command it runs.

## Clients

### Coding on the laptop — OpenCode (`oc`)

Installed (1.19.4, `anomalyco/tap`), config generated by `lmtune harness opencode --write` and
validated with `opencode debug config` / `opencode models` (exactly one model:
`lmtune/qwen3.8-27b`). End-to-end run: it read a file with its tool, then the model's attempt to
run `bash` was refused by policy (ask → auto-reject in non-interactive mode). Settings:

- only the local provider enabled; model `lmtune/qwen3.8-27b`; `share: disabled`; `autoupdate: false`
- `oc` wrapper exports `OPENCODE_DISABLE_{AUTOUPDATE,MODELS_FETCH,LSP_DOWNLOAD,EXTERNAL_SKILLS}=1`
  (switches found in the 1.19.4 binary; there are no config keys for them)
- permissions: bash ask, webfetch ask, edit allow
- API key read from the key file (`{file:~/.config/lmtune/api-key}`), never stored in the config
- one-time network use: OpenCode installs its `@ai-sdk/openai-compatible` npm package into
  `~/.config/opencode/node_modules` on first run (a package download; no prompts or code leave)

Measured: the first request of a session prefills OpenCode's 7.6K-token system prompt + tools
in 85 s (89 tok/s); the next turn prefilled only 144 new tokens (2.1 s). One slot
(`parallel = 1`), so phone and laptop requests queue.

### Phone

- **Open WebUI** (chosen for research and synced history): `https://mac.tail1234.ts.net:8443`
  → log in (password + authenticator code) → Share → Add to Home Screen. Web search: toggle it
  per chat; queries go through the private SearXNG, results are injected directly (no embedding
  model).
- **SearXNG directly** (search without the model, still over Tor): `https://mac.tail1234.ts.net:9443`
  once `lmtune expose on searxng` is on → Add to Home Screen; on the Mac also
  `http://127.0.0.1:8888`. No login: who can open it is decided by the tailnet ACL.

#### Open WebUI MFA (verified against the image source, open-webui 0.12.3 @ 91ea4e3)

`ENABLE_MFA=True` requires persistent config: with `ENABLE_PERSISTENT_CONFIG=False` (lmtune's old
setting) startup fails ("MFA requires authentication and persistent configuration"). But
persistent config alone lets the database override env: in a control run with
`ENABLE_MFA=True` and no reset, the stored `auth.mfa.enable = false` won and MFA stayed off, and
a value edited in the table survived the restart. So lmtune also sets `RESET_CONFIG_ON_START`,
which clears the config table at every start and re-seeds it from env (both reverted in the same
run). MFA secrets live on the account rows, not in that table, and survive the reset.

Measured on a throwaway instance (same image and env, no network, own volume): existing
sessions are rejected once MFA is on (401), so every account signs in again; sign-in then
returns an `enroll` step (QR code + manual key), a wrong code is refused, the right one issues a
session and 10 recovery codes; the next sign-in asks for a code, and a replayed code is refused.
Password sign-in carries the MFA step in the response body, so `http://127.0.0.1:3000` works as
well as the tailnet URL (only the OAuth flow checks the origin against `WEBUI_URL`).

Lost authenticator *and* recovery codes: an operator reset issues a one-time, 30-minute recovery
token, and the next sign-in asks for it and re-enrolls (measured on the throwaway instance):

```bash
podman exec -w /app/backend lmtune-webui python3 -c "from open_webui import app; app()" mfa reset you@example.com --reason "lost phone"
```
- **llama.cpp's built-in UI** (minimal, per-device history): `https://mac.tail1234.ts.net`;
  enter the API key once in its settings (`lmtune key show` on the Mac; move it with your
  password manager, not chat/notes).

The first message after 10 idle minutes wakes the model: measured 3.7 s total vs 1.6 s awake
(cold start 8.2 s; asleep the server holds 0.5 GB instead of 20 GB).

### Research and general questions — options considered (B deployed)

| | A. Built-in UI + OpenCode research | B. Open WebUI | C. A + SearXNG search tool |
|---|---|---|---|
| Phone research (web search) | no | yes | laptop only (OpenCode) |
| Chat history | per device | synced (server-side, on this Mac) | per device |
| New services on the Mac | none | Open WebUI (podman) + SearXNG | SearXNG (podman) |
| Attack surface | smallest | largest (web app with accounts, plugins that can run code) | small |
| Privacy setup needed | none | telemetry env vars off, signup off after first account, plugins off | none beyond SearXNG config |
| Search privacy | queries go to whatever engine OpenCode uses | via SearXNG (strips identity) | via SearXNG |

### Research over Tor (measured)

Search queries and fetched pages necessarily leave the Mac; they leave through Tor. Measured
against a temporary direct-route SearXNG (same image/settings minus the proxy), queries
alternated:

| | search (6 queries) | results | page fetch (6 sites × 2) |
|---|---|---|---|
| direct | median 0.73 s | 32–39 | 0.04–0.3 s |
| Tor | median 1.0 s | 23–26 | 1–7 s typical; one site (arstechnica) ~50 s both times |

Page-fetch cost is small next to prefill: a 20,000-char page cap (`fetch_max_chars`) is ≈5K
tokens ≈ a minute of prefill. Only `https://` pages load (Tor's tunnel is CONNECT-only; plain
`http://` is refused, never sent in clear through an exit).

#### Making search over Tor reliable (paired A/B, 20 trials per round, 2026-10-22)

Searches sometimes came back empty. Each round ran two SearXNG instances side by side on one
test Tor, restarted Tor every trial (fresh circuits), and sent both the same 3 queries.

| round | compared | searches with results | verdict |
|---|---|---|---|
| 1 | one Tor identity vs 4 rotating identities + a circuit per site (`IsolateDestAddr`) | 58/60 vs 34/60 (24–0 discordant, p<0.001) | **rejected**: more fresh exits → more rate-limits |
| 3 | default engines vs allowlist, both cold-started each trial | 53/60 vs 57/60; results/search 30 vs 52 | allowlist **adopted** |
| 4 | allowlist with 3 s vs 10 s engine timeout, cold-started | 53/54 vs 57/57; 14 of 57 searches needed >3 s | 10 s **adopted** (median +0.3 s, p90 4.8 s) |

(Round 2 compared timeouts on instances of different age and is discarded: SearXNG's per-engine
suspensions accumulate over an instance's life, which confounded it.)

Two failure modes were behind it:

- **Only Brave worked.** SearXNG's default web engines over Tor: DuckDuckGo (html) "access
  denied", Google CSE and Wikipedia "too many requests". When an exit was rate-limited by Brave,
  SearXNG suspended Brave (180 s, for every circuit) and searches returned nothing. A survey of 26
  engines (3 queries each over Tor) found brave, duckduckgo web, bing, yahoo and yep answering;
  Google, Qwant, GMX (CAPTCHA), Yandex, Seznam and most small engines blocked. lmtune keeps only
  `brave, duckduckgo web, bing, yahoo, yep, wikidata` — also fewer third parties per query.
- **3 s per engine is too short over Tor.** A slow circuit made every engine time out at once.
  `extra_proxy_timeout` doesn't help: SearXNG adds it only to engines with an `.onion` URL.

SearXNG's startup Tor check fails occasionally (~6 of 62 starts in these runs); it then exits and
podman restarts it, so this self-heals in production.

## Containers

The model server runs natively: Linux containers on macOS have no Metal; the only GPU path is
Vulkan through podman's krunkit VM. Measured (same prompts, native vs container, A-B-A):

| image | GPU in VM | 2 GB model gen / prompt | 27B (Qwen3.8) |
|---|---|---|---|
| native Metal (b11669) | — | 68.8–71.1 / 946–1251 tok/s | 13.2 (MTP greedy) – 14.3 (MTP prob) tok/s |
| `ggml-org/llama.cpp:server-vulkan-b11669` | none (venus finds no device) → CPU | 43.3 / 400 (61% / 42%) | not tried (CPU) |
| `quay.io/ramalama/ramalama` | Virtio-GPU Venus ✓ | 53.7 / 556 (78% / 44%) | OOM-killed in a 26 GB VM (weights held twice); its llama.cpp also ignores the MTP head |

Verdict: native. `backend = "podman"` stays in `serve.toml` for a future image/runtime that
measures close; the side services (Open WebUI, SearXNG) run in podman.

## Media: images, transcription, ffmpeg (measured 2026-10-23)

Understanding, not generation: the 27B reads images through its vision projector, and a
sandboxed **media worker** gives clients ffmpeg, exiftool and whisper.cpp as MCP tools.

### Vision: Qwen3.8's projector (`mmproj`)

`unsloth/Qwen3.8-27B-GGUF` ships `mmproj-F16.gguf` (0.93 GB, sha256-verified; F16 over BF16 for
Metal). llama-server b11669 takes it with `--mmproj` (also `--image-min-tokens`,
`--image-max-tokens`, `--mmproj-offload`; `lmtune doctor` checks flags per build). `ServeConfig`
has `mmproj` (off unless set); `doctor` budgets it (file + 128 MiB measured buffers) and the
OpenCode harness then declares the model image-capable.

Same prompts and seeds as `lmtune bench` (4 × 512 tokens, seeds 1 and 2), the production argv
± `--mmproj`, each phase a fresh server, phases interleaved:

| phase | config | decode seed 1 / 2 (tok/s) | MTP acceptance | peak footprint | swapped out during phase |
|---|---|---|---|---|---|
| A1 | production | 14.03 / 14.89 | 0.638 / 0.725 | 7.6 GB | (swap used +295 MB) |
| B1 | + mmproj, + 6 image requests | 14.02 / — | 0.638 | ≥ 8.1 GB | **> 512 MB: swap guard aborted it** |
| A2 | production | 14.14 / 15.14 | 0.638 / 0.725 | 7.6 GB | 0 |
| B2 | + mmproj | 14.02 / 14.95 | 0.638 / 0.725 | 8.6 GB | **245 MB** |
| A3 | production | 14.14 / 15.15 | 0.638 / 0.725 | 7.6 GB | 0 |
| B3 | + mmproj, `cache_ram_mib` 1024 | 14.12 / 15.08 | 0.638 / 0.725 | 7.9 GB | 0 |
| A4 | production | 13.96 / 15.19 | 0.630 / 0.725 | 7.6 GB | 0 |

- **Decode: no cost.** Every B value sits inside the A range (13.96–14.14 / 14.89–15.19). MTP
  is unaffected: same draft context, identical acceptance counts request for request.
- **Memory: +0.93–1.0 GB footprint** (the projector's weights live in GPU buffers, not the
  mmapped file), present from load. With the 2 GB prompt cache full, that tipped this Mac into
  swap in both B runs, never in an A run with the swap-out counter. Halving the prompt cache
  (B3) brought the footprint back to +0.3 GB over A and swapping to zero.
  `lmtune doctor`'s arithmetic said it fits (23.4 of the 26 GiB budget); the machine says the
  real margin is thinner (the podman VM, Claude, Creative Cloud etc. are not in that budget).

Image requests (MTP active on them too; 14.7–15.6 tok/s decode):

| input | image tokens + prompt | prefill | result |
|---|---|---|---|
| 512×512 photos (eagle, guitar, sunflower, fortune cookie) | 327 | 4.6–5.0 s | accurate subject, colors, composition; the fortune cookie described exactly but not named |
| 612×792 receipt page, ~10 px monospace, white background | 554 | 7.7 s | **character-perfect** transcription, total and card digits right |
| 612×792 config file, same font | 547 | 7.5 s | one line wrong: a dropped comma, `draft-mtp` → `draft-ntp` |
| the same two pages as transparent PNGs (black text on alpha) | 554 / 547 | 7.5–7.8 s | "the image is entirely black" — 4 of 4 tries |

At load llama-server warns that Qwen-VL models need ≥ 1024 image tokens for *grounding*
(bounding boxes; `--image-min-tokens 1024`). Not needed for the reading and description above;
worth measuring if you ask it where things are in an image.

**Transparent images look black to llama-server** (control above: same pixels flattened on
white read perfectly). The model said so instead of inventing text. The media worker's
`convert` flattens images onto white for this reason.

### Media worker (`lmtune media`)

One stdlib process speaks MCP (streamable HTTP, JSON responses) on `127.0.0.1:8095/mcp` behind
its own bearer key (`~/.config/lmtune/media-key`, 0600), and runs inside `sandbox-exec` with a
generated profile:

| | allowed | denied |
|---|---|---|
| network | accept on its own loopback port | everything else: DNS, outbound (loopback too, so not even llama-server), binding other ports |
| home folder | read/write the media folder (`~/lmtune-media`); read the whisper models, lmtune's own install, `media.toml`, its key | everything else (`~/.ssh`, `~/.config/lmtune/api-key`, `~/Documents`, …) |
| elsewhere | read system/Homebrew; write nowhere but the media folder | writes to `/tmp` etc. |

Tools: `list_files`, `probe` (ffprobe), `metadata` (exiftool), `transcribe` (ffmpeg → 16 kHz
WAV → whisper.cpp; saves `out/<name>.txt`/`.srt`), `convert` (cut/scale/transcode; images
flattened on white), `frames` (JPEG stills from video, for the vision model). Paths are
relative to the media folder and checked (no `..`, absolute paths, or symlinks out); ffmpeg
runs with `-protocol_whitelist file,pipe`; one heavy job at a time. The worker refuses to start
unsandboxed (`sandbox_check`), and `lmtune media verify` runs probes *inside* the same profile,
each blocked case with a working twin. On this Mac: **18/18** — no DNS; no connection to
1.1.1.1 or to llama-server on loopback; ffmpeg can't open an `http://` input; no listening on
other ports; `~/.config/lmtune/api-key`, `~/.ssh`, `~/Documents` unreadable; no writes to home
or `/tmp`; the media folder, the whisper model and its key work; 401 without the key, the six
tools with it. Paths that try to leave the folder (`../../etc/hosts`, `~/.ssh/config`) and
shell-ish arguments are refused before anything runs.

The sandbox costs a fixed **~5 s per transcription**: Metal can't write its shader cache
(`$DARWIN_USER_CACHE_DIR/com.apple.metal{,fe}`), so whisper's kernels compile every run
(base.en on 87 s: 6.6 s sandboxed vs 1.45 s with those two folders writable, bisected rule by
rule). Left denied: that cache is shared by every app's GPU code and the worker parses untrusted
files; it's ~3% on an hour of audio. Allowing it is one `file-write*` line if short clips matter.

**Why native, not a podman container** (measured): whisper turbo on CPU with 2 threads (what a
container in the 2-vCPU VM gets) ran at **1.0× realtime** vs **14×** on Metal for the same 35 s
clip; and the VM deliberately has no Mac folders, so a container could only see media by
re-opening a host share for the whole VM.

#### Transcription (whisper.cpp 1.9.5, Metal, 4 threads)

Samples: speech from macOS `say` with known scripts (35 s, 87 s, 212 s), plus the 87 s clip
band-limited to telephone quality with pink noise at ~18 dB and ~5 dB SNR. WER after number
normalization.

| model | size | speed (× realtime) | peak memory | WER clean | WER 18 dB | WER 5 dB |
|---|---|---|---|---|---|---|
| `ggml-large-v3-turbo-q5_0` (default) | 574 MB | 17–22 (12–14 on 35 s); 13 through the sandboxed worker on 212 s | 0.86–0.88 GB | 0% | 0.7% | **1.8%** |
| `ggml-base.en-q5_1` | 60 MB | 46–51 (41 on 35 s) | 0.25–0.29 GB | 0–0.15% | 2.6% | **9.1%**, lost a whole paragraph |

Clean synthetic speech doesn't separate the models; noise does. An hour of audio ≈ 3.5 min with
turbo.

**With the 27B loaded:** idle, whisper ran at the same speed with zero swap-outs. Transcribing
back-to-back *while the 27B decodes* (worst case, `lmtune bench` workload, A-B-A):

| | LLM decode | LLM prefill | whisper turbo | swapped out |
|---|---|---|---|---|
| LLM alone (before / after) | 14.01 / 14.13 tok/s | 78.4 / 78.6 tok/s | — | 0 |
| together, default QoS | **8.84 tok/s (−37%)** | 58.6 (−25%) | 11.3–12.6× (−35%) | ~650 MB |
| together, whisper at background QoS (`taskpolicy -b`) | 10.21 tok/s (−27%) | 62.5 (−20%) | 7.5–8.9× (−53%) | ~470 MB |

They share one GPU and the memory margin, Background QoS is not the fix: with the LLM idle it makes whisper **38% slower** (13.3–13.5×
vs 21.7–22×, alternated), and under contention it only stretches the same loss — about 1.9
decode-minutes per hour of audio either way. So the worker runs at default QoS (launchd
`ProcessType=Standard`), and long transcriptions belong in idle time: the background-jobs idea
in a background-jobs design (start a job, get notified) is the right home for them. Memory: 0.86 GB for turbo
on top of a full prompt cache swapped ~0.5–0.65 GB out; with the model idle it didn't swap.

### OpenCode

`lmtune harness opencode` adds the worker when its key exists (`mcp.media`: remote, loopback
URL, `Authorization: Bearer {file:…media-key}` so the key never sits in the config, `oauth:
false`, timeout = `timeout_s`; OpenCode's default 5 s would cut transcriptions off). With
`mmproj` set, the model gets `attachment: true` and image input. Validated against opencode
1.19.4 through an `OPENCODE_CONFIG` overlay (your `opencode.json` untouched): `oc mcp list` →
`✓ media connected` (so the `{file:…}` key works; the server 401s anything else), `oc debug
config` masks the header. End to end, `oc run "list the media files, transcribe noisy5.wav and
tell me when the backup test is due"`: the 27B called `media_list_files` then
`media_transcribe` (5 dB-SNR clip, 11 s in the sandbox) and answered "November 5th"; 186 s in
total, most of it the first prefill.

### Open WebUI (not wired; what it would take)

Open WebUI supports MCP tool servers natively, but it sits only on internal networks
(`lmtune-app`, `lmtune-web`) with no route to the host, and must stay that way. The worker must be
reached through the existing relay rather than a new path:

1. `lmtune-relay` (already on `lmtune-app` and the routed `lmtune` network, already piping to
   `host.containers.internal:8080`) gets a second pipe: `:8095 → host.containers.internal:8095`.
   No new container, network or egress: the worker itself still has no network at all.
2. `media.toml`: `extra_hosts = ["<relay's lmtune-app IP>:8095"]` (the worker refuses unknown
   Host headers).
3. Open WebUI: an MCP tool-server connection to `http://<relay IP>:8095/mcp` with the media key
   as a bearer token, delivered as a podman secret, not env in `inspect`. Verify the exact
   setting name against the pinned image's source first (as done for every other setting).
4. `stack verify`: probes that Open WebUI reaches the worker with the key, gets 401 without it,
   and still has no route anywhere else.
5. **Files are the real gap**: uploads land in Open WebUI's volume inside the VM, which the
   worker can't see (by design). Options: an upload tool with a size cap (base64 in a JSON-RPC
   call; fine for photos, not for an hour of audio), or point Open WebUI's own speech-to-text at
   `whisper-server` (whisper.cpp ships one) through the same relay. Measure the second before
   building the first.

### Recommendation

**Keep**

1. **The media worker as built**: native, sandboxed, `large-v3-turbo-q5_0` by default, at
   default QoS. **Deployed 2026-10-23**: `lmtune media install` (login agent `com.lmtune.media`,
   `~/lmtune-media`, `~/.config/lmtune/media-key`), `lmtune media verify` 18/18 on the installed
   copy, OpenCode config rewritten (`oc mcp list`: connected; old config in `opencode.json.bak`).
2. **Vision, with the prompt cache halved**: `mmproj = "~/models/qwen38/mmproj-F16.gguf"` and
   `cache_ram_mib = 1024` in `serve.toml`. **Deployed 2026-10-23** (backup
   `serve.toml.bak-2026-10-23`); after the restart: `lmtune bench` 14.39 tok/s, MTP 63%, a receipt
   read correctly through the production server. Measured: no decode cost, MTP intact, no swap (1 of 1 run; with the 2 GB
   cache, 2 of 2 runs swapped). The price is saved conversations: ~2 OpenCode-sized states
   (~0.5 GB each) instead of ~4, so juggling more long conversations re-prefills more often
   (~85 s for OpenCode's 7.6K-token prefix). If that matters more than images, leave `mmproj`
   unset: support stays in `ServeConfig`, off by default.

**Measured and rejected**: the projector with the 2 GB prompt cache (swap); `base.en` as the
default (9% WER and a dropped paragraph at 5 dB SNR; the 60 MB file can stay as a fast option);
a podman container for the worker (CPU-only, 14× slower, needs a VM folder share); background
QoS for whisper (38% slower, saves the LLM nothing overall); a writable Metal cache in the sandbox
(~5 s per run isn't worth a shared-cache write path).

**Not built yet**: Open WebUI wiring (above), and running long transcriptions as background
jobs at idle times (planned): transcription while you chat costs the chat 27–37% decode speed.
