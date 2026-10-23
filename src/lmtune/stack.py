"""Research/chat stack in podman: Open WebUI + SearXNG, locked down, using the local llama-server,
with every byte of web traffic going through Tor.

Every Open WebUI setting below was verified against the source of the image it runs
(open-webui main @ 91ea4e3): names exist, and `RESET_CONFIG_ON_START=True` re-seeds the config
from these env values at every start, so security settings can't drift through the admin UI.

Networks (all podman bridges inside the VM):

    lmtune-app  internal, no DNS   Open WebUI ── relay                 (the model, nothing else)
    lmtune-web  internal, no DNS   Open WebUI, SearXNG ── Tor gateway  (the web, only via Tor)
    lmtune      routed             relay ──► host.containers.internal:8080 (llama-server)
    lmtune-egress routed           Tor gateway ──► bridges / Tor network

Open WebUI and SearXNG sit only on internal networks: they have no route to the internet, the
host or the VM, so if the gateway is down research fails instead of leaking (`lmtune stack
verify --fail-closed` proves it). SearXNG reaches engines through Tor's SOCKS port with remote
DNS, and refuses to search unless check.torproject.org says the proxy is Tor. Open WebUI fetches
pages through Tor's HTTP CONNECT tunnel (its loader honors only HTTP proxies) and resolves names
only through Tor's DNSPort (its SSRF guard needs an answer before it fetches). The gateway runs
the sandbox's `anon-tor` image with lmtune's own torrc and data, reading the sandbox's bridge
lines read-only; it shares nothing else with the sandbox.

- Open WebUI is published on host loopback only; remote access is `tailscale serve` (tailnet).
- SearXNG is published on host loopback too, only if `searxng_port` is set, for your own browsers
  (`lmtune expose on searxng` puts it on the tailnet). It has no login: the tailnet ACL is its
  access control. Podman publishes from an internal network without giving the VM an address on
  it, so SearXNG still has no route out but Tor (measured).
- Nothing else is published.
- Secrets (server API key, JWT signing key, SearXNG key) are podman secrets, not env in `inspect`.
- Nothing on the Mac's disk is mounted into a container: generated configs (torrc, SearXNG
  settings) are podman secrets too, mounted read-only at their paths. So the podman VM needs no
  host folder shares at all (`lmtune doctor` fails if it has any).
- Images run pinned by digest (registry images) or image ID (the locally built Tor image).
- The first account created becomes admin (Open WebUI doesn't gate it on ENABLE_SIGNUP), so the
  stack stays loopback-only until you've created it; signup is otherwise disabled.
- Every login needs a TOTP code (ENABLE_MFA); accounts enroll at their next sign-in.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import re
import secrets
import shutil
import time
import tomllib
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from . import keys
from .launchd import Launchd, plist
from .paths import Paths
from .proc import Result, Runner, run

STACK_LABEL = "com.lmtune.stack"
WEBUI = "lmtune-webui"
SEARXNG = "lmtune-searxng"
GATEWAY = "lmtune-tor"
RELAY = "lmtune-relay"
CONTAINERS = (WEBUI, SEARXNG, RELAY, GATEWAY)
NET_APP = "lmtune-app"
NET_WEB = "lmtune-web"
NET_EGRESS = "lmtune-egress"
INTERNAL_NETS = (NET_APP, NET_WEB)
SECRETS = {"api": "lmtune-api-key", "webui": "lmtune-webui-secret", "searxng": "lmtune-searxng-secret"}
CONFIGS = {"torrc": "lmtune-torrc", "searxng": "lmtune-searxng-settings"}  # files, as secrets
TOR_DATA = "lmtune-tor-data"
SOCKS_PORT, TUNNEL_PORT, DNS_PORT = 9050, 9080, 53
SANDBOX_NET = ipaddress.ip_network("10.152.0.0/16")
# Web search over Tor, from paired A/B runs (docs/ARCHITECTURE.md). SearXNG's defaults leave Brave
# as the only engine that answers through Tor exits (DuckDuckGo, Google CSE and Wikipedia block or
# rate-limit them), so one rate-limited exit suspended it and searches came back empty; with these,
# three independent indexes (Brave, Bing via Bing/DDG/Yahoo, Yep) answer. Wikidata adds infoboxes.
SEARCH_ENGINES = ("brave", "duckduckgo web", "bing", "yahoo", "yep", "wikidata")
SEARCH_SURVEYED = "2026-10-22"
SEARCH_TIMEOUT_S = 10.0  # the anon sandbox allocates persona subnets here

# Tor's own startup: own the data volume (a fresh volume is root's), then bind as root and drop
# to debian-tor (torrc `User`). The image's entrypoint is the sandbox's and only accepts its
# 10.152/16 addresses, so lmtune brings its own.
TOR_START = (
    "set -eu; chown -R debian-tor:debian-tor /var/lib/tor; chmod 700 /var/lib/tor; exec tor -f /etc/lmtune/torrc"
)
# chown/chmod above (CHOWN, DAC_OVERRIDE, FOWNER), DNSPort 53 (NET_BIND_SERVICE), the drop (SETGID/SETUID)
TOR_CAPABILITIES = ("CHOWN", "DAC_OVERRIDE", "FOWNER", "NET_BIND_SERVICE", "SETGID", "SETUID")
NO_IPV6 = ["--sysctl", "net.ipv6.conf.all.disable_ipv6=1", "--sysctl", "net.ipv6.conf.default.disable_ipv6=1"]

# A byte pipe from the app network to llama-server on the host: the only way Open WebUI reaches
# anything outside its internal networks. Stdlib only; runs in the SearXNG image's Python.
RELAY_PY = """\
import asyncio, sys
listen, port, target, tport = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
async def pipe(r, w):
    try:
        while data := await r.read(65536):
            w.write(data)
            await w.drain()
        if w.can_write_eof():
            w.write_eof()
    except OSError:
        pass
async def handle(cr, cw):
    try:
        ur, uw = await asyncio.open_connection(target, tport)
    except OSError:
        cw.close()
        return
    await asyncio.gather(pipe(cr, uw), pipe(ur, cw))
    uw.close()
    cw.close()
async def main():
    server = await asyncio.start_server(handle, listen, port)
    async with server:
        await server.serve_forever()
asyncio.run(main())
"""

# What `verify` runs inside a container: one network action, one JSON line out. Stdlib only; the
# proxy is chosen per probe (an empty ProxyHandler means direct, whatever the environment says).
PROBE_PY = """\
import json, socket, sys, urllib.request
def out(ok, detail, limit=400):
    print(json.dumps({"ok": ok, "detail": str(detail)[:limit]}))
    sys.exit(0)
mode, args = sys.argv[1], sys.argv[2:]
try:
    if mode == "connect":
        socket.create_connection((args[0], int(args[1])), timeout=8).close()
        out(True, "connected")
    if mode == "resolve":
        out(True, " ".join(sorted({a[4][0] for a in socket.getaddrinfo(args[0], 443, proto=socket.IPPROTO_TCP)})))
    if mode == "get":  # proxy: "" direct, "env" the container's own proxy settings, else that proxy
        proxy = args[1] if len(args) > 1 else ""
        chosen = {"http": proxy, "https": proxy} if proxy else {}
        handlers = [] if proxy == "env" else [urllib.request.ProxyHandler(chosen)]
        opener = urllib.request.build_opener(*handlers)
        with opener.open(args[0], timeout=60) as r:
            out(True, r.read(2000000).decode(errors="replace"), limit=None)
    out(False, "unknown probe " + mode)
except Exception as e:
    out(False, type(e).__name__ + ": " + str(e))
"""


@dataclass(frozen=True)
class StackConfig:
    podman: str = "/opt/homebrew/bin/podman"
    network: str = "lmtune"  # routed: the relay's way to the host
    webui_image: str = "ghcr.io/open-webui/open-webui:main"
    searxng_image: str = "docker.io/searxng/searxng:latest"
    tor_image: str = "localhost/anon-tor:latest"  # built by ~/Documents/code/sandbox
    tor_bridges_volume: str = "anon-tor-conf"  # mounted read-only; "" = reach Tor without bridges
    app_subnet: str = "10.153.1.0/24"  # internal: Open WebUI ── relay
    web_subnet: str = "10.153.2.0/24"  # internal: Open WebUI, SearXNG ── Tor gateway
    webui_port: int = 3000  # host loopback
    webui_https_port: int = 8443  # tailnet
    searxng_port: int = 0  # host loopback, for your browsers; 0 = SearXNG isn't published
    searxng_https_port: int = 9443  # tailnet (`lmtune expose on searxng`)
    webui_memory: str = "1536m"
    searxng_memory: str = "512m"
    tor_memory: str = "256m"
    relay_memory: str = "64m"
    # fetch_url page text cap. Every character becomes prefill (~4 chars/token at ~90 tok/s), so
    # 20000 chars ≈ 5K tokens ≈ a minute per page; unbounded pages can take many minutes.
    fetch_max_chars: int = 20000
    tor_bootstrap_s: int = 180  # how long `up` waits for Tor before starting SearXNG
    server_port: int = 8080  # the lmtune llama-server, reached via host.containers.internal
    model_alias: str = "qwen3.8-27b"
    tailnet_host: str = ""  # e.g. mac.tailXXXX.ts.net; sets WEBUI_URL and CORS

    @property
    def webui_url(self) -> str:
        if self.tailnet_host:
            return f"https://{self.tailnet_host}:{self.webui_https_port}"
        return f"http://127.0.0.1:{self.webui_port}"


@dataclass(frozen=True)
class Addresses:
    """Fixed addresses on the internal networks (no DNS there, so everything is by IP)."""

    relay: str
    webui_app: str
    gateway: str
    searxng: str
    webui_web: str

    @classmethod
    def of(cls, c: StackConfig) -> Addresses:
        app, web = ipaddress.ip_network(c.app_subnet), ipaddress.ip_network(c.web_subnet)
        return cls(str(app[2]), str(app[3]), str(web[10]), str(web[2]), str(web[3]))

    @property
    def tunnel(self) -> str:
        return f"http://{self.gateway}:{TUNNEL_PORT}"


def problems(c: StackConfig) -> list[str]:
    out = []
    nets = {}
    for name, value in (("app_subnet", c.app_subnet), ("web_subnet", c.web_subnet)):
        try:
            net = ipaddress.ip_network(value)
        except ValueError:
            out.append(f"{name} {value!r} is not a network")
            continue
        if not isinstance(net, ipaddress.IPv4Network) or net.prefixlen > 24 or not net.is_private:
            out.append(f"{name} {value} must be a private IPv4 /24 or larger")
        if net.overlaps(SANDBOX_NET):
            out.append(f"{name} {value} overlaps the anon sandbox's {SANDBOX_NET}")
        nets[name] = net
    if len(nets) == 2 and nets["app_subnet"].overlaps(nets["web_subnet"]):
        out.append("app_subnet and web_subnet overlap")
    if c.fetch_max_chars <= 0:
        out.append("fetch_max_chars must be positive")
    if c.searxng_port and not 1024 <= c.searxng_port <= 65535:
        out.append(f"searxng_port {c.searxng_port} must be 0 (unpublished) or 1024-65535")
    if c.searxng_port in (c.webui_port, c.server_port):
        out.append(f"searxng_port {c.searxng_port} is already Open WebUI's or the model server's")
    if c.searxng_https_port in (c.webui_https_port, 443):
        out.append(f"searxng_https_port {c.searxng_https_port} is already Open WebUI's or the model server's")
    return out


def load(path: Path) -> StackConfig:
    if not path.exists():
        return StackConfig()
    raw = tomllib.loads(path.read_text()).get("stack", {})
    unknown = set(raw) - {f.name for f in fields(StackConfig)}
    if unknown:
        raise ValueError(f"unknown [stack] keys: {', '.join(sorted(unknown))}")
    cfg = StackConfig(**raw)
    if errs := problems(cfg):
        raise ValueError("; ".join(errs))
    return cfg


def torrc(c: StackConfig) -> str:
    a = Addresses.of(c)
    bridges = "%include /etc/tor/conf.d/*.conf\n" if c.tor_bridges_volume else ""
    return f"""\
# Generated by lmtune: a Tor client for web research, never a relay.
User debian-tor
DataDirectory /var/lib/tor
# Listeners on the internal web network only; nothing is published to the Mac.
SocksPort {a.gateway}:{SOCKS_PORT} IsolateSOCKSAuth
HTTPTunnelPort {a.gateway}:{TUNNEL_PORT} IsolateDestAddr
DNSPort {a.gateway}:{DNS_PORT}
SocksPolicy accept {c.web_subnet}
SocksPolicy reject *
SafeSocks 1
ClientOnly 1
AvoidDiskWrites 1
Log notice stdout
# Bridge lines live in the sandbox's volume (read-only here), never in this file.
{bridges}"""


def searxng_settings(c: StackConfig) -> str:
    a = Addresses.of(c)
    return f"""\
# Generated by lmtune. The secret key comes from the SEARXNG_SECRET podman secret.
use_default_settings:
  engines:
    # Only engines that answer over Tor (surveyed {SEARCH_SURVEYED}); every query reaches only these.
    keep_only: [{", ".join(SEARCH_ENGINES)}]
general:
  enable_metrics: false
server:
  # Private: Open WebUI, plus your own browsers if searxng_port is set (loopback, and the tailnet
  # via `lmtune expose on searxng`, where the tailnet ACL decides who gets in). The limiter and
  # public_instance are for instances open to the internet and need Valkey, which isn't here
  # (with public_instance SearXNG exits without it).
  limiter: false
  public_instance: false
  # Result images come through SearXNG, so over Tor, instead of from your browser's own address.
  image_proxy: true
search:
  safe_search: 0
  formats: [html, json]   # Open WebUI queries the JSON API
outgoing:
  # Every engine request goes through lmtune's Tor gateway, names resolved by Tor (socks5h).
  proxies:
    "all://":
      - socks5h://searxng:lmtune@{a.gateway}:{SOCKS_PORT}
  # SearXNG asks check.torproject.org through the proxy and refuses to search unless it is Tor.
  using_tor_proxy: true
  # Per engine. The 3 s default cut 1 search in 4 short over Tor (measured); extra_proxy_timeout
  # would not help: SearXNG adds it only to engines with an .onion address.
  request_timeout: {SEARCH_TIMEOUT_S}
engines:
{"".join(f"  - name: {name}\n    disabled: false\n" for name in SEARCH_ENGINES)}"""


def webui_env(c: StackConfig) -> dict[str, str]:
    """Open WebUI environment: local model only, nothing phoning home, no code execution, web
    only through Tor."""
    a = Addresses.of(c)
    return {
        # model: only the local llama-server, through the relay (key arrives as a podman secret)
        "OPENAI_API_BASE_URLS": f"http://{a.relay}:{c.server_port}/v1",
        "ENABLE_OLLAMA_API": "False",
        "ENABLE_DIRECT_CONNECTIONS": "False",  # users can't add their own external APIs
        "DEFAULT_MODELS": c.model_alias,
        # accounts
        "WEBUI_AUTH": "True",
        # TOTP at every login; an account without one enrolls at its next sign-in. Secrets are
        # encrypted with a key derived from WEBUI_SECRET_KEY: lose that file and they're unreadable.
        "ENABLE_MFA": "True",
        "ENABLE_SIGNUP": "False",  # the first account (admin) is still allowed; nobody after it
        "DEFAULT_USER_ROLE": "pending",
        "ENABLE_OAUTH_SIGNUP": "False",
        "WEBUI_URL": c.webui_url,
        # `;`-separated (config.py: .split(';')): the tailnet URL plus local access on this Mac
        "CORS_ALLOW_ORIGIN": f"{c.webui_url};http://127.0.0.1:{c.webui_port};http://localhost:{c.webui_port}",
        # nothing leaves except explicit web research
        "OFFLINE_MODE": "True",  # HF_HUB_OFFLINE, no update checks, no model auto-downloads
        "HF_HUB_OFFLINE": "1",
        "ENABLE_VERSION_UPDATE_CHECK": "False",
        "ANONYMIZED_TELEMETRY": "False",
        "DO_NOT_TRACK": "True",
        "SCARF_NO_ANALYTICS": "True",
        "ENABLE_COMMUNITY_SHARING": "False",
        "ENABLE_FORWARD_USER_INFO_HEADERS": "False",
        "ENABLE_USER_WEBHOOKS": "False",
        # no code execution surfaces
        "ENABLE_CODE_EXECUTION": "False",
        "ENABLE_CODE_INTERPRETER": "False",
        "ENABLE_IMAGE_GENERATION": "False",
        "ENABLE_EVALUATION_ARENA_MODELS": "False",
        # every extra task is another call on a single-slot thinking model, delaying your next message
        "ENABLE_TITLE_GENERATION": "False",
        "ENABLE_TAGS_GENERATION": "False",
        "ENABLE_FOLLOW_UP_GENERATION": "False",
        "ENABLE_AUTOCOMPLETE_GENERATION": "False",
        # research: search_web via the private SearXNG, fetch_url for whole pages, both over Tor
        "ENABLE_WEB_SEARCH": "True",
        "WEB_SEARCH_ENGINE": "searxng",
        "SEARXNG_QUERY_URL": f"http://{a.searxng}:8080/search?q=<query>",
        "BYPASS_WEB_SEARCH_EMBEDDING_AND_RETRIEVAL": "True",
        "WEB_FETCH_MAX_CONTENT_LENGTH": str(c.fetch_max_chars),
        "ENABLE_RAG_LOCAL_WEB_FETCH": "False",  # SSRF guard: never fetch private/loopback addresses
        "ENABLE_WEB_LOADER_SSL_VERIFICATION": "True",
        # the only way out is Tor's CONNECT tunnel (https only); the model and SearXNG are direct
        **dict.fromkeys(("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"), a.tunnel),
        **dict.fromkeys(("NO_PROXY", "no_proxy"), f"{a.relay},{a.searxng},127.0.0.1,localhost"),
        # env is the source of truth: MFA refuses to start without persistent config
        # (utils/mfa.py), so the config table is wiped at every start and re-seeded from env.
        # Admin-UI edits last until the next restart, as they did in memory before.
        "ENABLE_PERSISTENT_CONFIG": "True",
        "RESET_CONFIG_ON_START": "True",
    }


HARDENING = ["--security-opt", "no-new-privileges", "--cap-drop", "ALL", "--restart", "always"]


@dataclass
class ContainerState:
    name: str
    exists: bool
    running: bool = False
    status: str = ""
    image: str = ""
    mem_usage: str = ""
    networks: tuple[str, ...] = ()


@dataclass(frozen=True)
class Probe:
    name: str
    ok: bool
    detail: str


def http_get(url: str, timeout: float = 60) -> tuple[bool | None, str]:
    """GET from this Mac the way a browser here would, never through a proxy: (True, body) or
    (False, why it failed)."""
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(url, timeout=timeout) as r:
            return True, r.read(2_000_000).decode(errors="replace")
    except (OSError, http.client.HTTPException) as e:  # URLError, HTTPError and timeouts are OSErrors
        return False, f"{type(e).__name__}: {e}"


def tor_bootstrap(log: str) -> int | None:
    """Latest `Bootstrapped N%` in Tor's log, or None if it hasn't started bootstrapping."""
    found = re.findall(r"Bootstrapped (\d+)%", log)
    return int(found[-1]) if found else None


class Stack:
    def __init__(
        self,
        cfg: StackConfig,
        paths: Paths,
        runner: Runner = run,
        *,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        fetch: Callable[[str], tuple[bool | None, str]] = http_get,
    ):
        self.cfg, self.paths, self.run = cfg, paths, runner
        self.sleep, self.clock, self.fetch = sleep, clock, fetch
        self.addr = Addresses.of(cfg)

    def pm(self, *args: str, check: bool = True) -> Result:
        r = self.run([self.cfg.podman, *args])
        if check and not r.ok:
            raise RuntimeError(f"podman {' '.join(args[:3])} failed ({r.code}): {(r.err or r.out).strip()[-400:]}")
        return r

    # -- files and secrets -------------------------------------------------------------------

    @property
    def searxng_settings(self) -> Path:
        return self.paths.config_dir / "searxng" / "settings.yml"

    @property
    def torrc(self) -> Path:
        return self.paths.config_dir / "tor" / "torrc"

    def _secret_file(self, name: str) -> Path:
        return self.paths.config_dir / f"{name}.secret"

    def _ensure_files(self) -> None:
        for path, text in ((self.searxng_settings, searxng_settings(self.cfg)), (self.torrc, torrc(self.cfg))):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
        for name in ("webui", "searxng"):
            f = self._secret_file(name)
            if not f.exists():
                keys.write(f, secrets.token_hex(32))

    def _sync_configs(self) -> None:
        """Generated configs travel as podman secrets (the client uploads them), never as bind
        mounts, so the VM doesn't need to see the Mac's disk."""
        for name, src in ((CONFIGS["torrc"], self.torrc), (CONFIGS["searxng"], self.searxng_settings)):
            self.pm("secret", "rm", name, check=False)
            self.pm("secret", "create", name, str(src))

    def _sync_secrets(self) -> None:
        """(Re)create podman secrets from the 0600 files, so `lmtune key rotate` propagates."""
        api = self.paths.api_key
        if not api.exists():
            raise RuntimeError(f"{api} missing; run `lmtune key init`")
        sources = {
            SECRETS["api"]: api,
            SECRETS["webui"]: self._secret_file("webui"),
            SECRETS["searxng"]: self._secret_file("searxng"),
        }
        staging = self.paths.state_dir / "secrets"
        staging.mkdir(parents=True, exist_ok=True)
        staging.chmod(0o700)
        for name, src in sources.items():
            # podman stores secrets byte-for-byte; a trailing newline would end up in the HTTP header
            tmp = staging / name
            keys.write(tmp, keys.read(src))
            try:
                self.pm("secret", "rm", name, check=False)
                self.pm("secret", "create", name, str(tmp))
            finally:
                tmp.unlink(missing_ok=True)

    # -- images ------------------------------------------------------------------------------

    def _image_ref(self, image: str) -> str:
        """Exactly the local image verified: `repo@sha256:…` for registry images, the image ID for
        local builds (which have no registry digest to pin)."""
        local = image.startswith("localhost/")
        r = self.pm("image", "inspect", image, "--format", "{{.Id}}" if local else "{{.Digest}}", check=False)
        ref = r.out.strip()
        if not r.ok or not ref:
            hint = "build it with the sandbox" if local else "run `lmtune stack update`"
            raise RuntimeError(f"image {image} not present locally; {hint}")
        if local:
            return ref
        if not ref.startswith("sha256:"):
            raise RuntimeError(f"image {image} has no digest; run `lmtune stack update`")
        return f"{image.rsplit(':', 1)[0]}@{ref}"

    def update(self) -> list[str]:
        out = []
        for image in (self.cfg.webui_image, self.cfg.searxng_image):
            self.pm("pull", image)
            out.append(self._image_ref(image))
        out.append(f"{self._image_ref(self.cfg.tor_image)} ({self.cfg.tor_image}, local build: not pulled)")
        return out

    # -- networks ----------------------------------------------------------------------------

    def network_problems(self, name: str, subnet: str | None) -> list[str]:
        """How an existing network falls short of what lmtune creates (internal ones must leave
        no route out and no resolver besides Tor)."""
        r = self.pm("network", "inspect", name, check=False)
        if not r.ok:
            return [f"{name} is missing"]
        try:
            d: Any = json.loads(r.out)[0]
        except json.JSONDecodeError, IndexError:
            return [f"can't inspect {name}"]
        if subnet is None:
            return []
        out = []
        if d.get("internal") is not True:
            out.append(f"{name} isn't internal, so it routes to the internet")
        if d.get("dns_enabled") is not False:
            out.append(f"{name} runs podman's DNS server")
        if d.get("ipv6_enabled") is not False:
            out.append(f"{name} has IPv6 enabled")
        subnets = d.get("subnets") or []
        if [s.get("subnet") for s in subnets] != [subnet]:
            out.append(f"{name} isn't {subnet}")
        if any(s.get("gateway") for s in subnets):
            out.append(f"{name} gives the VM an address on it")
        return out

    def _ensure_networks(self) -> None:
        specs: list[tuple[str, str | None]] = [
            (NET_APP, self.cfg.app_subnet),
            (NET_WEB, self.cfg.web_subnet),
            (self.cfg.network, None),
            (NET_EGRESS, None),
        ]
        for name, subnet in specs:
            found = self.network_problems(name, subnet)
            if not found:
                continue
            if found != [f"{name} is missing"]:
                self.pm("network", "rm", name)  # wrong shape; containers on it were removed first
            internal = ["--internal", "--disable-dns", "--subnet", subnet] if subnet else []
            self.pm("network", "create", *internal, name)

    # -- containers --------------------------------------------------------------------------

    def gateway_args(self) -> list[str]:
        c, a = self.cfg, self.addr
        bridges = ["-v", f"{c.tor_bridges_volume}:/etc/tor/conf.d:ro"] if c.tor_bridges_volume else []
        caps = [arg for cap in TOR_CAPABILITIES for arg in ("--cap-add", cap)]
        return [
            "run",
            "-d",
            "--name",
            GATEWAY,
            "--network",
            f"{NET_WEB}:ip={a.gateway}",
            "--network",
            NET_EGRESS,
            "--dns-search",
            ".",
            "--read-only",
            *HARDENING,
            *caps,
            "--memory",
            c.tor_memory,
            "--pids-limit",
            "64",
            "-v",
            f"{TOR_DATA}:/var/lib/tor",
            *bridges,
            "--secret",
            f"{CONFIGS['torrc']},type=mount,target=/etc/lmtune/torrc,mode=0444",
            "--entrypoint",
            "/bin/sh",
            self._image_ref(c.tor_image),
            "-c",
            TOR_START,
        ]

    def relay_args(self) -> list[str]:
        c, a = self.cfg, self.addr
        return [
            "run",
            "-d",
            "--name",
            RELAY,
            "--network",
            c.network,
            "--network",
            f"{NET_APP}:ip={a.relay}",
            "--read-only",
            *HARDENING,
            "--memory",
            c.relay_memory,
            "--pids-limit",
            "32",
            "--user",
            "65534:65534",
            "--entrypoint",
            "python3",
            self._image_ref(c.searxng_image),
            "-I",
            "-c",
            RELAY_PY,
            a.relay,
            str(c.server_port),
            "host.containers.internal",
            str(c.server_port),
        ]

    def searxng_args(self) -> list[str]:
        c, a = self.cfg, self.addr
        # host loopback only, and only when asked for; it stays on the internal network either way
        published = ["-p", f"127.0.0.1:{c.searxng_port}:8080"] if c.searxng_port else []
        return [
            "run",
            "-d",
            "--name",
            SEARXNG,
            "--network",
            f"{NET_WEB}:ip={a.searxng}",
            "--dns",
            "none",  # names are resolved by Tor (socks5h); there is no resolver here at all
            *NO_IPV6,
            *HARDENING,
            "--memory",
            c.searxng_memory,
            "--pids-limit",
            "256",
            "-e",
            "GRANIAN_HOST=0.0.0.0",  # the image binds `::`, which doesn't exist without IPv6
            *published,
            "--secret",
            f"{CONFIGS['searxng']},type=mount,target=/etc/searxng/settings.yml,mode=0444",
            "--secret",
            f"{SECRETS['searxng']},type=env,target=SEARXNG_SECRET",
            self._image_ref(c.searxng_image),
        ]

    def webui_args(self) -> list[str]:
        c, a = self.cfg, self.addr
        env = [arg for k, v in webui_env(c).items() for arg in ("-e", f"{k}={v}")]
        return [
            "run",
            "-d",
            "--name",
            WEBUI,
            "--network",
            f"{NET_APP}:ip={a.webui_app}",
            "--network",
            f"{NET_WEB}:ip={a.webui_web}",
            "--dns",
            a.gateway,  # Tor's DNSPort: the SSRF guard resolves before fetching, through Tor
            "--dns-search",
            ".",  # no search domain (the Mac's tailnet name would be appended and sent to Tor)
            *NO_IPV6,
            *HARDENING,
            "--memory",
            c.webui_memory,
            "--pids-limit",
            "512",
            "-p",
            f"127.0.0.1:{c.webui_port}:8080",  # host loopback only
            "-v",
            "lmtune-webui-data:/app/backend/data",
            "--secret",
            f"{SECRETS['api']},type=env,target=OPENAI_API_KEYS",
            "--secret",
            f"{SECRETS['webui']},type=env,target=WEBUI_SECRET_KEY",
            *env,
            self._image_ref(c.webui_image),
        ]

    # -- lifecycle ---------------------------------------------------------------------------

    def ensure_machine(self) -> None:
        r = self.pm("machine", "inspect", check=False)
        try:
            state = json.loads(r.out)[0].get("State", "") if r.ok else ""
        except json.JSONDecodeError, IndexError:
            state = ""
        if state != "running":
            self.pm("machine", "start")

    def vm_shares(self) -> list[str] | None:
        """Folders of the Mac mounted into the podman VM (podman's default shares /Users,
        /private, /var/folders and ~/.config/containers read-write). None: couldn't tell."""
        r = self.pm("machine", "ssh", "findmnt -t virtiofs -n -o TARGET", check=False)
        return [line.strip() for line in r.out.splitlines() if line.strip()] if r.ok else None

    def wait_for_tor(self, timeout_s: float, since: float | None = None) -> int | None:
        """Wait for `Bootstrapped 100%`, reading only log lines written after `since` (a Unix
        time) so a restarted gateway isn't judged by its previous run."""
        deadline = self.clock() + timeout_s
        while True:
            pct = tor_bootstrap(self.logs(GATEWAY, 200, since=since))
            if pct == 100 or self.clock() >= deadline:
                return pct
            self.sleep(2)

    def up(self) -> list[ContainerState]:
        if errs := problems(self.cfg):
            raise RuntimeError("; ".join(errs))
        self.ensure_machine()
        self._ensure_files()
        self._sync_secrets()
        self._sync_configs()
        for name in CONTAINERS:  # recreate: picks up config, secrets and pins; frees the networks
            self.pm("rm", "-f", "-v", "-t", "10", name, check=False)
        self._ensure_networks()
        self.pm(*self.gateway_args())
        self.pm(*self.relay_args())
        # SearXNG checks at startup that its proxy is Tor and exits if it can't tell yet
        self.wait_for_tor(self.cfg.tor_bootstrap_s)
        self.pm(*self.searxng_args())
        self.pm(*self.webui_args())
        return self.status()

    def down(self) -> None:
        for name in CONTAINERS:
            self.pm("rm", "-f", "-v", "-t", "10", name, check=False)

    def status(self) -> list[ContainerState]:
        out = []
        stats = self._stats()
        for name in CONTAINERS:
            r = self.pm("container", "inspect", name, check=False)
            if not r.ok:
                out.append(ContainerState(name, exists=False))
                continue
            d: Any = json.loads(r.out)[0]
            st = d.get("State", {})
            nets = d.get("NetworkSettings", {}).get("Networks") or {}
            out.append(
                ContainerState(
                    name,
                    True,
                    bool(st.get("Running")),
                    str(st.get("Status", "")),
                    str(d.get("ImageName", "")),
                    stats.get(name, ""),
                    tuple(sorted(nets)),
                )
            )
        return out

    def expected_networks(self) -> dict[str, tuple[str, ...]]:
        return {
            WEBUI: tuple(sorted((NET_APP, NET_WEB))),
            SEARXNG: (NET_WEB,),
            RELAY: tuple(sorted((self.cfg.network, NET_APP))),
            GATEWAY: tuple(sorted((NET_WEB, NET_EGRESS))),
        }

    def _stats(self) -> dict[str, str]:
        r = self.pm("stats", "--no-stream", "--format", "{{.Name}}\t{{.MemUsage}}", check=False)
        return dict(line.split("\t", 1) for line in r.out.splitlines() if "\t" in line) if r.ok else {}

    # -- verification ------------------------------------------------------------------------

    def probe(self, container: str, mode: str, *args: str) -> tuple[bool | None, str]:
        """(True, …) worked, (False, …) was refused/unreachable, (None, …) the probe itself
        couldn't run, which never counts as either."""
        r = self.pm("exec", container, "python3", "-I", "-c", PROBE_PY, mode, *args, check=False)
        try:
            d = json.loads(r.out.strip().splitlines()[-1])
            return bool(d["ok"]), str(d["detail"])
        except json.JSONDecodeError, IndexError, KeyError, TypeError:
            return None, f"probe didn't run ({r.code}): {(r.err or r.out).strip()[-200:]}"

    def _search(self, query: str, *, published: bool = False) -> tuple[int | None, list[str]]:
        """(result count, engines that failed) for a SearXNG query, sent the way Open WebUI sends
        it (its own proxy settings, so NO_PROXY must keep SearXNG direct), or with `published`
        from this Mac to the loopback port, the way your browser sends it. None: didn't run."""
        path = f"/search?format=json&q={query.replace(' ', '+')}"
        if published:
            ok, body = self.fetch(f"http://127.0.0.1:{self.cfg.searxng_port}{path}")
        else:
            ok, body = self.probe(WEBUI, "get", f"http://{self.addr.searxng}:8080{path}", "env")
        if not ok:
            return (0 if ok is False else None), [body]
        try:
            d = json.loads(body)
        except json.JSONDecodeError:
            return 0, ["unparseable SearXNG answer"]
        return len(d.get("results", [])), [str(e[0]) for e in d.get("unresponsive_engines", [])]

    def _searches(
        self, query: str, attempts: int, *, until_found: bool, published: bool = False
    ) -> tuple[list[int | None], list[str]]:
        """Result counts of up to `attempts` searches, 5 s apart. SearXNG sends every engine over
        one Tor circuit, so one slow exit fails a whole query: a working path is allowed retries
        (`until_found`), and a path that must be closed has to fail every attempt."""
        counts: list[int | None] = []
        failed: list[str] = []
        for i in range(attempts):
            if i:
                self.sleep(5)
            n, failed = self._search(f"{query} {i}" if i else query, published=published)
            counts.append(n)
            if until_found and n:
                break
        return counts, failed

    def _fetch_exit(self) -> tuple[str, str]:
        """How a page fetch made the way Open WebUI makes it (its proxy settings) gets out:
        "tor", "other" (a leak), "blocked", or "error" (the probe didn't run)."""
        ok, body = self.probe(WEBUI, "get", "https://check.torproject.org/api/ip", "env")
        if not ok:
            return ("blocked" if ok is False else "error"), body
        try:
            return ("tor" if json.loads(body)["IsTor"] is True else "other"), body.strip()
        except json.JSONDecodeError, KeyError, TypeError:
            return "other", body[:200]

    def verify(self, *, fail_closed: bool = False) -> list[Probe]:
        """Probe the properties the design promises, from inside the containers. Each "must be
        blocked" probe has a "must work" twin on the same path, so a probe that can't run can't
        pass as isolation."""
        a, port = self.addr, str(self.cfg.server_port)
        # by address: a name the probing container can't resolve would "block" for the wrong reason
        resolved, host = self.probe(RELAY, "resolve", "host.containers.internal")
        host_ip = host.split(" ")[0] if resolved else "host.containers.internal"
        out: list[Probe] = []

        def expect(name: str, want: bool, result: tuple[bool | None, str]) -> None:
            ok, detail = result
            label = {True: "works: ", False: "blocked: ", None: "error: "}[ok]
            out.append(Probe(name, ok is want, label + detail[:160]))

        pct = tor_bootstrap(self.logs(GATEWAY, 200))
        out.append(Probe("tor bootstrapped", pct == 100, f"{pct}%" if pct is not None else "no bootstrap in log"))
        expect("relay → host model server", True, self.probe(RELAY, "connect", host_ip, port))
        expect("webui → model via relay", True, self.probe(WEBUI, "get", f"http://{a.relay}:{port}/health", "env"))
        expect("webui → Tor gateway", True, self.probe(WEBUI, "connect", a.gateway, str(TUNNEL_PORT)))
        expect("webui → host directly", False, self.probe(WEBUI, "connect", host_ip, port))
        expect("webui → internet directly", False, self.probe(WEBUI, "connect", "1.1.1.1", "443"))
        expect("searxng → Tor gateway", True, self.probe(SEARXNG, "connect", a.gateway, str(SOCKS_PORT)))
        expect("searxng → host directly", False, self.probe(SEARXNG, "connect", host_ip, port))
        expect("searxng → internet directly", False, self.probe(SEARXNG, "connect", "1.1.1.1", "443"))
        expect("webui resolves only via Tor", True, self.probe(WEBUI, "resolve", "check.torproject.org"))
        how, detail = self._fetch_exit()
        out.append(Probe("webui page fetch exits via Tor", how == "tor", f"{how}: {detail[:150]}"))

        def found(counts: list[int | None], failed: list[str]) -> str:
            return f"{counts[-1]} results (try {len(counts)}); unresponsive: {', '.join(failed)[:120] or 'none'}"

        counts, failed = self._searches("tor project", 3, until_found=True)
        out.append(Probe("search via Tor", bool(counts[-1]), found(counts, failed)))
        published = bool(self.cfg.searxng_port)
        if published:  # your browser's way in: from this Mac to the loopback port
            counts, failed = self._searches("tor browser", 3, until_found=True, published=True)
            out.append(Probe("search via published port", bool(counts[-1]), found(counts, failed)))
        if not fail_closed:
            return out

        self.pm("stop", "-t", "10", GATEWAY)
        try:
            counts, failed = self._searches("onion routing", 3, until_found=False)
            closed = all(n == 0 for n in counts)
            out.append(Probe("gateway down: search fails", closed, f"{counts} results; {', '.join(failed)[:120]}"))
            if published:
                counts, failed = self._searches("hidden services", 3, until_found=False, published=True)
                closed = all(n == 0 for n in counts)
                detail = f"{counts} results; {', '.join(failed)[:120]}"
                out.append(Probe("gateway down: published search fails", closed, detail))
            how, detail = self._fetch_exit()
            out.append(Probe("gateway down: page fetch fails", how == "blocked", f"{how}: {detail[:150]}"))
            expect("gateway down: no DNS", False, self.probe(WEBUI, "resolve", "check.torproject.org"))
            expect("gateway down: no direct route", False, self.probe(WEBUI, "connect", "1.1.1.1", "443"))
        finally:
            since = time.time()
            self.pm("start", GATEWAY)
        pct = self.wait_for_tor(self.cfg.tor_bootstrap_s, since=since)
        counts, failed = self._searches("free software", 3, until_found=True)
        back = pct == 100 and bool(counts[-1])
        out.append(Probe("gateway back: search works", back, f"tor {pct}%, {counts[-1]} results (try {len(counts)})"))
        return out

    # -- login persistence ---------------------------------------------------------------------

    @property
    def agent_plist(self) -> Path:
        return self.paths.launch_agents / f"{STACK_LABEL}.plist"

    def install(self) -> Path:
        """Run `lmtune stack up` once at every login (the podman VM doesn't start by itself)."""
        exe = shutil.which("lmtune") or str(Path.home() / ".local/bin/lmtune")
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        self.agent_plist.parent.mkdir(parents=True, exist_ok=True)
        self.agent_plist.write_bytes(
            plist(STACK_LABEL, [exe, "stack", "up"], self.paths.log_dir / "stack.log", keep_alive=False)
        )
        ld = Launchd(STACK_LABEL, self.run)
        if ld.state().loaded:
            ld.bootout()
        ld.bootstrap(self.agent_plist)
        return self.agent_plist

    def uninstall(self) -> None:
        Launchd(STACK_LABEL, self.run).bootout()
        self.agent_plist.unlink(missing_ok=True)

    def logs(self, name: str, n: int, since: float | None = None) -> str:
        window = ["--since", str(int(since))] if since is not None else []
        r = self.pm("logs", "--tail", str(n), *window, name, check=False)
        return r.out + r.err
