"""Open WebUI + SearXNG + Tor stack: generated podman commands, configs, secret handling and the
verify logic (podman faked)."""

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from lmtune import doctor, keys, stack
from lmtune.paths import Paths
from lmtune.proc import Result

GOOD_NET = {"internal": True, "dns_enabled": False, "ipv6_enabled": False}
ADDR = stack.Addresses.of(stack.StackConfig())


def probe_ok(detail: str = "connected") -> Result:
    return Result(0, json.dumps({"ok": True, "detail": detail}) + "\n", "")


def probe_blocked(detail: str = "OSError: [Errno 101] Network is unreachable") -> Result:
    return Result(0, json.dumps({"ok": False, "detail": detail}) + "\n", "")


SEARCH = json.dumps({"results": [{"url": "https://a"}, {"url": "https://b"}], "unresponsive_engines": [["ddg", "x"]]})


def healthy_network(argv: list[str]) -> Result:
    """What the probes see when the stack is built as designed and Tor is up."""
    container, mode, args = argv[1], argv[6], argv[7:]  # exec <c> python3 -I -c <script> <mode> …
    if mode == "resolve":
        return probe_ok("169.254.1.2" if args[0] == "host.containers.internal" else "116.202.120.181")
    if mode == "connect":
        target = args[0]
        if container == stack.RELAY or target == ADDR.gateway:
            return probe_ok()
        return probe_blocked()
    if mode == "get":
        url = args[0]
        if "/health" in url:
            return probe_ok('{"status":"ok"}')
        if "check.torproject.org" in url:
            return probe_ok('{"IsTor":true,"IP":"192.0.2.1"}')
        if "/search" in url:
            return probe_ok(SEARCH)
    return Result(1, "", "unexpected probe")


class FakePodman:
    def __init__(self, network: Callable[[list[str]], Result] = healthy_network):
        self.calls: list[list[str]] = []
        self.secret_bytes: dict[str, bytes] = {}
        self.networks: dict[str, dict[str, object]] = {}
        self.network = network
        self.tor_log = "Bootstrapped 5% (conn)\nBootstrapped 100% (done): Done\n"
        self.published: dict[str, str] = {stack.WEBUI: "8080/tcp -> 127.0.0.1:3000"}
        self.attached: dict[str, list[str]] = {}
        self.shares: list[str] = []

    def __call__(self, cmd: Sequence[str]) -> Result:
        c = list(cmd)[1:]
        self.calls.append(c)
        match c[:2]:
            case ["secret", "create"]:
                self.secret_bytes[c[2]] = Path(c[3]).read_bytes()
            case ["image", "inspect"]:
                if c[-1] == "{{.Id}}":
                    return Result(0, "f" * 64 + "\n", "")
                return Result(0, "sha256:" + "a" * 64 + "\n", "")
            case ["machine", "inspect"]:
                return Result(0, json.dumps([{"State": "running"}]), "")
            case ["machine", "ssh"]:
                return Result(0, "".join(f"{m}\n" for m in self.shares), "")
            case ["network", "inspect"]:
                net = self.networks.get(c[2])
                return Result(0, json.dumps([net]), "") if net is not None else Result(125, "", "no such network")
            case ["network", "create"]:
                name = c[-1]
                subnet = c[c.index("--subnet") + 1] if "--subnet" in c else None
                self.networks[name] = {
                    "internal": "--internal" in c,
                    "dns_enabled": "--disable-dns" not in c,
                    "ipv6_enabled": False,
                    "subnets": [{"subnet": subnet}] if subnet else [{"subnet": "10.89.0.0/24", "gateway": "10.89.0.1"}],
                }
            case ["network", "rm"]:
                self.networks.pop(c[2], None)
            case ["container", "inspect"]:
                name = c[2]
                nets = self.attached.get(
                    name, list(stack.Stack(stack.StackConfig(), _paths()).expected_networks()[name])
                )
                return Result(
                    0,
                    json.dumps(
                        [
                            {
                                "State": {"Running": True, "Status": "running"},
                                "ImageName": "x",
                                "NetworkSettings": {"Networks": {n: {} for n in nets}},
                            }
                        ]
                    ),
                    "",
                )
            case ["logs", *_]:
                return Result(0, self.tor_log, "")
            case ["port", name]:
                return Result(0, self.published.get(name, ""), "")
            case ["exec", *_]:
                return self.network(c)
        return Result(0, "", "")

    def run_args(self, name: str) -> list[str]:
        return next(c for c in self.calls if c[:1] == ["run"] and name in c)

    def run_order(self) -> list[str]:
        return [c[c.index("--name") + 1] for c in self.calls if c[:1] == ["run"]]


def _paths(root: Path = Path("/nonexistent")) -> Paths:
    return Paths(root / "cfg", root / "state", root / "logs", root / "agents")


def pairs(args: list[str]) -> dict[str, str]:
    return dict(args[i + 1].split("=", 1) for i, a in enumerate(args) if a == "-e")


def opt_values(args: list[str], flag: str) -> list[str]:
    return [args[i + 1] for i, a in enumerate(args) if a == flag]


@pytest.fixture
def env(tmp_path):
    paths = _paths(tmp_path)
    keys.write(paths.api_key, keys.generate())
    fake = FakePodman()
    st = stack.Stack(stack.StackConfig(tailnet_host="box.tail.ts.net"), paths, fake, sleep=lambda _s: None)
    st.up()
    return st, fake, paths


# -- secrets, publishing -------------------------------------------------------------------------


def test_secrets_are_exact_bytes_and_never_in_argv(env):
    _, fake, paths = env
    key = keys.read(paths.api_key)
    assert fake.secret_bytes[stack.SECRETS["api"]] == key.encode()  # no trailing newline
    assert all(not fake.secret_bytes[name].endswith(b"\n") for name in stack.SECRETS.values())  # headers
    flat = " ".join(" ".join(c) for c in fake.calls)
    assert key not in flat  # never on a command line
    assert not any((paths.state_dir / "secrets").iterdir())  # staging copies removed


def test_only_webui_is_published_and_only_on_loopback(env):
    _, fake, _ = env
    for name in stack.CONTAINERS:
        published = opt_values(fake.run_args(name), "-p")
        assert published == (["127.0.0.1:3000:8080"] if name == stack.WEBUI else []), name


def test_searxng_is_published_on_loopback_only_when_asked(tmp_path):
    paths = _paths(tmp_path)
    keys.write(paths.api_key, keys.generate())
    fake = FakePodman()
    stack.Stack(stack.StackConfig(searxng_port=8888), paths, fake, sleep=lambda _s: None).up()
    expected = {stack.WEBUI: ["127.0.0.1:3000:8080"], stack.SEARXNG: ["127.0.0.1:8888:8080"]}
    for name in stack.CONTAINERS:
        assert opt_values(fake.run_args(name), "-p") == expected.get(name, []), name
    args = fake.run_args(stack.SEARXNG)  # still only on the internal network, still no resolver
    assert opt_values(args, "--network") == [f"{stack.NET_WEB}:ip={ADDR.searxng}"]
    assert opt_values(args, "--dns") == ["none"]


# -- networks ------------------------------------------------------------------------------------


def test_internal_networks_have_no_route_dns_or_sandbox_overlap(env):
    _, fake, _ = env
    for name, subnet in ((stack.NET_APP, "10.153.1.0/24"), (stack.NET_WEB, "10.153.2.0/24")):
        create = next(c for c in fake.calls if c[:2] == ["network", "create"] and c[-1] == name)
        assert "--internal" in create and "--disable-dns" in create
        assert create[create.index("--subnet") + 1] == subnet
    routed = [c[-1] for c in fake.calls if c[:2] == ["network", "create"] and "--internal" not in c]
    assert sorted(routed) == sorted(["lmtune", stack.NET_EGRESS])


def test_wrong_shaped_network_is_recreated(tmp_path):
    paths = _paths(tmp_path)
    keys.write(paths.api_key, keys.generate())
    fake = FakePodman()
    # someone created lmtune-web as a routed network with DNS: it must not be trusted
    fake.networks[stack.NET_WEB] = {"internal": False, "dns_enabled": True, "ipv6_enabled": False, "subnets": []}
    fake.networks[stack.NET_APP] = {**GOOD_NET, "subnets": [{"subnet": "10.153.1.0/24"}]}
    st = stack.Stack(stack.StackConfig(), paths, fake, sleep=lambda _s: None)
    st.up()
    assert ["network", "rm", stack.NET_WEB] in fake.calls
    assert ["network", "rm", stack.NET_APP] not in fake.calls  # already right: left alone
    assert st.network_problems(stack.NET_WEB, "10.153.2.0/24") == []


def test_network_problems_names_each_escape():
    fake = FakePodman()
    fake.networks["n"] = {
        "internal": False,
        "dns_enabled": True,
        "ipv6_enabled": True,
        "subnets": [{"subnet": "10.9.9.0/24", "gateway": "10.9.9.1"}],
    }
    found = stack.Stack(stack.StackConfig(), _paths(), fake).network_problems("n", "10.153.2.0/24")
    assert len(found) == 5
    assert any("internet" in p for p in found) and any("DNS" in p for p in found)
    assert any("IPv6" in p for p in found) and any("VM an address" in p for p in found)


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"web_subnet": "10.152.7.0/24"}, "sandbox"),
        ({"app_subnet": "10.153.2.0/24"}, "overlap"),
        ({"web_subnet": "8.8.8.0/24"}, "private"),
        ({"app_subnet": "nope"}, "not a network"),
        ({"fetch_max_chars": 0}, "fetch_max_chars"),
        ({"searxng_port": 80}, "1024-65535"),
        ({"searxng_port": 3000}, "already Open WebUI's"),
        ({"searxng_port": 8080}, "already Open WebUI's or the model server's"),
        ({"searxng_https_port": 8443}, "already Open WebUI's"),
    ],
)
def test_bad_config_rejected(tmp_path, overrides, fragment):
    assert any(fragment in p for p in stack.problems(stack.StackConfig(**overrides)))
    p = tmp_path / "stack.toml"
    p.write_text("[stack]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in overrides.items()))
    with pytest.raises(ValueError, match=fragment):
        stack.load(p)


def test_unknown_stack_keys_rejected(tmp_path):
    p = tmp_path / "stack.toml"
    p.write_text("[stack]\nbogus = 1\n")
    with pytest.raises(ValueError, match="bogus"):
        stack.load(p)


# -- containers ----------------------------------------------------------------------------------


def test_gateway_starts_before_searxng(env):
    _, fake, _ = env
    order = fake.run_order()
    assert order.index(stack.GATEWAY) < order.index(stack.SEARXNG)  # SearXNG checks Tor at startup


def test_webui_reaches_only_internal_networks_and_tor(env):
    _, fake, _ = env
    args = fake.run_args(stack.WEBUI)
    assert opt_values(args, "--network") == [
        f"{stack.NET_APP}:ip={ADDR.webui_app}",
        f"{stack.NET_WEB}:ip={ADDR.webui_web}",
    ]
    assert opt_values(args, "--dns") == [ADDR.gateway]  # Tor's DNSPort is the only resolver
    assert opt_values(args, "--dns-search") == ["."]
    assert "net.ipv6.conf.all.disable_ipv6=1" in opt_values(args, "--sysctl")
    e = pairs(args)
    assert {e[k] for k in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy")} == {ADDR.tunnel}
    no_proxy = e["NO_PROXY"].split(",")
    assert ADDR.relay in no_proxy and ADDR.searxng in no_proxy and e["no_proxy"] == e["NO_PROXY"]
    assert e["OPENAI_API_BASE_URLS"] == f"http://{ADDR.relay}:8080/v1"
    assert e["SEARXNG_QUERY_URL"].startswith(f"http://{ADDR.searxng}:8080/")
    assert e["WEB_FETCH_MAX_CONTENT_LENGTH"] == "20000"


def test_webui_locked_down(env):
    _, fake, _ = env
    args = fake.run_args(stack.WEBUI)
    joined = " ".join(args)
    assert "--cap-drop ALL" in joined and "no-new-privileges" in joined
    assert "@sha256:" in args[-1]  # pinned by digest
    e = pairs(args)
    for k, v in {
        "ENABLE_SIGNUP": "False",
        "OFFLINE_MODE": "True",
        "ENABLE_OLLAMA_API": "False",
        "ENABLE_DIRECT_CONNECTIONS": "False",
        "ENABLE_CODE_INTERPRETER": "False",
        "ENABLE_MFA": "True",
        # MFA refuses to start without persistent config; the reset keeps env the source of truth
        "ENABLE_PERSISTENT_CONFIG": "True",
        "RESET_CONFIG_ON_START": "True",
        "ENABLE_RAG_LOCAL_WEB_FETCH": "False",
        "WEBUI_URL": "https://box.tail.ts.net:8443",
    }.items():
        assert e.get(k) == v, k
    assert "OPENAI_API_KEYS" not in e  # arrives as a secret
    # regression: localhost access broke websockets when only the tailnet URL was allowed
    assert e["CORS_ALLOW_ORIGIN"].split(";") == [
        "https://box.tail.ts.net:8443",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
    ]


def test_searxng_has_no_resolver_and_only_the_web_network(env):
    _, fake, _ = env
    args = fake.run_args(stack.SEARXNG)
    assert opt_values(args, "--network") == [f"{stack.NET_WEB}:ip={ADDR.searxng}"]
    assert opt_values(args, "--dns") == ["none"]
    assert "net.ipv6.conf.all.disable_ipv6=1" in opt_values(args, "--sysctl")
    assert pairs(args)["GRANIAN_HOST"] == "0.0.0.0"
    assert any("SEARXNG_SECRET" in a for a in args)
    assert "lmtune-searxng-settings,type=mount,target=/etc/searxng/settings.yml,mode=0444" in opt_values(
        args, "--secret"
    )


def test_nothing_from_the_macs_disk_is_mounted(env):
    """Configs arrive as podman secrets, so the VM needs no host folder shares."""
    _, fake, _ = env
    for name in stack.CONTAINERS:
        sources = [v.split(":")[0] for v in opt_values(fake.run_args(name), "-v")]
        assert not any(src.startswith("/") for src in sources), (name, sources)
    created = {c[2]: Path(c[3]) for c in fake.calls if c[:2] == ["secret", "create"]}
    assert created["lmtune-torrc"].read_text().startswith("# Generated by lmtune")
    assert "socks5h://" in created["lmtune-searxng-settings"].read_text()


def test_recreating_containers_drops_their_anonymous_volumes(env):
    st, fake, _ = env
    st.down()
    assert ["rm", "-f", "-v", "-t", "10", stack.SEARXNG] in fake.calls


def test_gateway_minimal_privileges_and_readonly_bridges(env):
    _, fake, _ = env
    args = fake.run_args(stack.GATEWAY)
    assert opt_values(args, "--network") == [f"{stack.NET_WEB}:ip={ADDR.gateway}", stack.NET_EGRESS]
    assert sorted(opt_values(args, "--cap-add")) == sorted(stack.TOR_CAPABILITIES)
    assert "--read-only" in args and "--cap-drop" in args
    volumes = opt_values(args, "-v")
    assert "anon-tor-conf:/etc/tor/conf.d:ro" in volumes
    assert opt_values(args, "--secret") == ["lmtune-torrc,type=mount,target=/etc/lmtune/torrc,mode=0444"]
    assert f"{stack.TOR_DATA}:/var/lib/tor" in volumes  # its own state, not the sandbox's
    assert args[args.index("--entrypoint") + 1] == "/bin/sh"
    assert args[args.index("--entrypoint") + 2] == "f" * 64  # local build pinned by image ID


def test_relay_is_the_only_container_on_the_routed_network(env):
    st, fake, _ = env
    for name in stack.CONTAINERS:
        nets = [n.split(":")[0] for n in opt_values(fake.run_args(name), "--network")]
        assert ("lmtune" in nets) == (name == stack.RELAY), name
    args = fake.run_args(stack.RELAY)
    assert opt_values(args, "--user") == ["65534:65534"] and "--read-only" in args
    assert args[-4:] == [ADDR.relay, "8080", "host.containers.internal", "8080"]
    assert st.expected_networks()[stack.RELAY] == ("lmtune", stack.NET_APP)


def test_no_bridges_volume_means_no_include(tmp_path):
    cfg = stack.StackConfig(tor_bridges_volume="")
    assert "%include" not in stack.torrc(cfg)
    paths = _paths(tmp_path)
    args = stack.Stack(cfg, paths, FakePodman()).gateway_args()
    assert not any("conf.d" in v for v in opt_values(args, "-v"))


# -- generated configs ---------------------------------------------------------------------------


def test_torrc_listens_only_on_the_web_network():
    rc = stack.torrc(stack.StackConfig())
    lines = [line for line in rc.splitlines() if line and not line.startswith("#")]
    listeners = [line for line in lines if line.split()[0] in ("SocksPort", "HTTPTunnelPort", "DNSPort")]
    assert [line.split()[1] for line in listeners] == [
        f"{ADDR.gateway}:9050",
        f"{ADDR.gateway}:9080",
        f"{ADDR.gateway}:53",
    ]
    assert "SocksPolicy accept 10.153.2.0/24" in lines and lines.index("SocksPolicy reject *") > lines.index(
        "SocksPolicy accept 10.153.2.0/24"
    )
    for required in ("ClientOnly 1", "SafeSocks 1", "User debian-tor", "%include /etc/tor/conf.d/*.conf"):
        assert required in lines


def test_searxng_settings_send_everything_through_tor():
    text = stack.searxng_settings(stack.StackConfig())
    assert f"socks5h://searxng:lmtune@{ADDR.gateway}:9050" in text  # remote DNS: names resolved by Tor
    assert "using_tor_proxy: true" in text  # refuses to search unless the proxy is Tor
    assert "keep_only: [brave, duckduckgo web, bing, yahoo, yep, wikidata]" in text  # only Tor-tolerant engines
    assert "radio browser" not in text  # bypassed the proxy for DNS; not kept
    for name in ("duckduckgo web", "bing", "yahoo", "yep"):  # off by default upstream
        assert f"  - name: {name}\n    disabled: false\n" in text
    assert "request_timeout: 10.0" in text
    settings = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert not any("extra_proxy_timeout" in line for line in settings)  # only ever applied to .onion engines


def test_searxng_settings_for_browsers_too():
    lines = [line.split("#")[0].rstrip() for line in stack.searxng_settings(stack.StackConfig()).splitlines()]
    server = lines[lines.index("server:") + 1 : lines.index("search:")]
    assert "  limiter: false" in server  # no Valkey here; access control is loopback + tailnet ACL
    assert "  public_instance: false" in server  # would force the limiter, and exit without Valkey
    assert "  image_proxy: true" in server  # result images over Tor, not from the browser's address


# -- lifecycle -----------------------------------------------------------------------------------


def test_status_and_down(env):
    st, fake, _ = env
    assert all(c.running for c in st.status())
    st.down()
    for name in stack.CONTAINERS:
        assert ["rm", "-f", "-v", "-t", "10", name] in fake.calls


def test_tor_bootstrap_parsing():
    assert stack.tor_bootstrap("Bootstrapped 10% (x)\nBootstrapped 85% (y)\n") == 85
    assert stack.tor_bootstrap("nothing yet") is None


def test_wait_for_tor_reads_only_the_new_run(tmp_path):
    fake = FakePodman()
    clock = iter(range(1000))
    st = stack.Stack(stack.StackConfig(), _paths(tmp_path), fake, sleep=lambda _s: None, clock=lambda: next(clock))
    assert st.wait_for_tor(10, since=1234.9) == 100
    assert any(c[:1] == ["logs"] and c[c.index("--since") + 1] == "1234" for c in fake.calls)
    fake.tor_log = "Bootstrapped 40% (loading)\n"
    assert st.wait_for_tor(5) == 40  # gives up at the deadline and says how far it got


# -- verify --------------------------------------------------------------------------------------


def verify_with(tmp_path, network, **kw):
    fake = FakePodman(network)
    st = stack.Stack(stack.StackConfig(), _paths(tmp_path), fake, sleep=lambda _s: None)
    return {p.name: p for p in st.verify(**kw)}, fake


def test_verify_passes_on_the_designed_network(tmp_path):
    probes, _ = verify_with(tmp_path, healthy_network)
    assert all(p.ok for p in probes.values()), [p for p in probes.values() if not p.ok]
    assert probes["search via Tor"].detail.startswith("2 results (try 1)")


def test_verify_catches_a_direct_route(tmp_path):
    def leaky(argv):
        if argv[6] == "connect" and argv[7] == "1.1.1.1":
            return probe_ok()
        return healthy_network(argv)

    probes, _ = verify_with(tmp_path, leaky)
    for name in ("webui → internet directly", "searxng → internet directly"):
        assert not probes[name].ok and probes[name].detail.startswith("works"), name
    assert probes["webui → host directly"].ok  # only the leak is flagged


def test_verify_catches_a_fetch_that_isnt_tor(tmp_path):
    def clearnet(argv):
        if argv[6] == "get" and "check.torproject.org" in argv[7]:
            return probe_ok('{"IsTor":false,"IP":"203.0.113.9"}')
        return healthy_network(argv)

    probes, _ = verify_with(tmp_path, clearnet)
    assert not probes["webui page fetch exits via Tor"].ok
    assert probes["webui page fetch exits via Tor"].detail.startswith("other")


def test_a_probe_that_cannot_run_never_passes_as_blocked(tmp_path):
    def broken(argv):
        return Result(127, "", "python3: not found")

    probes, _ = verify_with(tmp_path, broken)
    assert not probes["webui → internet directly"].ok
    assert not probes["searxng → host directly"].ok
    assert not any(p.ok for name, p in probes.items() if name != "tor bootstrapped")
    assert probes["webui → internet directly"].detail.startswith("error")


def gateway_stopped(fake: FakePodman) -> bool:
    stops = [i for i, c in enumerate(fake.calls) if c[:1] == ["stop"]]
    return bool(stops) and not any(c[:1] == ["start"] for c in fake.calls[stops[-1] :])


def gateway_aware(fake_ref: list[FakePodman]):
    """Healthy network whose Tor-dependent paths die while the gateway is stopped."""

    def net(argv):
        mode, args = argv[6], argv[7:]
        if gateway_stopped(fake_ref[0]):
            if mode == "resolve" and args[0] != "host.containers.internal":
                return probe_blocked("gaierror: [Errno -3] Temporary failure in name resolution")
            if mode == "get" and "check.torproject.org" in args[0]:
                return probe_blocked("URLError: <urlopen error timed out>")
            if mode == "get" and "/search" in args[0]:
                return probe_ok(json.dumps({"results": [], "unresponsive_engines": [["brave", "proxy"]]}))
        return healthy_network(argv)

    return net


def no_fetch(url: str) -> tuple[bool | None, str]:
    raise AssertionError(f"fetched {url} from the Mac, but SearXNG isn't published")


def published_search(fake_ref: list[FakePodman], *, bypasses_tor: bool = False):
    """What a browser on the Mac gets from the published port: results while Tor is up."""

    def fetch(url: str) -> tuple[bool | None, str]:
        assert url.startswith("http://127.0.0.1:8888/search?format=json&q=")
        if gateway_stopped(fake_ref[0]) and not bypasses_tor:
            return True, json.dumps({"results": [], "unresponsive_engines": [["brave", "proxy"]]})
        return True, SEARCH

    return fetch


def test_fail_closed_stops_and_always_restarts_the_gateway(tmp_path):
    ref: list[FakePodman] = []
    fake = FakePodman(gateway_aware(ref))
    ref.append(fake)
    st = stack.Stack(stack.StackConfig(), _paths(tmp_path), fake, sleep=lambda _s: None, fetch=no_fetch)
    probes = {p.name: p for p in st.verify(fail_closed=True)}
    assert all(p.ok for p in probes.values()), [p for p in probes.values() if not p.ok]
    assert len(probes) == 17  # unpublished: no published-port probes
    stop = fake.calls.index(["stop", "-t", "10", stack.GATEWAY])
    assert ["start", stack.GATEWAY] in fake.calls[stop:]


def published_stack(tmp_path, *, bypasses_tor: bool = False) -> dict[str, stack.Probe]:
    ref: list[FakePodman] = []
    fake = FakePodman(gateway_aware(ref))
    ref.append(fake)
    fetch = published_search(ref, bypasses_tor=bypasses_tor)
    st = stack.Stack(stack.StackConfig(searxng_port=8888), _paths(tmp_path), fake, sleep=lambda _s: None, fetch=fetch)
    return {p.name: p for p in st.verify(fail_closed=True)}


def test_verify_searches_through_the_published_port_and_it_fails_closed(tmp_path):
    probes = published_stack(tmp_path)
    assert all(p.ok for p in probes.values()), [p for p in probes.values() if not p.ok]
    assert len(probes) == 19
    assert probes["search via published port"].detail.startswith("2 results (try 1)")
    assert probes["gateway down: published search fails"].detail.startswith("[0, 0, 0]")


def test_fail_closed_catches_a_published_port_that_bypasses_tor(tmp_path):
    probes = published_stack(tmp_path, bypasses_tor=True)
    assert not probes["gateway down: published search fails"].ok
    assert probes["gateway down: published search fails"].detail.startswith("[2, 2, 2]")
    assert probes["gateway down: search fails"].ok  # only the published path is flagged


def test_published_port_unreachable_fails_its_must_work_twin(tmp_path):
    def refused(url: str) -> tuple[bool | None, str]:
        return False, "URLError: <urlopen error [Errno 61] Connection refused>"

    st = stack.Stack(stack.StackConfig(searxng_port=8888), _paths(tmp_path), FakePodman(), fetch=refused)
    probes = {p.name: p for p in st.verify()}
    assert not probes["search via published port"].ok  # "blocked" can't pass for "works"
    assert "Connection refused" in probes["search via published port"].detail


def test_fail_closed_catches_search_that_survives_without_tor(tmp_path):
    probes, _ = verify_with(tmp_path, healthy_network, fail_closed=True)  # search ignores the gateway
    assert not probes["gateway down: search fails"].ok
    assert probes["gateway down: search fails"].detail.startswith("[2, 2, 2]")
    assert not probes["gateway down: page fetch fails"].ok
    assert probes["gateway down: page fetch fails"].detail.startswith("tor")


# -- doctor --------------------------------------------------------------------------------------


def test_doctor_stack_checks_pass_on_the_designed_stack(tmp_path):
    fake = FakePodman()
    st = stack.Stack(stack.StackConfig(), _paths(tmp_path), fake)
    fake.networks[stack.NET_APP] = {**GOOD_NET, "subnets": [{"subnet": "10.153.1.0/24"}]}
    fake.networks[stack.NET_WEB] = {**GOOD_NET, "subnets": [{"subnet": "10.153.2.0/24"}]}
    checks = {c.name: c for c in doctor.check_stack(st)}
    assert all(c.level == "ok" for c in checks.values()), [c for c in checks.values() if c.level != "ok"]


def test_doctor_flags_attachment_publishing_and_tor(tmp_path):
    fake = FakePodman()
    st = stack.Stack(stack.StackConfig(), _paths(tmp_path), fake)
    fake.networks[stack.NET_APP] = {**GOOD_NET, "subnets": [{"subnet": "10.153.1.0/24"}]}
    fake.networks[stack.NET_WEB] = {**GOOD_NET, "dns_enabled": True, "subnets": [{"subnet": "10.153.2.0/24"}]}
    fake.attached[stack.SEARXNG] = [stack.NET_WEB, "podman"]  # a routed network sneaked in
    fake.published[stack.SEARXNG] = "8080/tcp -> 0.0.0.0:8888"
    fake.tor_log = "Bootstrapped 50% (loading_descriptors)\n"
    checks = {c.name: c for c in doctor.check_stack(st)}
    assert checks["attachment"].level == "fail" and stack.SEARXNG in checks["attachment"].detail
    assert checks["published"].level == "fail"
    assert checks["isolation"].level == "fail" and "DNS" in checks["isolation"].detail
    assert checks["tor"].level == "warn"


def test_doctor_allows_searxng_published_only_where_configured(tmp_path):
    fake = FakePodman()

    def published(**cfg: int) -> doctor.Check:
        checks = doctor.check_stack(stack.Stack(stack.StackConfig(**cfg), _paths(tmp_path), fake))
        return next(c for c in checks if c.name == "published")

    fake.published[stack.SEARXNG] = "8080/tcp -> 127.0.0.1:8888"
    assert published(searxng_port=8888).level == "ok"
    assert published().level == "fail"  # published, but nobody asked for it
    assert published(searxng_port=9999).level == "fail"  # not where configured
    fake.published[stack.SEARXNG] = "8080/tcp -> 0.0.0.0:8888"
    assert published(searxng_port=8888).level == "fail"  # beyond loopback
    del fake.published[stack.SEARXNG]
    missing = published(searxng_port=8888)
    assert missing.level == "warn" and missing.fix == "lmtune stack up"  # configured, not deployed yet


def test_doctor_fails_when_the_vm_can_see_the_macs_disk(tmp_path):
    fake = FakePodman()
    fake.shares = ["/Users", "/etc/containers"]
    checks = {c.name: c for c in doctor.check_stack(stack.Stack(stack.StackConfig(), _paths(tmp_path), fake))}
    assert checks["vm-shares"].level == "fail" and "/Users" in checks["vm-shares"].detail
    assert '"Mounts": []' in checks["vm-shares"].fix


def test_search_gets_retries_only_when_it_should_work(tmp_path):
    """One bad circuit can fail a query: working paths get 3 tries, closed paths must fail all 3."""
    attempts = {"n": 0}

    def flaky(argv):
        if argv[6] == "get" and "/search" in argv[7]:
            attempts["n"] += 1
            if attempts["n"] == 1:
                return probe_ok(json.dumps({"results": [], "unresponsive_engines": [["brave", "timeout"]]}))
        return healthy_network(argv)

    probes, _ = verify_with(tmp_path, flaky)
    assert probes["search via Tor"].ok and "(try 2)" in probes["search via Tor"].detail
