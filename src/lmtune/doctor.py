"""`lmtune doctor`: every check that has bitten a real run, with the fix for each."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from . import flags, gguf, keys, machine, stack
from .config import ServeConfig
from .serve import Server

Level = Literal["ok", "warn", "fail"]


@dataclass(frozen=True)
class Check:
    name: str
    level: Level
    detail: str
    fix: str = ""


def _safe(name: str, fn: Callable[[], Check]) -> Check:
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - a doctor must report, not crash
        return Check(name, "fail", f"check crashed: {type(e).__name__}: {e}")


def check_config(cfg: ServeConfig) -> Check:
    p = cfg.problems()
    return (
        Check("config", "fail", "; ".join(p), "edit ~/.config/lmtune/serve.toml")
        if p
        else Check("config", "ok", "valid")
    )


def check_runtime(cfg: ServeConfig) -> Check:
    if cfg.backend == "podman":
        return Check("runtime", "ok" if shutil.which("podman") else "fail", "podman backend (container)")
    if not shutil.which("llama-server"):
        return Check("runtime", "fail", "llama-server not on PATH", "brew install llama.cpp")
    tf = flags.discover("llama-server")
    bad = tf.missing(cfg.argv())
    if bad:
        return Check(
            "runtime",
            "fail",
            f"build {tf.build} rejects {' '.join(bad)}",
            "flags changed upstream; update serve.toml or re-run `lmtune tune run`",
        )
    return Check("runtime", "ok", f"llama-server build {tf.build} accepts every configured flag")


VISION_BUFFERS = 128 * 2**20  # measured: footprint +930-1003 MiB for an 885 MiB projector


def check_memory(cfg: ServeConfig, m: machine.Machine) -> Check:
    if not cfg.model.exists():
        return Check("memory", "fail", f"model {cfg.model} missing", "lmtune tune fetch <profile>")
    if cfg.mmproj is not None and not cfg.mmproj.exists():
        return Check("memory", "fail", f"mmproj {cfg.mmproj} missing", 'download it or set mmproj = ""')
    info = gguf.inspect(cfg.model)
    weights = cfg.model.stat().st_size
    kv = info.kv_bytes(cfg.ctx * cfg.parallel, cfg.kv)
    # vision: the projector's weights live in GPU buffers (not mmapped), plus its compute buffer
    vision = cfg.mmproj.stat().st_size + VISION_BUFFERS if cfg.mmproj is not None else 0
    need = weights + kv + vision + cfg.cache_ram_mib * 2**20 + 1.5 * 2**30  # + compute buffers/MTP context
    gpu = (m.gpu_working_set_mib or 0) * 2**20
    detail = (
        f"weights {weights / 2**30:.1f} + KV {kv / 2**30:.1f} "
        + (f"+ vision {vision / 2**30:.1f} " if vision else "")
        + f"+ cache {cfg.cache_ram_mib / 1024:.1f} "
        f"+ ~1.5 buffers = {need / 2**30:.1f} GiB of {m.ram_bytes / 2**30:.0f} GiB RAM"
    )
    if gpu and weights + kv + vision > gpu:
        return Check(
            "memory",
            "fail",
            detail + f"; exceeds GPU working set {gpu / 2**30:.1f} GiB",
            "lower ctx or use a smaller quant",
        )
    if need > m.ram_bytes - 6 * 2**30:
        return Check(
            "memory",
            "warn",
            detail + "; under 6 GiB left for macOS and apps",
            "lower cache_ram_mib/ctx, or rely on sleep_idle_s",
        )
    return Check("memory", "ok", detail)


def check_contention(ignore_pids: set[int]) -> Check:
    c = machine.contention(ignore_pids=ignore_pids)
    return (
        Check("contention", "ok", "no other LLM runtimes loaded")
        if c.clean
        else Check("contention", "warn", c.describe(), "stop them before benchmarking; they compete for RAM/bandwidth")
    )


def check_swap() -> Check:
    s = machine.swap_used_mb()
    if s is None:
        return Check("swap", "warn", "unknown")
    return Check("swap", "warn" if s > 8192 else "ok", f"{s / 1024:.1f} GiB used")


def check_gpu_idle() -> Check:
    g = machine.gpu_idle_utilization(samples=3, interval=0.3)
    if g is None:
        return Check("gpu", "warn", "utilization unreadable")
    return Check(
        "gpu",
        "warn" if g > 5 else "ok",
        f"{g:.0f}% busy before inference",
        "close GPU-heavy apps for steadier decode speed" if g > 5 else "",
    )


def check_key(cfg: ServeConfig) -> Check:
    if cfg.api_key_file is None:
        return Check("api-key", "fail", "no api_key_file configured", "lmtune key init")
    p = keys.problems(cfg.api_key_file)
    return (
        Check("api-key", "fail", "; ".join(p), "lmtune key init / chmod 600")
        if p
        else Check("api-key", "ok", f"{cfg.api_key_file} (0600)")
    )


def check_server(srv: Server) -> list[Check]:
    st = srv.status()
    out = []
    if not st.agent.loaded:
        out.append(Check("server", "warn", "not running", "lmtune serve start"))
    else:
        state = "sleeping (model unloaded)" if st.sleeping else ("busy" if st.busy else "idle")
        lvl: Level = "ok" if st.health == "ok" else "fail"
        out.append(
            Check(
                "server",
                lvl,
                f"{st.health}, {state}, pid {st.agent.pid}, agent {'installed' if st.installed else 'session-only'}",
                "" if lvl == "ok" else "lmtune serve logs",
            )
        )
    if st.listeners:
        out.append(
            Check(
                "listen",
                "ok" if st.loopback_only else "fail",
                ", ".join(st.listeners),
                "" if st.loopback_only else "something listens beyond loopback: check host/extra",
            )
        )
    e = st.exposure
    if e is not None:
        out.append(Check("tailnet", "ok" if e.url else "warn", e.url or e.detail, "" if e.url else "lmtune expose on"))
    return out


# podman can't change an existing machine's shares; they live in its JSON config
VM_SHARES_FIX = (
    'a container escape would reach these Mac folders: podman machine stop; set "Mounts": [] in '
    "~/.config/containers/podman/machine/<provider>/<machine>.json; podman machine start "
    "(and `volumes = []` under [machine] in ~/.config/containers/containers.conf for new machines)"
)


def check_stack(st: stack.Stack) -> list[Check]:
    """Structure only (inspect, no traffic); `lmtune stack verify` probes the paths themselves."""
    states = st.status()
    if not any(c.exists for c in states):
        return []  # not deployed: nothing to check
    out = [
        Check(c.name, "ok" if c.running else "fail", c.status or "absent", "" if c.running else "lmtune stack up")
        for c in states
    ]
    r = st.pm("port", stack.WEBUI, check=False)
    published = [line.split("->")[-1].strip() for line in r.out.splitlines() if "->" in line]
    loopback = all(p.startswith("127.0.0.1:") for p in published)
    out.append(
        Check(
            "webui-port",
            "ok" if loopback else "fail",
            ", ".join(published) or "not published",
            "" if loopback else "Open WebUI is published beyond loopback",
        )
    )
    others = {
        name: st.pm("port", name, check=False).out.strip() for name in (stack.SEARXNG, stack.RELAY, stack.GATEWAY)
    }
    # SearXNG only when searxng_port asks for it, and then exactly there on loopback
    sx = st.cfg.searxng_port
    allowed = {stack.SEARXNG: f"8080/tcp -> 127.0.0.1:{sx}"} if sx else {}
    leaks = [f"{name}: {ports}" for name, ports in others.items() if ports and ports != allowed.get(name)]
    unpublished = bool(sx) and not others[stack.SEARXNG]
    out.append(
        Check(
            "published",
            "fail" if leaks else "warn" if unpublished else "ok",
            "; ".join(leaks)
            or (f"searxng_port {sx} is set but SearXNG isn't published" if unpublished else "")
            or ("Open WebUI and SearXNG, loopback only" if sx else "only Open WebUI (loopback)"),
            "lmtune stack up" if leaks or unpublished else "",
        )
    )
    net_problems = [
        p
        for name, subnet in ((stack.NET_APP, st.cfg.app_subnet), (stack.NET_WEB, st.cfg.web_subnet))
        for p in st.network_problems(name, subnet)
    ]
    out.append(
        Check(
            "isolation",
            "fail" if net_problems else "ok",
            "; ".join(net_problems) or "internal networks: no route out, no DNS, no IPv6, no VM address",
            "lmtune stack up" if net_problems else "",
        )
    )
    expected = st.expected_networks()
    wrong = [
        f"{c.name} on {', '.join(c.networks) or 'nothing'}"
        for c in states
        if c.exists and c.networks != expected[c.name]
    ]
    out.append(
        Check(
            "attachment",
            "fail" if wrong else "ok",
            "; ".join(wrong) or "Open WebUI/SearXNG reach the web only via the Tor gateway",
            "lmtune stack up" if wrong else "",
        )
    )
    shares = st.vm_shares()
    out.append(
        Check(
            "vm-shares",
            "warn" if shares is None else "fail" if shares else "ok",
            "couldn't list" if shares is None else ", ".join(shares) or "the podman VM can't see the Mac's disk",
            "" if shares == [] else VM_SHARES_FIX,
        )
    )
    pct = stack.tor_bootstrap(st.logs(stack.GATEWAY, 200))
    out.append(
        Check(
            "tor",
            "ok" if pct == 100 else "warn",
            "bootstrapped" if pct == 100 else f"{pct}% bootstrapped" if pct is not None else "no bootstrap in log",
            "" if pct == 100 else "lmtune stack logs tor; research fails (closed) until Tor is up",
        )
    )
    return out


def run_all(cfg: ServeConfig, srv: Server) -> list[Check]:
    m = machine.detect(str(cfg.model.parent))
    ours = {srv.launchd.state().pid or -1}
    checks = [
        _safe("config", lambda: check_config(cfg)),
        _safe("runtime", lambda: check_runtime(cfg)),
        _safe("memory", lambda: check_memory(cfg, m)),
        _safe("api-key", lambda: check_key(cfg)),
        _safe("contention", lambda: check_contention(ours)),
        _safe("swap", check_swap),
        _safe("gpu", check_gpu_idle),
    ]
    try:
        checks += check_server(srv)
    except Exception as e:  # noqa: BLE001
        checks.append(Check("server", "fail", f"status crashed: {e}"))
    try:
        checks += check_stack(stack.Stack(stack.load(srv.paths.config_dir / "stack.toml"), srv.paths))
    except Exception as e:  # noqa: BLE001
        checks.append(Check("stack", "warn", f"could not inspect: {e}"))
    if m.power.low_power_mode:
        checks.append(Check("power", "warn", "Low Power Mode is on", "System Settings > Battery"))
    return checks
