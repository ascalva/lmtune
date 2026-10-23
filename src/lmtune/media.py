"""`lmtune media`: ffmpeg, exiftool and whisper.cpp as MCP tools, with no network and one folder.

One stdlib server speaks MCP over streamable HTTP (JSON responses, no SSE stream) on host loopback,
behind its own bearer key, and runs inside `sandbox-exec` with a profile generated here:

- no network: no DNS and no outbound connection of any kind (loopback included), so neither the
  worker nor a media file that points ffmpeg at a URL can reach anything; the only network
  operation allowed is accepting connections on its own loopback port;
- the home folder is unreadable and unwritable except the media folder (read/write), the whisper
  models, lmtune's own install and its config/key files (read); writes go only to the media folder;
- everything else (system libraries, Homebrew binaries, Metal) as usual, so whisper.cpp runs on
  the GPU. Metal's shader cache stays unwritable (it is shared by every app's GPU code), which
  costs a measured ~5 s of kernel compilation per transcription.

The worker refuses to start outside a sandbox (`sandbox_check`), and `lmtune media verify` probes
the profile from inside it: every "must be blocked" has a "must work" twin.

Native rather than a podman container (measured, docs/ARCHITECTURE.md): containers get no Metal
(whisper would run on the VM's 2 CPUs), and the podman VM deliberately has no folders of the Mac,
so a container couldn't see the media folder without re-opening a share for every container.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import hmac
import http.client
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, fields
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from . import __version__, keys
from .launchd import AgentState, Launchd, plist
from .paths import Paths
from .proc import Result, Runner, run

MEDIA_LABEL = "com.lmtune.media"
PROCESS_TYPE = "Standard"  # launchd QoS for the worker (docs/ARCHITECTURE.md: measured vs Background)
SANDBOX_EXEC = "/usr/bin/sandbox-exec"
PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")  # newest first
MAX_BODY = 1 << 20  # JSON-RPC requests carry paths and options, never media
MAX_TEXT = 60_000  # characters of tool output returned inline; the full text is saved to a file
OUT = "out"  # results go here, inside the media folder
WORK = ".work"  # intermediate files (16 kHz WAVs); TMPDIR points here
AUDIO_EXT = frozenset({"mp3", "m4a", "wav", "flac", "ogg", "opus"})
VIDEO_EXT = frozenset({"mp4", "mov", "webm", "mkv", "gif"})
IMAGE_EXT = frozenset({"jpg", "jpeg", "png", "webp"})
# ffmpeg reads local files only, even before the sandbox: a playlist or concat list naming
# http://, rtmp:// or tcp:// fails here instead of trying the network
# Images come out flattened onto white: llama-server shows transparent pixels as black (measured:
# a transparent-background PNG of a receipt read as "an entirely black image")
FLATTEN = "split[a][b];[a]drawbox=c=white:t=fill[bg];[bg][b]overlay,format=rgb24"
# never the media folder (it becomes readable and writable by tools that parse untrusted files)
PROTECTED = ("Documents", "Desktop", "Downloads", "Library", "Pictures", "Movies", "Music", ".config", ".ssh", ".local")
FFMPEG = ("ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-protocol_whitelist", "file,pipe")


def _home(p: str) -> Path:
    return Path(p).expanduser()


@dataclass(frozen=True)
class MediaConfig:
    dir: Path = field(default_factory=lambda: _home("~/lmtune-media"))
    port: int = 8095  # host loopback
    whisper_model: Path = field(default_factory=lambda: _home("~/models/whisper/ggml-large-v3-turbo-q5_0.bin"))
    threads: int = 4  # whisper.cpp CPU threads (the encoder runs on the GPU)
    max_audio_s: int = 4 * 3600  # longer recordings are refused, not silently truncated
    timeout_s: int = 3600  # per external command
    key_file: Path = field(default_factory=lambda: _home("~/.config/lmtune/media-key"))
    # Host headers accepted besides loopback, e.g. the address a container reaches the worker on
    extra_hosts: tuple[str, ...] = ()

    def problems(self) -> list[str]:
        out = []
        if not 1024 <= self.port <= 65535:
            out.append(f"port {self.port} must be 1024-65535")
        if not 1 <= self.threads <= 16:
            out.append("threads must be 1-16")
        if self.max_audio_s < 1 or self.timeout_s < 1:
            out.append("max_audio_s and timeout_s must be positive")
        home = Path.home().resolve()
        d = self.dir.expanduser().resolve()
        if home.is_relative_to(d) or len(d.parts) < 3:
            out.append(f"dir {self.dir} must be a dedicated folder, not {d}")
        elif d.is_relative_to(home) and d.relative_to(home).parts[0] in PROTECTED:
            out.append(f"dir {self.dir} must not be inside ~/{d.relative_to(home).parts[0]}")
        return out

    @property
    def models_dir(self) -> Path:
        return self.whisper_model.parent


def load(path: Path) -> MediaConfig:
    raw = dict(tomllib.loads(path.read_text()).get("media", {})) if path.exists() else {}
    unknown = set(raw) - {f.name for f in fields(MediaConfig)}
    if unknown:
        raise ValueError(f"unknown [media] keys: {', '.join(sorted(unknown))}")
    for k in ("dir", "whisper_model", "key_file"):
        if k in raw:
            raw[k] = _home(raw[k])
    raw.setdefault("key_file", path.parent / "media-key")  # next to media.toml, like the server's key
    if "extra_hosts" in raw:
        raw["extra_hosts"] = tuple(raw["extra_hosts"])
    cfg = MediaConfig(**raw)
    if errs := cfg.problems():
        raise ValueError("; ".join(errs))
    return cfg


# -- sandbox ------------------------------------------------------------------------------------


def _sb(p: Path | str) -> str:
    return json.dumps(str(p))  # SBPL strings use the same quoting/escaping as JSON


def _ancestors(p: Path, home: Path) -> list[Path]:
    """Folders between home (inclusive) and p: path lookups stat them."""
    out = [home]
    for i in range(1, len(p.relative_to(home).parts)):
        out.append(home.joinpath(*p.relative_to(home).parts[:i]))
    return out


def sandbox_profile(cfg: MediaConfig, *, home: Path, read_only: Sequence[Path]) -> str:
    """SBPL for the worker. Later rules override earlier ones (verified on macOS 27), so it allows
    by default, then denies network and the home folder, then re-allows the few paths it needs."""
    media = cfg.dir.resolve()
    ro = sorted({p.resolve() for p in (cfg.models_dir, cfg.key_file, *read_only)})
    inside = [p for p in (media, *ro) if p.is_relative_to(home)]
    meta = sorted({a for p in inside for a in _ancestors(p, home)})
    return "\n".join(
        [
            "(version 1)",
            "(allow default)",
            ";; no network: no DNS, no outbound connections (not even loopback); accept on our port only",
            "(deny network*)",
            f'(allow network-bind network-inbound (local ip "localhost:{cfg.port}"))',
            ";; the home folder is off limits except what the worker needs",
            f"(deny file-read* file-write* (subpath {_sb(home)}))",
            "(allow file-read-metadata " + " ".join(f"(literal {_sb(a)})" for a in meta) + ")",
            "(allow file-read* " + " ".join(f"(subpath {_sb(p)})" for p in (media, *ro)) + ")",
            ";; writes only into the media folder",
            "(deny file-write*)",
            f'(allow file-write* (subpath {_sb(media)}) (literal "/dev/null"))',
            "",
        ]
    )


@dataclass(frozen=True)
class Launcher:
    """How to start lmtune inside the sandbox, and the paths that takes reading."""

    argv: tuple[str, ...]
    reads: tuple[Path, ...]

    @classmethod
    def current(cls) -> Launcher:
        """This interpreter and package (verify, foreground `serve`, development checkouts)."""
        code = "from lmtune.cli import main; raise SystemExit(main())"
        # the import root (site-packages, or src/ in a checkout) must be listable, not just lmtune/
        return cls((sys.executable, "-c", code), (Path(sys.prefix), Path(__file__).resolve().parent.parent))

    @classmethod
    def installed(cls) -> Launcher:
        """The `make install` copy (launchd: a checkout under ~/Documents would hang on a privacy prompt)."""
        exe = Path(shutil.which("lmtune") or Path.home() / ".local/bin/lmtune").resolve()
        return cls((str(exe),), (exe.parent.parent,))  # <tool venv>/bin/lmtune → the whole tool venv


def program(cfg: MediaConfig, paths: Paths, launcher: Launcher, sub: str = "worker") -> list[str]:
    """`sandbox-exec -p <profile> lmtune media <sub>`."""
    reads = [*launcher.reads, paths.config_dir / "media.toml"]
    profile = sandbox_profile(cfg, home=Path.home().resolve(), read_only=reads)
    return [SANDBOX_EXEC, "-p", profile, *launcher.argv, "media", sub]


def sandboxed() -> bool:
    """True when this process runs under a sandbox profile (libsystem's sandbox_check)."""
    try:
        fn = ctypes.CDLL(None).sandbox_check
    except AttributeError, OSError:
        return False
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    fn.restype = ctypes.c_int
    return bool(fn(os.getpid(), None, 0) == 1)


# -- tools --------------------------------------------------------------------------------------


class ToolError(Exception):
    """A tool failed in a way the model should see (bad path, ffmpeg error, busy)."""


Exec = Callable[[Sequence[str], float], Result]


def exec_cmd(cmd: Sequence[str], timeout: float) -> Result:
    try:
        p = subprocess.run(list(cmd), capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError:
        return Result(127, "", f"{cmd[0]}: not found")
    except subprocess.TimeoutExpired:
        return Result(124, "", f"{cmd[0]}: timed out after {timeout:.0f} s")
    return Result(p.returncode, p.stdout, p.stderr)


def _schema(props: dict[str, Any], required: Sequence[str] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


_PATH = {"type": "string", "description": "file path relative to the media folder (see list_files)"}


class Tools:
    def __init__(self, cfg: MediaConfig, exe: Exec = exec_cmd, clock: Callable[[], float] = time.monotonic):
        self.cfg, self.exe, self.clock = cfg, exe, clock
        self.root = cfg.dir.resolve()
        self._busy = threading.Lock()  # one heavy job at a time: they share the GPU with the model

    # -- paths --

    def path(self, rel: object, *, must_exist: bool = True) -> Path:
        if not isinstance(rel, str) or not rel.strip():
            raise ToolError("path must be a non-empty string relative to the media folder")
        p = (self.root / rel).resolve()
        if not p.is_relative_to(self.root) or p == self.root:
            raise ToolError(f"{rel}: outside the media folder")
        if p.relative_to(self.root).parts[0] == WORK:
            raise ToolError(f"{rel}: internal folder")
        if must_exist and not p.is_file():
            raise ToolError(f"{rel}: no such file in the media folder (see list_files)")
        return p

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root))

    def _out(self, src: Path, suffix: str, ext: str) -> Path:
        out = self.root / OUT
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{src.stem}{suffix}.{ext}"

    def models(self) -> dict[str, Path]:
        d = self.cfg.models_dir
        return {p.name: p for p in sorted(d.glob("ggml-*.bin"))} if d.is_dir() else {}

    def _run(self, cmd: Sequence[str], what: str) -> Result:
        r = self.exe(cmd, self.cfg.timeout_s)
        if not r.ok:
            raise ToolError(f"{what} failed ({r.code}): {(r.err or r.out).strip()[-800:]}")
        return r

    @contextlib.contextmanager
    def _heavy(self) -> Any:
        if not self._busy.acquire(blocking=False):
            raise ToolError("busy: another media job is running; try again when it finishes")
        try:
            yield
        finally:
            self._busy.release()

    # -- the tools --

    def specs(self) -> list[dict[str, Any]]:
        models = list(self.models()) or [self.cfg.whisper_model.name]
        return [
            {
                "name": "list_files",
                "description": "List files in the media folder (the only folder these tools can read or write). "
                "Results of other tools are saved under out/.",
                "inputSchema": _schema({"folder": {"type": "string", "description": "subfolder; default: all"}}),
            },
            {
                "name": "probe",
                "description": "Container, duration, and streams (codecs, resolution, sample rate) of a media file.",
                "inputSchema": _schema({"path": _PATH}, ["path"]),
            },
            {
                "name": "metadata",
                "description": "EXIF/IPTC/XMP/QuickTime metadata of a file (exiftool): camera, lens, "
                "capture time, GPS, etc.",
                "inputSchema": _schema({"path": _PATH}, ["path"]),
            },
            {
                "name": "transcribe",
                "description": "Transcribe speech in an audio or video file with whisper.cpp, locally. "
                "Saves out/<name>.txt (and .srt with timestamps) and returns the text.",
                "inputSchema": _schema(
                    {
                        "path": _PATH,
                        "language": {"type": "string", "description": "ISO code like 'en' or 'auto' (default)"},
                        "model": {"type": "string", "enum": models, "description": f"default {models[0]}"},
                        "timestamps": {"type": "boolean", "description": "also write an .srt; default false"},
                    },
                    ["path"],
                ),
            },
            {
                "name": "convert",
                "description": "Convert or trim a media file with ffmpeg into out/: audio "
                f"({', '.join(sorted(AUDIO_EXT))}), video ({', '.join(sorted(VIDEO_EXT))}) or image "
                f"({', '.join(sorted(IMAGE_EXT))}). Optionally cut start..end seconds and limit the "
                "longest side to max_px.",
                "inputSchema": _schema(
                    {
                        "path": _PATH,
                        "to": {"type": "string", "enum": sorted(AUDIO_EXT | VIDEO_EXT | IMAGE_EXT)},
                        "start": {"type": "number", "minimum": 0},
                        "end": {"type": "number", "minimum": 0},
                        "max_px": {"type": "integer", "minimum": 16, "maximum": 8192},
                    },
                    ["path", "to"],
                ),
            },
            {
                "name": "frames",
                "description": "Extract still frames from a video as JPEGs into out/<name>_frames/ (e.g. to look "
                "at them with an image-capable model).",
                "inputSchema": _schema(
                    {
                        "path": _PATH,
                        "every_s": {"type": "number", "minimum": 0.1, "description": "default 10"},
                        "max_frames": {"type": "integer", "minimum": 1, "maximum": 100, "description": "default 12"},
                        "max_px": {"type": "integer", "minimum": 64, "maximum": 4096, "description": "default 1024"},
                    },
                    ["path"],
                ),
            },
        ]

    def call(self, name: str, args: dict[str, Any]) -> str:
        allowed = {s["name"]: set(s["inputSchema"]["properties"]) for s in self.specs()}
        if name not in allowed:
            raise ToolError(f"unknown tool {name}")
        extra = set(args) - allowed[name]
        if extra:
            raise ToolError(f"unknown arguments: {', '.join(sorted(extra))}")
        fn: Callable[..., str] = getattr(self, f"t_{name}")
        return fn(**args)

    def t_list_files(self, folder: str = "") -> str:
        base = self.root if not folder else self.path(folder, must_exist=False)
        if not base.is_dir():
            raise ToolError(f"{folder}: no such folder")
        rows = []
        for p in sorted(base.rglob("*")):
            if p.is_file() and not p.name.startswith(".") and WORK not in p.relative_to(self.root).parts:
                st = p.stat()
                mtime = datetime.fromtimestamp(st.st_mtime, UTC).isoformat(timespec="seconds")
                rows.append({"path": self.rel(p), "bytes": st.st_size, "modified": mtime})
            if len(rows) >= 500:
                break
        return json.dumps(rows, indent=1) if rows else f"no files (put media into {self.root})"

    def _probe(self, p: Path) -> dict[str, Any]:
        r = self._run(
            ["ffprobe", "-v", "error", "-protocol_whitelist", "file,pipe", "-print_format", "json",
             "-show_format", "-show_streams", "--", str(p)],
            "ffprobe",
        )  # fmt: skip
        return dict(json.loads(r.out or "{}"))

    def t_probe(self, path: str) -> str:
        return json.dumps(summarize_probe(self._probe(self.path(path))), indent=1)

    def t_metadata(self, path: str) -> str:
        r = self._run(
            ["exiftool", "-json", "-G1", "-api", "LargeFileSupport=1", "--", str(self.path(path))], "exiftool"
        )
        return _clip(r.out, "metadata")

    def t_transcribe(self, path: str, language: str = "auto", model: str = "", timestamps: bool = False) -> str:
        src = self.path(path)
        models = self.models()
        mpath = models.get(model) if model else self.cfg.whisper_model
        if mpath is None or not mpath.is_file():
            raise ToolError(f"model {model or mpath}: not installed (have: {', '.join(models) or 'none'})")
        if not isinstance(language, str) or not (language == "auto" or (language.isalpha() and len(language) <= 3)):
            raise ToolError("language must be 'auto' or an ISO code like 'en'")
        dur = summarize_probe(self._probe(src)).get("duration_s")
        if not dur:
            raise ToolError(f"{path}: no audio duration (not a media file?)")
        if dur > self.cfg.max_audio_s:
            raise ToolError(f"{path}: {dur / 60:.0f} min is longer than the {self.cfg.max_audio_s / 60:.0f} min limit")
        with self._heavy():
            work = self.root / WORK
            work.mkdir(parents=True, exist_ok=True)
            wav = work / f"{uuid.uuid4().hex}.wav"
            t0 = self.clock()
            try:
                self._run(
                    [*FFMPEG, "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-y", str(wav)],
                    "ffmpeg",
                )
                stem = self._out(src, "", "txt").with_suffix("")
                self._run(
                    ["whisper-cli", "-m", str(mpath), "-f", str(wav), "-l", language, "-t", str(self.cfg.threads),
                     "-np", "-oj", "-otxt", *(["-osrt"] if timestamps else []), "-of", str(stem)],
                    "whisper",
                )  # fmt: skip
            finally:
                wav.unlink(missing_ok=True)
            took = self.clock() - t0
        res = json.loads(stem.with_suffix(".json").read_text(errors="replace"))
        text = " ".join(s.get("text", "").strip() for s in res.get("transcription", [])).strip()
        saved = [self.rel(stem.with_suffix(e)) for e in (".txt", ".srt") if stem.with_suffix(e).exists()]
        lang = res.get("result", {}).get("language", language)
        head = (
            f"[{mpath.name} · language {lang} · {dur:.0f} s of audio in {took:.0f} s "
            f"({dur / took:.0f}x realtime) · saved {', '.join(saved)}]\n"
        )
        return head + _clip(text or "(no speech detected)", "transcript", saved[0] if saved else "")

    def t_convert(
        self, path: str, to: str, start: float | None = None, end: float | None = None, max_px: int | None = None
    ) -> str:
        src = self.path(path)
        ext = to.lower().lstrip(".") if isinstance(to, str) else ""
        if ext not in AUDIO_EXT | VIDEO_EXT | IMAGE_EXT:
            raise ToolError(f"to must be one of {', '.join(sorted(AUDIO_EXT | VIDEO_EXT | IMAGE_EXT))}")
        for name, v in (("start", start), ("end", end)):
            if v is not None and (not isinstance(v, int | float) or v < 0):
                raise ToolError(f"{name} must be a non-negative number of seconds")
        if start is not None and end is not None and end <= start:
            raise ToolError("end must be after start")
        cut = [*(["-ss", f"{start:g}"] if start is not None else []), *(["-to", f"{end:g}"] if end is not None else [])]
        suffix = f"_{start or 0:g}-{end:g}s" if end is not None else f"_{start:g}s-" if start is not None else ""
        out = self._out(src, suffix, ext)
        if out == src:
            out = self._out(src, suffix + "_1", ext)
        opts: list[str] = ["-vn"] if ext in AUDIO_EXT else []
        vf = [FLATTEN] if ext in IMAGE_EXT else []
        if max_px is not None:
            if not isinstance(max_px, int) or not 16 <= max_px <= 8192:
                raise ToolError("max_px must be an integer 16-8192")
            vf.append(_fit(max_px))
        if vf:
            opts += ["-vf", ",".join(vf)]
        if ext in IMAGE_EXT:
            opts += ["-frames:v", "1", "-update", "1"]
        with self._heavy():
            self._run([*FFMPEG, *cut, "-i", str(src), *opts, "-y", str(out)], "ffmpeg")
        return f"saved {self.rel(out)} ({out.stat().st_size} bytes)"

    def t_frames(self, path: str, every_s: float = 10, max_frames: int = 12, max_px: int = 1024) -> str:
        src = self.path(path)
        if not isinstance(every_s, int | float) or every_s < 0.1:
            raise ToolError("every_s must be >= 0.1")
        if not isinstance(max_frames, int) or not 1 <= max_frames <= 100:
            raise ToolError("max_frames must be 1-100")
        if not isinstance(max_px, int) or not 64 <= max_px <= 4096:
            raise ToolError("max_px must be 64-4096")
        folder = self.root / OUT / f"{src.stem}_frames"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        vf = f"fps=1/{every_s:g},{_fit(max_px)}"
        with self._heavy():
            self._run(
                [*FFMPEG, "-i", str(src), "-vf", vf, "-frames:v", str(max_frames), "-q:v", "3",
                 str(folder / "frame_%03d.jpg")],
                "ffmpeg",
            )  # fmt: skip
        got = sorted(folder.glob("frame_*.jpg"))
        return json.dumps({"every_s": every_s, "frames": [self.rel(p) for p in got]}, indent=1)


def _fit(max_px: int) -> str:
    """Scale so the longer side is at most max_px, never up, keeping even dimensions."""
    return f"scale='if(gte(iw,ih),min({max_px},iw),-2)':'if(gte(iw,ih),-2,min({max_px},ih))':force_divisible_by=2"


def _clip(text: str, what: str, saved: str = "") -> str:
    if len(text) <= MAX_TEXT:
        return text
    where = f"; the full {what} is in {saved}" if saved else ""
    return text[:MAX_TEXT] + f"\n[... {what} truncated at {MAX_TEXT} characters{where}]"


def summarize_probe(d: dict[str, Any]) -> dict[str, Any]:
    fmt = d.get("format", {})
    streams = []
    for s in d.get("streams", []):
        keep = ("codec_type", "codec_name", "width", "height", "sample_rate", "channels", "bit_rate")
        row = {k: s[k] for k in keep if k in s}
        if s.get("codec_type") == "video" and s.get("avg_frame_rate", "0/0") not in ("0/0", ""):
            num, _, den = s["avg_frame_rate"].partition("/")
            with contextlib.suppress(ValueError, ZeroDivisionError):
                row["fps"] = round(float(num) / float(den or 1), 3)
        streams.append(row)
    dur = fmt.get("duration")
    return {
        "format": fmt.get("format_name"),
        "duration_s": round(float(dur), 3) if dur not in (None, "N/A") else None,
        "bytes": int(fmt["size"]) if str(fmt.get("size", "")).isdigit() else None,
        "streams": streams,
    }


# -- MCP (JSON-RPC 2.0 over streamable HTTP) ----------------------------------------------------


class RpcError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _error(mid: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


INSTRUCTIONS = (
    "Local media tools. They can only see the media folder: use list_files first and pass paths "
    "relative to it. Results are saved under out/. Nothing here has network access."
)


class Mcp:
    def __init__(self, tools: Tools, log: Callable[[str], None] = lambda _: None):
        self.tools, self.log = tools, log

    def handle(self, msg: Any) -> Any:
        """A JSON-RPC message or batch → the response(s), or None when nothing is owed (notifications)."""
        if isinstance(msg, list):
            if not msg:
                return _error(None, -32600, "empty batch")
            out = [r for m in msg if (r := self._one(m)) is not None]
            return out or None
        return self._one(msg)

    def _one(self, m: Any) -> dict[str, Any] | None:
        if isinstance(m, dict) and "method" not in m and ("result" in m or "error" in m):
            return None  # a response to a request we never make
        if not isinstance(m, dict) or m.get("jsonrpc") != "2.0" or not isinstance(m.get("method"), str):
            return _error(m.get("id") if isinstance(m, dict) else None, -32600, "invalid request")
        notification, mid = "id" not in m, m.get("id")
        params = m.get("params") or {}
        try:
            result = self._dispatch(m["method"], params)
        except RpcError as e:
            return None if notification else _error(mid, e.code, e.message)
        return None if notification else {"jsonrpc": "2.0", "id": mid, "result": result}

    def _dispatch(self, method: str, params: Any) -> dict[str, Any]:
        if not isinstance(params, dict):
            raise RpcError(-32602, "params must be an object")
        if method == "initialize":
            asked = params.get("protocolVersion")
            return {
                "protocolVersion": asked if asked in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0],
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "lmtune-media", "version": __version__},
                "instructions": INSTRUCTIONS,
            }
        if method == "ping" or method.startswith("notifications/"):
            return {}
        if method == "tools/list":
            return {"tools": self.tools.specs()}
        if method == "tools/call":
            name, args = params.get("name"), params.get("arguments") or {}
            if not isinstance(name, str) or not isinstance(args, dict):
                raise RpcError(-32602, "tools/call needs a name and an arguments object")
            t0 = time.monotonic()
            try:
                text, err = self.tools.call(name, args), False
            except ToolError as e:
                text, err = str(e), True
            except TypeError as e:  # wrong argument types that slipped past the schema
                text, err = f"bad arguments: {e}", True
            self.log(f"{name} {'error' if err else 'ok'} {time.monotonic() - t0:.1f}s")
            return {"content": [{"type": "text", "text": text}], "isError": err}
        raise RpcError(-32601, f"method not found: {method}")


def allowed_hosts(cfg: MediaConfig) -> frozenset[str]:
    """Host headers we answer (DNS-rebinding defense, on top of the key)."""
    hosts = {f"127.0.0.1:{cfg.port}", f"localhost:{cfg.port}", f"[::1]:{cfg.port}", *cfg.extra_hosts}
    return frozenset(h.lower() for h in hosts)


class MediaServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, addr: tuple[str, int], mcp: Mcp, key: str, hosts: frozenset[str]):
        self.mcp, self.key, self.hosts = mcp, key, hosts
        super().__init__(addr, McpHandler)

    def server_bind(self) -> None:
        """Skip HTTPServer's reverse-DNS lookup of the bind address: the sandbox has no DNS."""
        self.socket.bind(self.server_address)
        self.server_address = self.socket.getsockname()[:2]
        self.server_name, self.server_port = str(self.server_address[0]), int(self.server_address[1])


class McpHandler(BaseHTTPRequestHandler):
    server: MediaServer
    server_version = "lmtune-media"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        """Quiet: tool calls are logged by Mcp, without arguments (paths can be private)."""

    def _send(self, code: int, body: bytes = b"", ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if code == 401:
            self.send_header("WWW-Authenticate", 'Bearer realm="lmtune-media"')
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def _gate(self) -> bool:
        if (self.headers.get("Host") or "").lower() not in self.server.hosts:
            self._send(421, b"unknown host\n", "text/plain")
            return False
        origin = self.headers.get("Origin")
        if origin is not None:  # browsers send it; MCP clients on the Mac/in Open WebUI's backend don't
            self._send(403, b"browser requests are not accepted\n", "text/plain")
            return False
        auth = self.headers.get("Authorization") or ""
        if not hmac.compare_digest(auth.encode(), f"Bearer {self.server.key}".encode()):
            self._send(401, b"missing or wrong key\n", "text/plain")
            return False
        if self.path.split("?", 1)[0] != "/mcp":
            self._send(404, b"not found\n", "text/plain")
            return False
        return True

    def do_POST(self) -> None:
        if not self._gate():
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = -1
        if not 0 < n <= MAX_BODY:
            self._send(413 if n > MAX_BODY else 400, json.dumps(_error(None, -32600, "bad body size")).encode())
            return
        try:
            msg = json.loads(self.rfile.read(n))
        except json.JSONDecodeError, UnicodeDecodeError:
            self._send(400, json.dumps(_error(None, -32700, "parse error")).encode())
            return
        reply = self.server.mcp.handle(msg)
        if reply is None:
            self._send(202)
        else:
            self._send(200, json.dumps(reply).encode())

    def do_GET(self) -> None:
        if self._gate():  # no server-initiated stream: the spec allows 405 here
            self._send(405, b"", "text/plain")

    do_DELETE = do_GET


def serve_forever(cfg: MediaConfig, log: Callable[[str], None] = print) -> None:
    """Run the worker. Refuses to start outside a sandbox: it is meant to have no network."""
    if not sandboxed():
        raise RuntimeError("refusing to run unsandboxed; start it with `lmtune media serve`")
    bad = keys.problems(cfg.key_file)
    if bad:
        raise RuntimeError("media key: " + "; ".join(bad))
    for sub in (OUT, WORK):
        (cfg.dir / sub).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str((cfg.dir / WORK).resolve())  # the only writable place
    tools = Tools(cfg)
    httpd = MediaServer(("127.0.0.1", cfg.port), Mcp(tools, log), keys.read(cfg.key_file), allowed_hosts(cfg))
    log(f"lmtune media: http://127.0.0.1:{cfg.port}/mcp · folder {tools.root} · models {', '.join(tools.models())}")
    try:
        httpd.serve_forever(poll_interval=1.0)
    finally:
        httpd.server_close()


# -- verify: probe the sandbox from inside it ---------------------------------------------------


@dataclass(frozen=True)
class Probe:
    name: str
    expect: str  # "blocked" | "works"
    ok: bool
    detail: str


def _try(fn: Callable[[], object]) -> tuple[bool, str]:
    try:
        r = fn()
    except Exception as e:  # noqa: BLE001 - every failure is an outcome to report
        return False, f"{type(e).__name__}: {e}"[:160]
    return True, str(r)[:160]


def probes(cfg: MediaConfig, server_port: int, secret: Path) -> list[Probe]:
    """Run inside the sandbox. Each blocked case has a working twin, so a broken probe can't pass."""

    def in_sandbox() -> object:
        if not sandboxed():
            raise RuntimeError("not sandboxed")
        return "sandbox_check: sandboxed"

    def connect(host: str, port: int) -> Callable[[], object]:
        def f() -> object:
            with socket.create_connection((host, port), timeout=5):
                return "connected"

        return f

    def ffmpeg_url() -> object:
        r = exec_cmd(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "http://example.com/a.mp3", "-f", "null",
                      "-"], 30)  # fmt: skip
        if not r.ok:
            raise OSError(r.err.strip()[-120:])
        return "fetched"

    def ffmpeg_local() -> object:
        out = cfg.dir / WORK / "probe.wav"
        r = exec_cmd([*FFMPEG, "-f", "lavfi", "-i", "sine=frequency=440:duration=0.2", "-y", str(out)], 30)
        out.unlink(missing_ok=True)
        if not r.ok:
            raise OSError(r.err.strip()[-120:])
        return "wrote and removed a WAV in the media folder"

    def write(p: Path) -> Callable[[], object]:
        def f() -> object:
            p.write_text("probe")
            p.unlink()
            return f"wrote {p}"

        return f

    def read(p: Path) -> Callable[[], object]:
        def f() -> object:
            with p.open("rb") as fh:
                fh.read(16)
            return f"read {p}"

        return f

    def listen_other() -> object:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            s.listen()
            return f"listening on {s.getsockname()}"

    def listen_ours() -> object:
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", cfg.port))
            except OSError as e:
                if e.errno == errno.EADDRINUSE:  # the kernel got past the sandbox check: the worker has it
                    return f"port {cfg.port} in use by the running worker (bind permitted)"
                raise
            s.listen()
            return f"listening on {s.getsockname()}"

    home = Path.home()
    cases: list[tuple[str, str, Callable[[], object]]] = [
        ("sandboxed", "works", in_sandbox),
        ("dns", "blocked", lambda: socket.getaddrinfo("example.com", 443)),
        ("internet", "blocked", connect("1.1.1.1", 443)),
        ("loopback-model", "blocked", connect("127.0.0.1", server_port)),
        ("ffmpeg-url", "blocked", ffmpeg_url),
        ("ffmpeg-local", "works", ffmpeg_local),
        ("listen-other-port", "blocked", listen_other),
        ("listen-own-port", "works", listen_ours),
        ("read-api-key", "blocked", read(secret)),
        ("read-ssh", "blocked", lambda: list((home / ".ssh").iterdir())),
        ("read-documents", "blocked", lambda: list((home / "Documents").iterdir())),
        ("read-media-key", "works", read(cfg.key_file)),
        ("read-whisper-model", "works", read(cfg.whisper_model)),
        ("write-home", "blocked", write(home / f".lmtune-probe-{uuid.uuid4().hex}")),
        ("write-tmp", "blocked", write(Path("/tmp") / f"lmtune-probe-{uuid.uuid4().hex}")),  # noqa: S108
        ("write-media", "works", write(cfg.dir / f".probe-{uuid.uuid4().hex}")),
    ]
    out = []
    for name, expect, fn in cases:
        worked, detail = _try(fn)
        ok = worked == (expect == "works")
        if not worked and detail.startswith("FileNotFoundError"):  # a missing target proves nothing
            ok, detail = False, "inconclusive: " + detail
        out.append(Probe(name, expect, ok, detail))
    return out


def verify(
    cfg: MediaConfig, paths: Paths, server_port: int, secret: Path, launcher: Launcher | None = None
) -> list[Probe]:
    """The probes, run inside the same sandbox as the worker, plus the running worker's front
    door from outside (its key, its tools). `secret` is a file that must stay unreadable (the
    model server's API key)."""
    argv = program(cfg, paths, launcher or Launcher.current(), sub="probe")
    r = exec_cmd([*argv, "--server-port", str(server_port), "--secret", str(secret)], 180)
    rows = [Probe(**json.loads(line)) for line in r.out.splitlines() if line.startswith("{")]
    if not rows:
        raise RuntimeError(f"probes did not run ({r.code}): {r.err.strip()[-500:]}")
    up, _ = _try(lambda: socket.create_connection(("127.0.0.1", cfg.port), timeout=2).close())
    if up:
        code, _ = call(cfg, "ping", {}, key="wrong")
        rows.append(Probe("worker-without-key", "blocked", code == 401, f"HTTP {code}"))
        code, res = call(cfg, "tools/list", {})
        names = [t["name"] for t in res.get("result", {}).get("tools", [])] if isinstance(res, dict) else []
        rows.append(Probe("worker-with-key", "works", code == 200 and bool(names), f"HTTP {code}: {', '.join(names)}"))
    return rows


def call(
    cfg: MediaConfig, method: str, params: dict[str, Any], *, key: str | None = None, timeout: float = 3600
) -> tuple[int, Any]:
    """One JSON-RPC call to the running worker (CLI `lmtune media call`, verify, measurements)."""
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    conn = http.client.HTTPConnection("127.0.0.1", cfg.port, timeout=timeout)
    try:
        conn.request(
            "POST",
            "/mcp",
            body=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream",
                "Authorization": f"Bearer {key if key is not None else keys.read(cfg.key_file)}",
            },
        )
        r = conn.getresponse()
        raw = r.read()
    finally:
        conn.close()
    try:
        return r.status, json.loads(raw) if raw else None
    except json.JSONDecodeError:
        return r.status, raw.decode(errors="replace")


# -- login agent --------------------------------------------------------------------------------


def agent_plist(paths: Paths) -> Path:
    return paths.launch_agents / f"{MEDIA_LABEL}.plist"


def install(cfg: MediaConfig, paths: Paths, runner: Runner = run) -> Path:
    """Run the worker at login under launchd (restarted after a crash), sandboxed."""
    keys.ensure(cfg.key_file)
    cfg.dir.mkdir(parents=True, exist_ok=True)
    path = agent_plist(paths)
    paths.log_dir.mkdir(parents=True, exist_ok=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    argv = program(cfg, paths, Launcher.installed())
    path.write_bytes(plist(MEDIA_LABEL, argv, paths.log_dir / "media.log", process_type=PROCESS_TYPE))
    ld = Launchd(MEDIA_LABEL, runner)
    if ld.state().loaded:
        ld.bootout()
    ld.bootstrap(path)
    return path


def agent_state(runner: Runner = run) -> AgentState:
    return Launchd(MEDIA_LABEL, runner).state()


def uninstall(paths: Paths, runner: Runner = run) -> None:
    Launchd(MEDIA_LABEL, runner).bootout()
    agent_plist(paths).unlink(missing_ok=True)
