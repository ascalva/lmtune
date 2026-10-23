"""Filesystem locations. Overridable via environment for tests and unusual setups."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

LABEL = "com.lmtune.server"


def _env_path(var: str, default: str) -> Path:
    return Path(os.environ.get(var, default)).expanduser()


@dataclass(frozen=True)
class Paths:
    config_dir: Path
    state_dir: Path
    log_dir: Path
    launch_agents: Path

    @classmethod
    def default(cls) -> Paths:
        return cls(
            config_dir=_env_path("LMTUNE_CONFIG_DIR", "~/.config/lmtune"),
            state_dir=_env_path("LMTUNE_STATE_DIR", "~/.local/state/lmtune"),
            log_dir=_env_path("LMTUNE_LOG_DIR", "~/Library/Logs/lmtune"),
            launch_agents=_env_path("LMTUNE_LAUNCH_AGENTS", "~/Library/LaunchAgents"),
        )

    @property
    def serve_config(self) -> Path:
        return self.config_dir / "serve.toml"

    @property
    def api_key(self) -> Path:
        return self.config_dir / "api-key"

    @property
    def server_log(self) -> Path:
        return self.log_dir / "server.log"

    @property
    def installed_plist(self) -> Path:
        """Agent in ~/Library/LaunchAgents: loaded by launchd at every login."""
        return self.launch_agents / f"{LABEL}.plist"

    @property
    def session_plist(self) -> Path:
        """Agent outside LaunchAgents: runs under launchd for this login session only."""
        return self.state_dir / f"{LABEL}.plist"
