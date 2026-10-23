"""Shared type aliases."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

Log = Callable[[str], None]
JSON = dict[str, Any]
# exactness samples: prefix index -> one token-id list per seed
Samples = dict[int, list[list[int]]]
