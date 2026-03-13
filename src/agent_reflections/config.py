"""Configuration loading from .env files for the reflection pipeline."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SourceConfig:
    """A single context source: a directory path and how many files to sample."""

    path: Path
    count: int

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"Source count must be >= 1, got {self.count}")


@dataclass(frozen=True)
class ReflectConfig:
    """Full configuration for the reflection context gathering pipeline."""

    session_dir: Path
    session_depth: int
    sources: dict[str, SourceConfig] = field(default_factory=dict)
    model: str = "grok-4-1-fast-reasoning"
    api_key_file: Path | None = None

    def __post_init__(self) -> None:
        if self.session_depth < 1:
            raise ValueError(f"Session depth must be >= 1, got {self.session_depth}")


def _expand_path(raw: str) -> Path:
    """Expand ~ and env vars in a path string."""
    return Path(os.path.expanduser(os.path.expandvars(raw.strip())))


def _parse_env_file(env_path: Path) -> dict[str, str]:
    """Parse a .env file into a dict. Ignores comments and blank lines."""
    values: dict[str, str] = {}
    if not env_path.is_file():
        return values
    text = env_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
        if not match:
            continue
        key = match.group(1)
        val = match.group(2).strip()
        # Strip surrounding quotes if present
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        values[key] = val
    return values


def _build_sources(env: dict[str, str]) -> dict[str, SourceConfig]:
    """Build source configs from REFLECT_*_PATH + REFLECT_*_COUNT pairs."""
    sources: dict[str, SourceConfig] = {}

    # Named sources: MEMORY, PROVERBS, RULES
    for name in ("MEMORY", "PROVERBS", "RULES"):
        path_key = f"REFLECT_{name}_PATH"
        count_key = f"REFLECT_{name}_COUNT"
        path_val = env.get(path_key, "").strip()
        if not path_val:
            continue
        count_val = env.get(count_key, "5")
        try:
            count = int(count_val)
        except ValueError:
            raise ValueError(f"{count_key}={count_val!r} is not a valid integer") from None
        sources[name.lower()] = SourceConfig(path=_expand_path(path_val), count=count)

    # Extra sources: REFLECT_EXTRA_SOURCE_N + REFLECT_EXTRA_COUNT_N
    extra_pattern = re.compile(r"^REFLECT_EXTRA_SOURCE_(\d+)$")
    for key, val in env.items():
        m = extra_pattern.match(key)
        if not m or not val.strip():
            continue
        idx = m.group(1)
        count_key = f"REFLECT_EXTRA_COUNT_{idx}"
        count_val = env.get(count_key, "2")
        try:
            count = int(count_val)
        except ValueError:
            raise ValueError(f"{count_key}={count_val!r} is not a valid integer") from None
        sources[f"extra_{idx}"] = SourceConfig(path=_expand_path(val), count=count)

    return sources


def load_config(env_path: Path | None = None) -> ReflectConfig:
    """Load reflection config from a .env file.

    Falls back to sensible defaults when env_path is None or file is missing.
    """
    default_env_path = Path.home() / ".reflect.env"
    effective_path = env_path or default_env_path

    env = _parse_env_file(effective_path)

    session_dir_raw = env.get("REFLECT_SESSION_DIR", "~/.claude/projects/")
    session_dir = _expand_path(session_dir_raw)

    depth_raw = env.get("REFLECT_SESSION_DEPTH", "10")
    try:
        session_depth = int(depth_raw)
    except ValueError:
        raise ValueError(f"REFLECT_SESSION_DEPTH={depth_raw!r} is not a valid integer") from None

    sources = _build_sources(env)

    model = env.get("REFLECT_MODEL", "grok-4-1-fast-reasoning")

    api_key_raw = env.get("REFLECT_API_KEY_FILE", "").strip()
    api_key_file = _expand_path(api_key_raw) if api_key_raw else None

    return ReflectConfig(
        session_dir=session_dir,
        session_depth=session_depth,
        sources=sources,
        model=model,
        api_key_file=api_key_file,
    )
