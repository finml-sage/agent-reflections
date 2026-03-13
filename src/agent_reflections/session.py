"""JSONL session parser for Claude Code session files."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_EXTRACTABLE_TYPES = {"user", "assistant"}
_SKIP_BLOCK_TYPES = {"tool_use", "tool_result"}
_UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)


@dataclass
class SessionExtract:
    """Extracted content from a session file."""

    session_path: Path
    exchanges: list[Exchange] = field(default_factory=list)

    def as_text(self) -> str:
        """Render the extract as a labeled text block."""
        if not self.exchanges:
            return "[No exchanges extracted]"
        parts: list[str] = []
        for ex in self.exchanges:
            parts.append(f"--- {ex.role.upper()} ---")
            parts.append(ex.content)
            if ex.thinking:
                parts.append(f"  [thinking] {ex.thinking}")
        return "\n\n".join(parts)


@dataclass
class Exchange:
    """A single extracted exchange (user prompt or assistant response)."""

    role: str
    content: str
    thinking: str = ""


def find_latest_session(session_dir: Path) -> Path:
    """Find the most recently modified .jsonl session file (recursive, UUID-named only)."""
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session directory does not exist: {session_dir}")

    candidates: list[tuple[float, Path]] = []
    for jsonl_path in session_dir.rglob("*.jsonl"):
        if "subagents" in jsonl_path.parts:
            continue
        if not _UUID_PATTERN.match(jsonl_path.stem):
            continue
        candidates.append((jsonl_path.stat().st_mtime, jsonl_path))
    if not candidates:
        raise FileNotFoundError(f"No .jsonl session files found under: {session_dir}")

    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def _extract_content_from_blocks(blocks: list[dict]) -> tuple[str, str]:
    """Extract (content_text, thinking_text) from content blocks."""
    text_parts: list[str] = []
    thinking_parts: list[str] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type in _SKIP_BLOCK_TYPES:
            continue
        elif block_type == "thinking":
            thinking = block.get("thinking", "").strip()
            if thinking:
                thinking_parts.append(thinking)
        elif block_type == "text":
            text = block.get("text", "").strip()
            if text:
                text_parts.append(text)

    return "\n".join(text_parts), "\n".join(thinking_parts)

def _is_tool_only_entry(entry: dict) -> bool:
    """Check if a user entry contains only tool_result blocks (no human text)."""
    msg = entry.get("message", {})
    content = msg.get("content", "")
    if isinstance(content, str):
        return False
    if not isinstance(content, list):
        return True
    return all(
        isinstance(b, dict) and b.get("type") in _SKIP_BLOCK_TYPES
        for b in content
    )


def _parse_entry(entry: dict) -> Exchange | None:
    """Parse a single JSONL entry into an Exchange, or None if filtered out."""
    entry_type = entry.get("type")
    if entry_type not in _EXTRACTABLE_TYPES:
        return None

    msg = entry.get("message", {})
    role = msg.get("role", entry_type)
    content = msg.get("content", "")

    if isinstance(content, str):
        text = content.strip()
        if not text:
            return None
        return Exchange(role=role, content=text)

    if isinstance(content, list):
        if role == "user" and _is_tool_only_entry(entry):
            return None
        text, thinking = _extract_content_from_blocks(content)
        if not text and not thinking:
            return None
        return Exchange(role=role, content=text, thinking=thinking)

    return None


def extract_session(session_path: Path, depth: int = 10) -> SessionExtract:
    """Extract the last N meaningful exchanges from a session JSONL file."""
    if depth < 1:
        raise ValueError(f"Depth must be >= 1, got {depth}")
    if not session_path.is_file():
        raise FileNotFoundError(f"Session file not found: {session_path}")

    all_exchanges: list[Exchange] = []
    with open(session_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue  # Skip malformed lines
            exchange = _parse_entry(entry)
            if exchange is not None:
                all_exchanges.append(exchange)

    recent = all_exchanges[-depth:]
    return SessionExtract(session_path=session_path, exchanges=recent)
