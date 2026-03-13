"""Context bundle assembly — combines session extract with sampled sources."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

from .config import ReflectConfig, SourceConfig
from .session import SessionExtract, extract_session, find_latest_session


@dataclass
class ContextBundle:
    """Assembled context ready for the reflection pipeline."""

    session: SessionExtract
    sources: dict[str, list[str]] = field(default_factory=dict)

    def as_text(self) -> str:
        """Render the full context bundle as labeled text."""
        parts: list[str] = []

        parts.append("=== SESSION CONTEXT ===")
        parts.append(self.session.as_text())

        for name, contents in self.sources.items():
            parts.append(f"\n=== SOURCE: {name.upper()} ===")
            for i, content in enumerate(contents, 1):
                parts.append(f"--- {name} [{i}] ---")
                parts.append(content)

        return "\n\n".join(parts)


def _collect_md_files(directory: Path) -> list[Path]:
    """Collect readable .md files from a directory, skipping hidden files."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {directory}")

    md_files: list[Path] = []
    for p in directory.rglob("*.md"):
        # Skip hidden files and directories
        if any(part.startswith(".") for part in p.relative_to(directory).parts):
            continue
        if p.is_file():
            md_files.append(p)
    return md_files


def _read_file_safe(path: Path, max_bytes: int = 50_000) -> str:
    """Read a text file, truncating if too large. Raises on failure."""
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"Cannot read {path}: not valid UTF-8 text") from None
    if len(text) > max_bytes:
        text = text[:max_bytes] + f"\n[... truncated at {max_bytes} bytes]"
    return text


def _sample_source(source: SourceConfig) -> list[str]:
    """Read and randomly sample files from a source directory."""
    md_files = _collect_md_files(source.path)
    if not md_files:
        raise FileNotFoundError(f"No .md files found in source: {source.path}")

    sample_count = min(source.count, len(md_files))
    sampled = random.sample(md_files, sample_count)

    contents: list[str] = []
    for path in sampled:
        text = _read_file_safe(path)
        contents.append(text)
    return contents


def assemble_context(config: ReflectConfig) -> ContextBundle:
    """Assemble the full context bundle from config.

    1. Finds and parses the most recent session file
    2. Samples from each configured source directory
    3. Returns a ContextBundle ready for the reflection pipeline

    Raises:
        FileNotFoundError: If session dir or a required source dir is missing.
    """
    # Step 1: Session extraction
    session_path = find_latest_session(config.session_dir)
    session_extract = extract_session(session_path, depth=config.session_depth)

    # Step 2: Sample from each configured source
    sampled_sources: dict[str, list[str]] = {}
    for name, source_config in config.sources.items():
        contents = _sample_source(source_config)
        sampled_sources[name] = contents

    return ContextBundle(session=session_extract, sources=sampled_sources)
