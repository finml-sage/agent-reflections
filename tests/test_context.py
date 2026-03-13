"""Tests for context bundle assembly."""

import json
from pathlib import Path

import pytest

from agent_reflections.config import ReflectConfig, SourceConfig
from agent_reflections.context import (
    ContextBundle,
    _collect_md_files,
    _read_file_safe,
    _sample_source,
    assemble_context,
)
from agent_reflections.session import SessionExtract


def _write_session_file(directory: Path, content: str | None = None) -> Path:
    """Create a minimal valid session JSONL file in a directory."""
    session_file = directory / "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
    if content is None:
        entries = [
            {"type": "user", "message": {"role": "user", "content": "Hello"}},
            {
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi there!"}],
                },
            },
        ]
        content = "\n".join(json.dumps(e) for e in entries)
    session_file.write_text(content)
    return session_file


class TestCollectMdFiles:
    def test_collects_md_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("# A")
        (tmp_path / "b.md").write_text("# B")
        (tmp_path / "c.txt").write_text("not md")
        result = _collect_md_files(tmp_path)
        assert len(result) == 2
        assert all(p.suffix == ".md" for p in result)

    def test_searches_recursively(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.md").write_text("# Nested")
        result = _collect_md_files(tmp_path)
        assert len(result) == 1

    def test_skips_hidden_files(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden.md").write_text("# Hidden")
        (tmp_path / "visible.md").write_text("# Visible")
        result = _collect_md_files(tmp_path)
        assert len(result) == 1
        assert result[0].name == "visible.md"

    def test_skips_hidden_directories(self, tmp_path: Path) -> None:
        hidden_dir = tmp_path / ".hidden"
        hidden_dir.mkdir()
        (hidden_dir / "file.md").write_text("# Hidden dir file")
        (tmp_path / "visible.md").write_text("# Visible")
        result = _collect_md_files(tmp_path)
        assert len(result) == 1

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            _collect_md_files(tmp_path / "nonexistent")

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        result = _collect_md_files(tmp_path)
        assert result == []


class TestReadFileSafe:
    def test_reads_normal_file(self, tmp_path: Path) -> None:
        f = tmp_path / "test.md"
        f.write_text("Hello world")
        assert _read_file_safe(f) == "Hello world"

    def test_truncates_large_file(self, tmp_path: Path) -> None:
        f = tmp_path / "large.md"
        f.write_text("x" * 100_000)
        result = _read_file_safe(f, max_bytes=1000)
        assert len(result) < 1100  # 1000 + truncation message
        assert "truncated" in result

    def test_binary_file_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "binary.md"
        f.write_bytes(b"\x80\x81\x82\xff\xfe")
        with pytest.raises(ValueError, match="not valid UTF-8"):
            _read_file_safe(f)


class TestSampleSource:
    def test_samples_correct_count(self, tmp_path: Path) -> None:
        for i in range(10):
            (tmp_path / f"file_{i}.md").write_text(f"# File {i}")
        source = SourceConfig(path=tmp_path, count=3)
        result = _sample_source(source)
        assert len(result) == 3

    def test_samples_all_when_fewer_than_count(self, tmp_path: Path) -> None:
        (tmp_path / "only.md").write_text("# Only one")
        source = SourceConfig(path=tmp_path, count=5)
        result = _sample_source(source)
        assert len(result) == 1

    def test_empty_source_raises(self, tmp_path: Path) -> None:
        source = SourceConfig(path=tmp_path, count=3)
        with pytest.raises(FileNotFoundError, match="No .md files"):
            _sample_source(source)


class TestAssembleContext:
    def test_assembles_with_session_only(self, tmp_path: Path) -> None:
        _write_session_file(tmp_path)
        config = ReflectConfig(session_dir=tmp_path, session_depth=10)
        bundle = assemble_context(config)
        assert isinstance(bundle, ContextBundle)
        assert len(bundle.session.exchanges) > 0
        assert bundle.sources == {}

    def test_assembles_with_sources(self, tmp_path: Path) -> None:
        _write_session_file(tmp_path)
        src_dir = tmp_path / "memories"
        src_dir.mkdir()
        for i in range(5):
            (src_dir / f"mem_{i}.md").write_text(f"# Memory {i}\nContent {i}")

        config = ReflectConfig(
            session_dir=tmp_path,
            session_depth=10,
            sources={"memory": SourceConfig(path=src_dir, count=2)},
        )
        bundle = assemble_context(config)
        assert "memory" in bundle.sources
        assert len(bundle.sources["memory"]) == 2

    def test_missing_session_dir_raises(self, tmp_path: Path) -> None:
        config = ReflectConfig(
            session_dir=tmp_path / "nonexistent", session_depth=10
        )
        with pytest.raises(FileNotFoundError):
            assemble_context(config)

    def test_as_text_includes_all_sections(self, tmp_path: Path) -> None:
        _write_session_file(tmp_path)
        src_dir = tmp_path / "rules"
        src_dir.mkdir()
        (src_dir / "rule1.md").write_text("# Rule 1\nDo the thing.")

        config = ReflectConfig(
            session_dir=tmp_path,
            session_depth=10,
            sources={"rules": SourceConfig(path=src_dir, count=1)},
        )
        bundle = assemble_context(config)
        text = bundle.as_text()
        assert "=== SESSION CONTEXT ===" in text
        assert "=== SOURCE: RULES ===" in text
        assert "Do the thing." in text


class TestContextBundle:
    def test_empty_bundle_text(self) -> None:
        bundle = ContextBundle(
            session=SessionExtract(session_path=Path("/dev/null"))
        )
        text = bundle.as_text()
        assert "=== SESSION CONTEXT ===" in text
        assert "[No exchanges extracted]" in text
