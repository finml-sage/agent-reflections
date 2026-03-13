"""Tests for JSONL session parsing."""

from pathlib import Path

import pytest

from agent_reflections.session import (
    Exchange,
    SessionExtract,
    extract_session,
    find_latest_session,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestFindLatestSession:
    def test_finds_most_recent_file(self, tmp_path: Path) -> None:
        # Create two session files with different mtimes
        old = tmp_path / "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
        new = tmp_path / "11111111-2222-3333-4444-555555555555.jsonl"
        old.write_text('{"type":"user","message":{"role":"user","content":"old"}}')
        new.write_text('{"type":"user","message":{"role":"user","content":"new"}}')
        # Touch new file to ensure it's newer
        import os
        import time

        os.utime(old, (time.time() - 100, time.time() - 100))

        result = find_latest_session(tmp_path)
        assert result == new

    def test_searches_recursively(self, tmp_path: Path) -> None:
        nested = tmp_path / "project-a"
        nested.mkdir()
        f = nested / "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
        f.write_text('{"type":"user","message":{"role":"user","content":"hi"}}')
        result = find_latest_session(tmp_path)
        assert result == f

    def test_skips_subagent_files(self, tmp_path: Path) -> None:
        subdir = tmp_path / "proj" / "subagents"
        subdir.mkdir(parents=True)
        sub_file = subdir / "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
        sub_file.write_text('{"type":"user","message":{"role":"user","content":"sub"}}')

        # Only subagent file exists -- should raise
        with pytest.raises(FileNotFoundError, match="No .jsonl session files"):
            find_latest_session(tmp_path)

    def test_skips_non_uuid_files(self, tmp_path: Path) -> None:
        config_file = tmp_path / "config.jsonl"
        config_file.write_text('{"type":"config"}')
        with pytest.raises(FileNotFoundError, match="No .jsonl session files"):
            find_latest_session(tmp_path)

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            find_latest_session(tmp_path / "nonexistent")

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No .jsonl session files"):
            find_latest_session(tmp_path)


class TestExtractSession:
    def test_extracts_from_fixture(self) -> None:
        result = extract_session(FIXTURES / "sample_session.jsonl", depth=20)
        assert isinstance(result, SessionExtract)
        # Fixture has: 1 user str, 1 assistant+thinking, 1 assistant tool_use (filtered),
        # 1 user tool_result (filtered), 1 assistant text, 1 user str, 1 assistant+thinking
        # = 5 exchanges (2 user str + 3 assistant with text)
        assert len(result.exchanges) == 5

    def test_filters_tool_use_entries(self) -> None:
        result = extract_session(FIXTURES / "sample_session.jsonl", depth=20)
        for ex in result.exchanges:
            # No exchange should be a pure tool_use
            assert "tool_use" not in ex.content.lower() or ex.role != "assistant"

    def test_filters_tool_result_user_entries(self) -> None:
        result = extract_session(FIXTURES / "sample_session.jsonl", depth=20)
        for ex in result.exchanges:
            if ex.role == "user":
                assert "tool_result" not in ex.content

    def test_extracts_thinking_blocks(self) -> None:
        result = extract_session(FIXTURES / "sample_session.jsonl", depth=20)
        thinking_found = any(ex.thinking for ex in result.exchanges)
        assert thinking_found, "Expected at least one exchange with thinking"

    def test_respects_depth_limit(self) -> None:
        result = extract_session(FIXTURES / "sample_session.jsonl", depth=2)
        assert len(result.exchanges) == 2

    def test_depth_larger_than_total(self) -> None:
        result = extract_session(FIXTURES / "sample_session.jsonl", depth=100)
        assert len(result.exchanges) == 5  # All available exchanges

    def test_invalid_depth_raises(self) -> None:
        with pytest.raises(ValueError, match="Depth must be >= 1"):
            extract_session(FIXTURES / "sample_session.jsonl", depth=0)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Session file not found"):
            extract_session(tmp_path / "nonexistent.jsonl")

    def test_as_text_output(self) -> None:
        result = extract_session(FIXTURES / "sample_session.jsonl", depth=3)
        text = result.as_text()
        assert "--- ASSISTANT ---" in text or "--- USER ---" in text
        assert len(text) > 0

    def test_handles_malformed_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
        f.write_text(
            'not json at all\n'
            '{"type":"user","message":{"role":"user","content":"valid line"}}\n'
            '{broken json\n'
        )
        result = extract_session(f, depth=10)
        assert len(result.exchanges) == 1
        assert result.exchanges[0].content == "valid line"

    def test_empty_content_filtered(self, tmp_path: Path) -> None:
        f = tmp_path / "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
        f.write_text(
            '{"type":"user","message":{"role":"user","content":""}}\n'
            '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":""}]}}\n'
            '{"type":"user","message":{"role":"user","content":"real content"}}\n'
        )
        result = extract_session(f, depth=10)
        assert len(result.exchanges) == 1
        assert result.exchanges[0].content == "real content"


class TestSessionExtract:
    def test_empty_extract_as_text(self) -> None:
        extract = SessionExtract(session_path=Path("/dev/null"))
        assert extract.as_text() == "[No exchanges extracted]"
