"""Tests for the CLI entry point."""

import json
from pathlib import Path

import pytest

from agent_reflections.cli import _build_parser, main


def _write_session_file(directory: Path) -> Path:
    """Create a minimal valid session JSONL file in a directory."""
    session_file = directory / "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee.jsonl"
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
    session_file.write_text("\n".join(json.dumps(e) for e in entries))
    return session_file


class TestBuildParser:
    def test_requires_problem_argument(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_parses_problem(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--problem", "test problem"])
        assert args.problem == "test problem"

    def test_config_defaults_to_none(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--problem", "x"])
        assert args.config is None

    def test_parses_config_path(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["--problem", "x", "--config", "/tmp/my.env"])
        assert args.config == Path("/tmp/my.env")


class TestMain:
    def test_prints_problem_and_context(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        _write_session_file(tmp_path)
        env = tmp_path / ".reflect.env"
        env.write_text(f"REFLECT_SESSION_DIR={tmp_path}\n")

        main(["--problem", "why is the sky blue", "--config", str(env)])

        captured = capsys.readouterr()
        assert "Problem: why is the sky blue" in captured.out
        assert "=== SESSION CONTEXT ===" in captured.out
        assert "Layer 1/2/3 pipeline not yet wired" in captured.out

    def test_prints_placeholder_message(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        _write_session_file(tmp_path)
        env = tmp_path / ".reflect.env"
        env.write_text(f"REFLECT_SESSION_DIR={tmp_path}\n")

        main(["--problem", "test", "--config", str(env)])

        captured = capsys.readouterr()
        assert "--- Layer 1/2/3 pipeline not yet wired (Module 2) ---" in captured.out

    def test_missing_session_dir_exits_with_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        env = tmp_path / ".reflect.env"
        env.write_text(f"REFLECT_SESSION_DIR={tmp_path / 'nonexistent'}\n")

        with pytest.raises(SystemExit) as exc_info:
            main(["--problem", "test", "--config", str(env)])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error:" in captured.err

    def test_uses_default_config_when_not_specified(
        self, tmp_path: Path, capsys: pytest.CaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _write_session_file(tmp_path)
        # Point the default session dir to our tmp_path via env override
        env = tmp_path / ".reflect.env"
        env.write_text(f"REFLECT_SESSION_DIR={tmp_path}\n")

        # Override the default path to our tmp env
        monkeypatch.setattr(
            "agent_reflections.cli.load_config",
            lambda p: __import__("agent_reflections.config", fromlist=["load_config"]).load_config(
                env if p is None else p
            ),
        )

        main(["--problem", "test"])

        captured = capsys.readouterr()
        assert "Problem: test" in captured.out
