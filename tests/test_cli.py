"""Tests for the CLI entry point."""

import json
from pathlib import Path
from unittest.mock import patch

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


def _write_api_key_file(directory: Path) -> Path:
    """Create a minimal API key file."""
    key_file = directory / ".api.env"
    key_file.write_text("API_KEY=sk_test_12345\n")
    return key_file


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
    @patch("agent_reflections.cli.call_mercury", return_value="Layer 1 reflection output")
    def test_prints_mercury_response(
        self, mock_call: object, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write_session_file(tmp_path)
        key_file = _write_api_key_file(tmp_path)
        env = tmp_path / ".reflect.env"
        env.write_text(
            f"REFLECT_SESSION_DIR={tmp_path}\n"
            f"REFLECT_API_KEY_FILE={key_file}\n"
        )

        main(["--problem", "why is the sky blue", "--config", str(env)])

        captured = capsys.readouterr()
        assert "Layer 1 reflection output" in captured.out

    def test_exits_when_no_api_key_file_configured(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write_session_file(tmp_path)
        env = tmp_path / ".reflect.env"
        env.write_text(f"REFLECT_SESSION_DIR={tmp_path}\n")

        with pytest.raises(SystemExit) as exc_info:
            main(["--problem", "test", "--config", str(env)])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "REFLECT_API_KEY_FILE not configured" in captured.err

    def test_exits_when_api_key_file_missing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        _write_session_file(tmp_path)
        env = tmp_path / ".reflect.env"
        env.write_text(
            f"REFLECT_SESSION_DIR={tmp_path}\n"
            f"REFLECT_API_KEY_FILE={tmp_path / 'nonexistent.env'}\n"
        )

        with pytest.raises(SystemExit) as exc_info:
            main(["--problem", "test", "--config", str(env)])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "API key file not found" in captured.err

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

    @patch("agent_reflections.cli.call_mercury", return_value="reflection text")
    def test_uses_default_config_when_not_specified(
        self,
        mock_call: object,
        tmp_path: Path,
        capsys: pytest.CaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _write_session_file(tmp_path)
        key_file = _write_api_key_file(tmp_path)
        env = tmp_path / ".reflect.env"
        env.write_text(
            f"REFLECT_SESSION_DIR={tmp_path}\n"
            f"REFLECT_API_KEY_FILE={key_file}\n"
        )

        # Override the default path to our tmp env
        monkeypatch.setattr(
            "agent_reflections.cli.load_config",
            lambda p: __import__("agent_reflections.config", fromlist=["load_config"]).load_config(
                env if p is None else p
            ),
        )

        main(["--problem", "test"])

        captured = capsys.readouterr()
        assert "reflection text" in captured.out

    @patch("agent_reflections.cli.call_mercury")
    def test_mercury_error_exits_with_error(
        self, mock_call: object, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        from agent_reflections.mercury import MercuryError

        mock_call.side_effect = MercuryError("HTTP 500 from server")  # type: ignore[attr-defined]
        _write_session_file(tmp_path)
        key_file = _write_api_key_file(tmp_path)
        env = tmp_path / ".reflect.env"
        env.write_text(
            f"REFLECT_SESSION_DIR={tmp_path}\n"
            f"REFLECT_API_KEY_FILE={key_file}\n"
        )

        with pytest.raises(SystemExit) as exc_info:
            main(["--problem", "test", "--config", str(env)])

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "HTTP 500" in captured.err
