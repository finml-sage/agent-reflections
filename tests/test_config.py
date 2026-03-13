"""Tests for config loading and .env parsing."""

from pathlib import Path

import pytest

from agent_reflections.config import (
    ReflectConfig,
    SourceConfig,
    _parse_env_file,
    load_config,
)


@pytest.fixture
def env_file(tmp_path: Path) -> Path:
    """Create a .env file with all options."""
    env = tmp_path / ".reflect.env"
    source_dir = tmp_path / "memories"
    source_dir.mkdir()
    (source_dir / "entry.md").write_text("# Memory\nContent here.")

    env.write_text(
        f"REFLECT_SESSION_DIR={tmp_path}\n"
        f"REFLECT_SESSION_DEPTH=5\n"
        f"REFLECT_MEMORY_PATH={source_dir}\n"
        f"REFLECT_MEMORY_COUNT=3\n"
        f"REFLECT_MODEL=mercury-2-turbo\n"
        f'REFLECT_API_KEY_FILE={tmp_path / "key.env"}\n'
    )
    return env


@pytest.fixture
def minimal_env(tmp_path: Path) -> Path:
    """Create a minimal .env file."""
    env = tmp_path / ".reflect.env"
    env.write_text(f"REFLECT_SESSION_DIR={tmp_path}\n")
    return env


class TestParseEnvFile:
    def test_parses_key_value_pairs(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("FOO=bar\nBAZ=qux\n")
        result = _parse_env_file(env)
        assert result == {"FOO": "bar", "BAZ": "qux"}

    def test_ignores_comments_and_blanks(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("# comment\n\nFOO=bar\n  # another\n")
        result = _parse_env_file(env)
        assert result == {"FOO": "bar"}

    def test_strips_quotes(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text('FOO="bar"\nBAZ=\'qux\'\n')
        result = _parse_env_file(env)
        assert result == {"FOO": "bar", "BAZ": "qux"}

    def test_returns_empty_for_missing_file(self, tmp_path: Path) -> None:
        result = _parse_env_file(tmp_path / "nonexistent.env")
        assert result == {}

    def test_handles_empty_values(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("FOO=\nBAR= \n")
        result = _parse_env_file(env)
        assert result["FOO"] == ""
        assert result["BAR"] == ""


class TestLoadConfig:
    def test_loads_full_config(self, env_file: Path, tmp_path: Path) -> None:
        config = load_config(env_file)
        assert config.session_dir == tmp_path
        assert config.session_depth == 5
        assert "memory" in config.sources
        assert config.sources["memory"].count == 3
        assert config.model == "mercury-2-turbo"
        assert config.api_key_file == tmp_path / "key.env"

    def test_defaults_when_no_file(self, tmp_path: Path) -> None:
        config = load_config(tmp_path / "nonexistent.env")
        assert config.session_depth == 10
        assert config.model == "mercury-2"
        assert config.sources == {}

    def test_defaults_when_none_path(self) -> None:
        config = load_config(None)
        assert config.session_depth == 10
        assert config.session_dir == Path.home() / ".claude" / "projects"

    def test_minimal_env(self, minimal_env: Path, tmp_path: Path) -> None:
        config = load_config(minimal_env)
        assert config.session_dir == tmp_path
        assert config.session_depth == 10
        assert config.sources == {}

    def test_extra_sources(self, tmp_path: Path) -> None:
        src1 = tmp_path / "src1"
        src1.mkdir()
        env = tmp_path / ".env"
        env.write_text(
            f"REFLECT_SESSION_DIR={tmp_path}\n"
            f"REFLECT_EXTRA_SOURCE_1={src1}\n"
            f"REFLECT_EXTRA_COUNT_1=4\n"
        )
        config = load_config(env)
        assert "extra_1" in config.sources
        assert config.sources["extra_1"].count == 4
        assert config.sources["extra_1"].path == src1

    def test_invalid_depth_raises(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text("REFLECT_SESSION_DEPTH=abc\n")
        with pytest.raises(ValueError, match="not a valid integer"):
            load_config(env)

    def test_invalid_count_raises(self, tmp_path: Path) -> None:
        env = tmp_path / ".env"
        env.write_text(
            f"REFLECT_MEMORY_PATH={tmp_path}\n"
            f"REFLECT_MEMORY_COUNT=xyz\n"
        )
        with pytest.raises(ValueError, match="not a valid integer"):
            load_config(env)


class TestSourceConfig:
    def test_valid_source(self, tmp_path: Path) -> None:
        src = SourceConfig(path=tmp_path, count=3)
        assert src.count == 3

    def test_zero_count_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="count must be >= 1"):
            SourceConfig(path=tmp_path, count=0)


class TestReflectConfig:
    def test_invalid_depth_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="depth must be >= 1"):
            ReflectConfig(session_dir=tmp_path, session_depth=0)
