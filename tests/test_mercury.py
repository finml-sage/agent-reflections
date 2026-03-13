"""Tests for the Mercury-2 API client (Layer 1)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_reflections.context import ContextBundle
from agent_reflections.mercury import (
    LAYER_1_SYSTEM_PROMPT,
    MercuryError,
    build_request_body,
    build_user_message,
    call_mercury,
    parse_response,
    read_api_key,
)
from agent_reflections.session import Exchange, SessionExtract


def _make_bundle(session_text: str = "test session", source_text: str = "test source") -> ContextBundle:
    """Create a minimal ContextBundle for testing."""
    session = SessionExtract(
        session_path=Path("/dev/null"),
        exchanges=[Exchange(role="user", content=session_text)],
    )
    return ContextBundle(session=session, sources={"memory": [source_text]})


class TestReadApiKey:
    def test_reads_api_key(self, tmp_path: Path) -> None:
        key_file = tmp_path / ".api.env"
        key_file.write_text("API_KEY=sk_test_12345\n")
        result = read_api_key(key_file)
        assert result == "sk_test_12345"

    def test_reads_quoted_key(self, tmp_path: Path) -> None:
        key_file = tmp_path / ".api.env"
        key_file.write_text('API_KEY="sk_quoted_key"\n')
        result = read_api_key(key_file)
        assert result == "sk_quoted_key"

    def test_ignores_comments_and_blanks(self, tmp_path: Path) -> None:
        key_file = tmp_path / ".api.env"
        key_file.write_text("# This is a comment\n\nAPI_KEY=sk_real_key\n")
        result = read_api_key(key_file)
        assert result == "sk_real_key"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="API key file not found"):
            read_api_key(tmp_path / "nonexistent.env")

    def test_missing_key_raises(self, tmp_path: Path) -> None:
        key_file = tmp_path / ".api.env"
        key_file.write_text("OTHER_VAR=hello\n")
        with pytest.raises(ValueError, match="No API_KEY found"):
            read_api_key(key_file)

    def test_empty_key_raises(self, tmp_path: Path) -> None:
        key_file = tmp_path / ".api.env"
        key_file.write_text("API_KEY=\n")
        with pytest.raises(ValueError, match="No API_KEY found"):
            read_api_key(key_file)

    def test_empty_file_raises(self, tmp_path: Path) -> None:
        key_file = tmp_path / ".api.env"
        key_file.write_text("")
        with pytest.raises(ValueError, match="No API_KEY found"):
            read_api_key(key_file)


class TestBuildUserMessage:
    def test_contains_problem(self) -> None:
        bundle = _make_bundle()
        msg = build_user_message("I am stuck", bundle)
        assert msg.startswith("PROBLEM: I am stuck")

    def test_contains_context_fragments_header(self) -> None:
        bundle = _make_bundle()
        msg = build_user_message("test problem", bundle)
        assert "CONTEXT FRAGMENTS:" in msg

    def test_contains_session_content(self) -> None:
        bundle = _make_bundle(session_text="my session data")
        msg = build_user_message("test", bundle)
        assert "my session data" in msg

    def test_contains_source_content(self) -> None:
        bundle = _make_bundle(source_text="proverb about patience")
        msg = build_user_message("test", bundle)
        assert "proverb about patience" in msg


class TestBuildRequestBody:
    def test_structure(self) -> None:
        bundle = _make_bundle()
        body = build_request_body("test problem", bundle)
        assert body["model"] == "mercury-2"
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][1]["role"] == "user"

    def test_system_prompt_is_layer_1(self) -> None:
        bundle = _make_bundle()
        body = build_request_body("test", bundle)
        assert body["messages"][0]["content"] == LAYER_1_SYSTEM_PROMPT

    def test_custom_model(self) -> None:
        bundle = _make_bundle()
        body = build_request_body("test", bundle, model="mercury-3")
        assert body["model"] == "mercury-3"

    def test_user_message_contains_problem(self) -> None:
        bundle = _make_bundle()
        body = build_request_body("my specific problem", bundle)
        assert "my specific problem" in body["messages"][1]["content"]


class TestParseResponse:
    def test_parses_valid_response(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "This is the reflection output.",
                    }
                }
            ]
        }
        result = parse_response(json.dumps(response).encode())
        assert result == "This is the reflection output."

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(MercuryError, match="Invalid JSON"):
            parse_response(b"not json at all")

    def test_api_error_dict_raises(self) -> None:
        response = {"error": {"message": "Rate limit exceeded", "type": "rate_limit"}}
        with pytest.raises(MercuryError, match="Rate limit exceeded"):
            parse_response(json.dumps(response).encode())

    def test_api_error_string_raises(self) -> None:
        response = {"error": "Something went wrong"}
        with pytest.raises(MercuryError, match="Something went wrong"):
            parse_response(json.dumps(response).encode())

    def test_empty_choices_raises(self) -> None:
        response = {"choices": []}
        with pytest.raises(MercuryError, match="no choices"):
            parse_response(json.dumps(response).encode())

    def test_missing_choices_raises(self) -> None:
        response = {"id": "chatcmpl-123"}
        with pytest.raises(MercuryError, match="Unexpected API response"):
            parse_response(json.dumps(response).encode())

    def test_empty_content_raises(self) -> None:
        response = {"choices": [{"message": {"role": "assistant", "content": ""}}]}
        with pytest.raises(MercuryError, match="empty content"):
            parse_response(json.dumps(response).encode())

    def test_null_content_raises(self) -> None:
        response = {"choices": [{"message": {"role": "assistant", "content": None}}]}
        with pytest.raises(MercuryError, match="empty content"):
            parse_response(json.dumps(response).encode())

    def test_missing_message_key_raises(self) -> None:
        response = {"choices": [{"index": 0}]}
        with pytest.raises(MercuryError, match="Unexpected API response"):
            parse_response(json.dumps(response).encode())


class TestCallMercury:
    def _mock_response(self, content: str = "reflection result") -> bytes:
        return json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": content}}]}
        ).encode()

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_successful_call(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = self._mock_response("Layer 1 output")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        bundle = _make_bundle()
        result = call_mercury("my problem", bundle, api_key="sk_test")
        assert result == "Layer 1 output"

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_sends_correct_url(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = self._mock_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        bundle = _make_bundle()
        call_mercury("test", bundle, api_key="sk_test", base_url="https://example.com/v1")

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://example.com/v1/chat/completions"

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_sends_auth_header(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = self._mock_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        bundle = _make_bundle()
        call_mercury("test", bundle, api_key="sk_my_secret_key")

        req = mock_urlopen.call_args[0][0]
        assert req.get_header("Authorization") == "Bearer sk_my_secret_key"
        assert req.get_header("Content-type") == "application/json"

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_sends_correct_body(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = self._mock_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        bundle = _make_bundle()
        call_mercury("test problem", bundle, api_key="sk_test", model="mercury-2")

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["model"] == "mercury-2"
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == LAYER_1_SYSTEM_PROMPT
        assert "test problem" in body["messages"][1]["content"]

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_strips_trailing_slash_from_base_url(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = self._mock_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        bundle = _make_bundle()
        call_mercury("test", bundle, api_key="sk_test", base_url="https://example.com/v1/")

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://example.com/v1/chat/completions"

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_http_error_raises_mercury_error(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        error_body = b'{"error": {"message": "Unauthorized"}}'
        exc = urllib.error.HTTPError(
            url="https://api.inceptionlabs.ai/v1/chat/completions",
            code=401,
            msg="Unauthorized",
            hdrs={},  # type: ignore[arg-type]
            fp=MagicMock(read=MagicMock(return_value=error_body)),
        )
        mock_urlopen.side_effect = exc

        bundle = _make_bundle()
        with pytest.raises(MercuryError, match="HTTP 401"):
            call_mercury("test", bundle, api_key="bad_key")

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_url_error_raises_mercury_error(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        bundle = _make_bundle()
        with pytest.raises(MercuryError, match="Network error"):
            call_mercury("test", bundle, api_key="sk_test")

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_timeout_raises_mercury_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = TimeoutError()

        bundle = _make_bundle()
        with pytest.raises(MercuryError, match="timed out"):
            call_mercury("test", bundle, api_key="sk_test", timeout=5)


class TestSystemPrompt:
    def test_prompt_is_string(self) -> None:
        assert isinstance(LAYER_1_SYSTEM_PROMPT, str)

    def test_prompt_contains_key_sections(self) -> None:
        assert "SURFACE TENSIONS" in LAYER_1_SYSTEM_PROMPT
        assert "HIDDEN PATTERNS" in LAYER_1_SYSTEM_PROMPT
        assert "CONFLICTING GOALS" in LAYER_1_SYSTEM_PROMPT
        assert "WHAT IS BEING AVOIDED" in LAYER_1_SYSTEM_PROMPT
        assert "THE STRANGE CONNECTION" in LAYER_1_SYSTEM_PROMPT
        assert "EMOTIONAL UNDERCURRENT" in LAYER_1_SYSTEM_PROMPT
        assert "THE META-PATTERN" in LAYER_1_SYSTEM_PROMPT

    def test_prompt_starts_correctly(self) -> None:
        assert LAYER_1_SYSTEM_PROMPT.startswith(
            "You are a first-person inner voice"
        )

    def test_prompt_ends_correctly(self) -> None:
        assert LAYER_1_SYSTEM_PROMPT.endswith(
            "sees the loom while the rest of the mind sees only the thread."
        )

    def test_prompt_contains_conflict_model(self) -> None:
        assert "CONFLICT MODEL" in LAYER_1_SYSTEM_PROMPT
