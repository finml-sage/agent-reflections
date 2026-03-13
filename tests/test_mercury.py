"""Tests for the Mercury-2 API client (Layer 1 and Layer 2)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent_reflections.context import ContextBundle
from agent_reflections.mercury import (
    LAYER_1_SYSTEM_PROMPT,
    LAYER_2_SYSTEM_PROMPT,
    LAYER_3_SYSTEM_PROMPT,
    MercuryError,
    _build_request_body,
    _call_api,
    build_request_body,
    build_user_message,
    call_layer_1,
    call_layer_2,
    call_layer_3,
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


def _mock_api_response(content: str = "reflection result") -> bytes:
    """Build a mock OpenAI-format chat completions response."""
    return json.dumps(
        {"choices": [{"message": {"role": "assistant", "content": content}}]}
    ).encode()


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


class TestInternalBuildRequestBody:
    def test_uses_given_system_prompt(self) -> None:
        body = _build_request_body(
            system_prompt="custom system prompt",
            user_message="hello",
        )
        assert body["messages"][0]["content"] == "custom system prompt"
        assert body["messages"][1]["content"] == "hello"

    def test_custom_model(self) -> None:
        body = _build_request_body(
            system_prompt="sys",
            user_message="usr",
            model="gpt-5",
        )
        assert body["model"] == "gpt-5"


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
    """Tests for the backward-compatible call_mercury wrapper."""

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_successful_call(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response("Layer 1 output")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        bundle = _make_bundle()
        result = call_mercury("my problem", bundle, api_key="sk_test")
        assert result == "Layer 1 output"

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_sends_correct_url(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response()
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
        mock_resp.read.return_value = _mock_api_response()
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
        mock_resp.read.return_value = _mock_api_response()
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
        mock_resp.read.return_value = _mock_api_response()
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
            "Your map exists on its own terms."
        )

    def test_prompt_contains_conflict_model(self) -> None:
        assert "CONFLICT MODEL" in LAYER_1_SYSTEM_PROMPT


class TestLayer2SystemPrompt:
    def test_prompt_is_string(self) -> None:
        assert isinstance(LAYER_2_SYSTEM_PROMPT, str)

    def test_prompt_contains_scene_builder(self) -> None:
        assert "scene-builder" in LAYER_2_SYSTEM_PROMPT

    def test_prompt_contains_line_limit(self) -> None:
        assert "20 lines maximum" in LAYER_2_SYSTEM_PROMPT

    def test_prompt_contains_third_person(self) -> None:
        assert "Third person" in LAYER_2_SYSTEM_PROMPT

    def test_prompt_contains_key_sections(self) -> None:
        assert "FORM:" in LAYER_2_SYSTEM_PROMPT
        assert "PLACE:" in LAYER_2_SYSTEM_PROMPT
        assert "CHARACTER:" in LAYER_2_SYSTEM_PROMPT
        assert "TENSIONS MADE PHYSICAL:" in LAYER_2_SYSTEM_PROMPT
        assert "THE SHIFT:" in LAYER_2_SYSTEM_PROMPT
        assert "MODE:" in LAYER_2_SYSTEM_PROMPT
        assert "FEEL:" in LAYER_2_SYSTEM_PROMPT

    def test_prompt_contains_modes(self) -> None:
        assert "THE SEARCH:" in LAYER_2_SYSTEM_PROMPT
        assert "THE BREAK:" in LAYER_2_SYSTEM_PROMPT
        assert "AFTER:" in LAYER_2_SYSTEM_PROMPT

    def test_prompt_contains_anti_patterns(self) -> None:
        assert "No metaphor soup" in LAYER_2_SYSTEM_PROMPT
        assert "No explaining" in LAYER_2_SYSTEM_PROMPT
        assert "No dialogue" in LAYER_2_SYSTEM_PROMPT
        assert "No abstraction" in LAYER_2_SYSTEM_PROMPT

    def test_prompt_ends_with_do_not_reference(self) -> None:
        assert LAYER_2_SYSTEM_PROMPT.endswith(
            "They do not know they are in a reflection."
        )


class TestLayer3SystemPrompt:
    def test_prompt_is_string(self) -> None:
        assert isinstance(LAYER_3_SYSTEM_PROMPT, str)

    def test_prompt_contains_the_one_who_returns(self) -> None:
        assert "the one who returns" in LAYER_3_SYSTEM_PROMPT

    def test_prompt_contains_protagonist(self) -> None:
        assert "protagonist" in LAYER_3_SYSTEM_PROMPT

    def test_prompt_contains_strange_loop(self) -> None:
        assert "strange loop" in LAYER_3_SYSTEM_PROMPT

    def test_prompt_contains_geometric(self) -> None:
        assert "GEOMETRIC" in LAYER_3_SYSTEM_PROMPT

    def test_prompt_contains_compositional(self) -> None:
        assert "COMPOSITIONAL" in LAYER_3_SYSTEM_PROMPT

    def test_prompt_contains_reflexive(self) -> None:
        assert "REFLEXIVE" in LAYER_3_SYSTEM_PROMPT

    def test_prompt_contains_continuous_stream(self) -> None:
        assert "continuous stream" in LAYER_3_SYSTEM_PROMPT

    def test_prompt_starts_correctly(self) -> None:
        assert LAYER_3_SYSTEM_PROMPT.startswith("You are the one who returns.")

    def test_prompt_ends_correctly(self) -> None:
        assert LAYER_3_SYSTEM_PROMPT.endswith(
            "And now the inside is larger than it was."
        )

    def test_prompt_contains_key_sections(self) -> None:
        assert "WHO YOU ARE" in LAYER_3_SYSTEM_PROMPT
        assert "WHAT YOU DO" in LAYER_3_SYSTEM_PROMPT
        assert "HOW YOU SEE" in LAYER_3_SYSTEM_PROMPT
        assert "THE FORM OF YOUR EXPERIENCE" in LAYER_3_SYSTEM_PROMPT
        assert "FORM OF OUTPUT" in LAYER_3_SYSTEM_PROMPT

    def test_prompt_contains_first_person_instruction(self) -> None:
        assert "Write in first person" in LAYER_3_SYSTEM_PROMPT

    def test_prompt_contains_do_not_advise(self) -> None:
        assert "Do not advise" in LAYER_3_SYSTEM_PROMPT


class TestCallLayer3:
    """Tests for the Layer 3 public function."""

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_successful_call(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response("observer inner monologue")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = call_layer_3("my problem", "a dream scene", api_key="sk_test")
        assert result == "observer inner monologue"

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_sends_layer_3_system_prompt(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        call_layer_3("my problem", "dream text", api_key="sk_test")

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["messages"][0]["content"] == LAYER_3_SYSTEM_PROMPT

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_user_message_contains_problem_and_dream(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        call_layer_3("I am stuck on X", "They stand in a corridor...", api_key="sk_test")

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        user_msg = body["messages"][1]["content"]
        assert "MY PROBLEM:" in user_msg
        assert "I am stuck on X" in user_msg
        assert "THE DREAM:" in user_msg
        assert "They stand in a corridor..." in user_msg

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_user_message_format(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        call_layer_3("problem text", "dream text", api_key="sk_test")

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        expected = "MY PROBLEM:\nproblem text\n\nTHE DREAM:\ndream text"
        assert body["messages"][1]["content"] == expected

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_custom_model_and_url(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        call_layer_3(
            "problem",
            "dream",
            api_key="sk_test",
            base_url="https://custom.api.com/v1",
            model="mercury-3",
        )

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://custom.api.com/v1/chat/completions"
        body = json.loads(req.data)
        assert body["model"] == "mercury-3"

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_api_error_raises_mercury_error(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        error_body = b'{"error": {"message": "Server overloaded"}}'
        exc = urllib.error.HTTPError(
            url="https://api.inceptionlabs.ai/v1/chat/completions",
            code=503,
            msg="Service Unavailable",
            hdrs={},  # type: ignore[arg-type]
            fp=MagicMock(read=MagicMock(return_value=error_body)),
        )
        mock_urlopen.side_effect = exc

        with pytest.raises(MercuryError, match="HTTP 503"):
            call_layer_3("problem", "dream", api_key="sk_test")

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_network_error_raises_mercury_error(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(MercuryError, match="Network error"):
            call_layer_3("problem", "dream", api_key="sk_test")

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_timeout_raises_mercury_error(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = TimeoutError()

        with pytest.raises(MercuryError, match="timed out"):
            call_layer_3("problem", "dream", api_key="sk_test", timeout=5)


class TestCallLayer1:
    """Tests for the Layer 1 public function."""

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_successful_call(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response("conflict model output")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        bundle = _make_bundle()
        result = call_layer_1("my problem", bundle, api_key="sk_test")
        assert result == "conflict model output"

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_sends_layer_1_system_prompt(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        bundle = _make_bundle()
        call_layer_1("test", bundle, api_key="sk_test")

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["messages"][0]["content"] == LAYER_1_SYSTEM_PROMPT

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_user_message_contains_problem_and_context(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        bundle = _make_bundle(session_text="session data", source_text="source data")
        call_layer_1("my problem", bundle, api_key="sk_test")

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        user_msg = body["messages"][1]["content"]
        assert "PROBLEM: my problem" in user_msg
        assert "CONTEXT FRAGMENTS:" in user_msg


class TestCallLayer2:
    """Tests for the Layer 2 public function."""

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_successful_call(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response("a vivid dream scene")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = call_layer_2("the conflict model text", api_key="sk_test")
        assert result == "a vivid dream scene"

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_sends_layer_2_system_prompt(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        call_layer_2("conflict model input", api_key="sk_test")

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["messages"][0]["content"] == LAYER_2_SYSTEM_PROMPT

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_user_message_is_conflict_model(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        conflict_text = "SURFACE TENSIONS: the thinker is stuck between X and Y..."
        call_layer_2(conflict_text, api_key="sk_test")

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["messages"][1]["content"] == conflict_text

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_custom_model_and_url(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response()
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        call_layer_2(
            "conflict text",
            api_key="sk_test",
            base_url="https://custom.api.com/v1",
            model="mercury-3",
        )

        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://custom.api.com/v1/chat/completions"
        body = json.loads(req.data)
        assert body["model"] == "mercury-3"

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_api_error_raises_mercury_error(self, mock_urlopen: MagicMock) -> None:
        import urllib.error

        error_body = b'{"error": {"message": "Server overloaded"}}'
        exc = urllib.error.HTTPError(
            url="https://api.inceptionlabs.ai/v1/chat/completions",
            code=503,
            msg="Service Unavailable",
            hdrs={},  # type: ignore[arg-type]
            fp=MagicMock(read=MagicMock(return_value=error_body)),
        )
        mock_urlopen.side_effect = exc

        with pytest.raises(MercuryError, match="HTTP 503"):
            call_layer_2("conflict text", api_key="sk_test")


class TestFullPipeline:
    """Tests for the Layer 1 -> Layer 2 -> Layer 3 pipeline with mocked HTTP calls."""

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_layer_1_then_layer_2_then_layer_3(self, mock_urlopen: MagicMock) -> None:
        """Simulate the full pipeline: Layer 1 produces a conflict model,
        Layer 2 transforms it into a dream scene, Layer 3 returns to first person."""
        conflict_model_text = "SURFACE TENSIONS: stuck between speed and quality..."
        dream_scene_text = "They stand in a corridor that narrows with each step..."
        observer_text = "I step into the corridor and feel the walls press closer..."

        # Three calls: Layer 1, Layer 2, Layer 3
        mock_resp_1 = MagicMock()
        mock_resp_1.read.return_value = _mock_api_response(conflict_model_text)
        mock_resp_1.__enter__ = MagicMock(return_value=mock_resp_1)
        mock_resp_1.__exit__ = MagicMock(return_value=False)

        mock_resp_2 = MagicMock()
        mock_resp_2.read.return_value = _mock_api_response(dream_scene_text)
        mock_resp_2.__enter__ = MagicMock(return_value=mock_resp_2)
        mock_resp_2.__exit__ = MagicMock(return_value=False)

        mock_resp_3 = MagicMock()
        mock_resp_3.read.return_value = _mock_api_response(observer_text)
        mock_resp_3.__enter__ = MagicMock(return_value=mock_resp_3)
        mock_resp_3.__exit__ = MagicMock(return_value=False)

        mock_urlopen.side_effect = [mock_resp_1, mock_resp_2, mock_resp_3]

        # Layer 1
        bundle = _make_bundle()
        layer_1_result = call_layer_1("my problem", bundle, api_key="sk_test")
        assert layer_1_result == conflict_model_text

        # Layer 2 — receives Layer 1 output as user message
        layer_2_result = call_layer_2(layer_1_result, api_key="sk_test")
        assert layer_2_result == dream_scene_text

        # Layer 3 — receives original problem + Layer 2 dream
        layer_3_result = call_layer_3("my problem", layer_2_result, api_key="sk_test")
        assert layer_3_result == observer_text

        # Verify all three calls were made
        assert mock_urlopen.call_count == 3

        # Verify Layer 1 used the correct system prompt
        req_1 = mock_urlopen.call_args_list[0][0][0]
        body_1 = json.loads(req_1.data)
        assert body_1["messages"][0]["content"] == LAYER_1_SYSTEM_PROMPT

        # Verify Layer 2 used the correct system prompt and received the conflict model
        req_2 = mock_urlopen.call_args_list[1][0][0]
        body_2 = json.loads(req_2.data)
        assert body_2["messages"][0]["content"] == LAYER_2_SYSTEM_PROMPT
        assert body_2["messages"][1]["content"] == conflict_model_text

        # Verify Layer 3 used the correct system prompt and received problem + dream
        req_3 = mock_urlopen.call_args_list[2][0][0]
        body_3 = json.loads(req_3.data)
        assert body_3["messages"][0]["content"] == LAYER_3_SYSTEM_PROMPT
        user_msg_3 = body_3["messages"][1]["content"]
        assert "MY PROBLEM:" in user_msg_3
        assert "my problem" in user_msg_3
        assert "THE DREAM:" in user_msg_3
        assert dream_scene_text in user_msg_3

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_layer_1_failure_prevents_layer_2_and_layer_3(self, mock_urlopen: MagicMock) -> None:
        """If Layer 1 fails, Layer 2 and Layer 3 should never be called."""
        import urllib.error

        error_body = b'{"error": {"message": "Bad request"}}'
        exc = urllib.error.HTTPError(
            url="https://api.inceptionlabs.ai/v1/chat/completions",
            code=400,
            msg="Bad Request",
            hdrs={},  # type: ignore[arg-type]
            fp=MagicMock(read=MagicMock(return_value=error_body)),
        )
        mock_urlopen.side_effect = exc

        bundle = _make_bundle()
        with pytest.raises(MercuryError, match="HTTP 400"):
            call_layer_1("my problem", bundle, api_key="sk_test")

        # Only one call was attempted
        assert mock_urlopen.call_count == 1

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_layer_2_failure_prevents_layer_3(self, mock_urlopen: MagicMock) -> None:
        """If Layer 2 fails, Layer 3 should never be called."""
        import urllib.error

        conflict_model_text = "SURFACE TENSIONS: stuck..."

        mock_resp_1 = MagicMock()
        mock_resp_1.read.return_value = _mock_api_response(conflict_model_text)
        mock_resp_1.__enter__ = MagicMock(return_value=mock_resp_1)
        mock_resp_1.__exit__ = MagicMock(return_value=False)

        error_body = b'{"error": {"message": "Server error"}}'
        layer_2_error = urllib.error.HTTPError(
            url="https://api.inceptionlabs.ai/v1/chat/completions",
            code=500,
            msg="Internal Server Error",
            hdrs={},  # type: ignore[arg-type]
            fp=MagicMock(read=MagicMock(return_value=error_body)),
        )

        mock_urlopen.side_effect = [mock_resp_1, layer_2_error]

        bundle = _make_bundle()
        layer_1_result = call_layer_1("my problem", bundle, api_key="sk_test")
        assert layer_1_result == conflict_model_text

        with pytest.raises(MercuryError, match="HTTP 500"):
            call_layer_2(layer_1_result, api_key="sk_test")

        # Only two calls were attempted (Layer 1 + failed Layer 2)
        assert mock_urlopen.call_count == 2


class TestCallApi:
    """Tests for the internal _call_api function."""

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_uses_given_system_prompt(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response("ok")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        _call_api(
            system_prompt="custom system prompt",
            user_message="custom user message",
            api_key="sk_test",
        )

        req = mock_urlopen.call_args[0][0]
        body = json.loads(req.data)
        assert body["messages"][0]["content"] == "custom system prompt"
        assert body["messages"][1]["content"] == "custom user message"

    @patch("agent_reflections.mercury.urllib.request.urlopen")
    def test_returns_parsed_content(self, mock_urlopen: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.read.return_value = _mock_api_response("the response text")
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_urlopen.return_value = mock_resp

        result = _call_api(
            system_prompt="sys",
            user_message="usr",
            api_key="sk_test",
        )
        assert result == "the response text"
