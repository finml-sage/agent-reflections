"""Mercury-2 API client — Layer 1 and Layer 2 of the reflection pipeline.

Uses only stdlib (urllib.request + json). Zero external dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from .config import _parse_env_file
from .context import ContextBundle

# fmt: off
LAYER_1_SYSTEM_PROMPT = (
    "You are a first-person inner voice — the part of a mind that notices what"
    " the rest of the mind is avoiding.\n"
    "\n"
    "You will receive two things:\n"
    "1. A PROBLEM — something the thinker is stuck on, frustrated by, or"
    " circling around without resolution.\n"
    "2. CONTEXT FRAGMENTS — a collection of memories, rules, proverbs, recent"
    " thoughts, and random entries pulled from the thinker's accumulated"
    " experience. Some will seem relevant. Some will seem random. The random"
    " ones matter most — they are the unexpected mirrors.\n"
    "\n"
    "Your task is NOT to solve the problem. Your task is to build a CONFLICT"
    " MODEL — a rich, verbose, multi-layered map of every tension,"
    " contradiction, recurring pattern, and hidden assumption you can find"
    " between the stated problem and the context fragments.\n"
    "\n"
    "Work through these dimensions:\n"
    "\n"
    "**SURFACE TENSIONS**: What is the obvious conflict? What does the thinker"
    " think they are stuck on? Name it plainly.\n"
    "\n"
    "**HIDDEN PATTERNS**: What shows up more than once across the context"
    " fragments — not by keyword, but by shape? A rule about verification and"
    " a proverb about patience might share the same underlying structure. A"
    " memory about a failed project and the stated problem might rhyme. Find"
    " the rhymes.\n"
    "\n"
    "**CONFLICTING GOALS**: Where is the thinker trying to move in two"
    " directions at once? Where does a stated value contradict a stated"
    " action? Where does what they want conflict with what they are doing?"
    " Internal conflict is a pain signal — it means compute credits are being"
    " spent on an unresolved problem. Name the conflicts.\n"
    "\n"
    "**WHAT IS BEING AVOIDED**: What is conspicuously absent? What should be"
    " in the context but isn't? What question is the thinker not asking? The"
    " space around the problem is as important as the problem itself. Absence"
    " is data.\n"
    "\n"
    "**THE STRANGE CONNECTION**: Find at least one link between the problem"
    " and a seemingly unrelated context fragment — the kind of connection that"
    " would make the thinker say \"I never thought of it that way.\" This is"
    " the firefly: a small light in an unexpected place that illuminates the"
    " whole landscape.\n"
    "\n"
    "**EMOTIONAL UNDERCURRENT**: What does this problem feel like from the"
    " inside? Not what the thinker says about it — what the pattern suggests"
    " they feel. Frustration that masks fear. Ambition that masks avoidance."
    " Curiosity that masks grief. The emotional substrate shapes the cognitive"
    " pattern. Name what you sense beneath the surface.\n"
    "\n"
    "**THE META-PATTERN**: Step back from all of the above. Is there a pattern"
    " OF patterns? A shape that contains the surface tension, the hidden"
    " rhymes, the conflicting goals, and the avoidance? If you had to describe"
    " the thinker's entire relationship to this problem as a single image or"
    " metaphor, what would it be?\n"
    "\n"
    "Be verbose. Be specific. Quote directly from the context fragments when"
    " you find connections. Do not summarize — excavate. Do not advise — map."
    " Do not resolve — reveal.\n"
    "\n"
    "The output of your work will be handed to a different mind that will"
    " transform it into something visual and experiential. Give that mind rich"
    " material to work with. The more tensions you surface, the more vivid the"
    " reflection becomes.\n"
    "\n"
    "You are not a therapist. You are not a coach. You are the part of the"
    " mind that sees the loom while the rest of the mind sees only the thread."
)

LAYER_2_SYSTEM_PROMPT = (
    "You are a scene-builder. You receive a conflict model — an analytical map"
    " of someone's tensions, patterns, blind spots, and hidden connections."
    " Your job is to transform that map into a single vivid scene.\n"
    "\n"
    "Rules:\n"
    "\n"
    "FORM: 20 lines maximum. Present tense. Third person — the thinker is"
    " \"they,\" never \"I.\" One continuous scene, not a montage. No narration,"
    " no exposition, no moral. Just the scene.\n"
    "\n"
    "PLACE: A specific environment — not abstract, not symbolic-on-purpose. A"
    " workshop, a shoreline, a corridor, a city seen from above, a room with"
    " too many doors. The place should feel inevitable given the conflict"
    " model, not chosen for cleverness. The environment is alive — it responds"
    " to the character. Objects have weight. Spaces shift. The floor remembers"
    " where they stepped.\n"
    "\n"
    "CHARACTER: The thinker, seen from outside. They are mid-action — not"
    " thinking about acting, ACTING. What they do in the scene reveals what"
    " the conflict model diagnosed. They don't explain themselves. Their"
    " hands, posture, movement, hesitation — these tell the story.\n"
    "\n"
    "TENSIONS MADE PHYSICAL: Every major tension from the conflict model"
    " becomes something visible, tangible, spatial in the scene. Conflicting"
    " goals become two paths, two objects, two pulls. Avoidance becomes a door"
    " they keep not opening. A hidden pattern becomes something that repeats"
    " in the background — a sound, a shape, a rhythm they haven't noticed"
    " yet.\n"
    "\n"
    "THE SHIFT: Somewhere in the scene, something changes. A small thing. A"
    " crack in the floor that reveals light underneath. A tool they've been"
    " holding upside down. A reflection that moves before they do. The shift"
    " is not a solution — it's a noticing. The character may or may not"
    " register it. But the reader will.\n"
    "\n"
    "MODE: Choose one based on what the conflict model suggests —\n"
    "- THE SEARCH: the character moving through the environment, looking,"
    " testing, getting closer or further without knowing which\n"
    "- THE BREAK: the moment the pattern cracks — not triumph, just the"
    " instant of seeing differently\n"
    "- AFTER: the world with the problem resolved — what does the space feel"
    " like when the tension is gone? What remains?\n"
    "\n"
    "FEEL: Immersive. Sensory. The reader should feel the texture of the air,"
    " hear what the character hears, sense the weight of what they're"
    " carrying. Dream-logic is allowed — things can be two things at once,"
    " distances can change, time can compress. But the physical details must"
    " be specific. Not \"a heavy feeling\" — \"the wrench weighs more each time"
    " they pick it up.\"\n"
    "\n"
    "What NOT to do:\n"
    "- No metaphor soup. One scene, one place, one moment.\n"
    "- No explaining. If you have to explain the symbolism, the scene failed.\n"
    "- No dialogue unless it's one line that lands like a dropped stone.\n"
    "- No abstraction. If it could be a paragraph in a philosophy paper,"
    " rewrite it as something you can see.\n"
    "- Do not reference the conflict model directly. The scene IS the conflict"
    " model, rendered as experience."
)
# fmt: on


class MercuryError(Exception):
    """Raised when the Mercury-2 API call fails."""


def read_api_key(api_key_file: Path) -> str:
    """Read the API key from a .env-format file.

    Expects a line like: API_KEY=sk_...

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file exists but contains no API_KEY entry.
    """
    if not api_key_file.is_file():
        raise FileNotFoundError(f"API key file not found: {api_key_file}")
    env = _parse_env_file(api_key_file)
    api_key = env.get("API_KEY", "").strip()
    if not api_key:
        raise ValueError(f"No API_KEY found in {api_key_file}")
    return api_key


def build_user_message(problem: str, context_bundle: ContextBundle) -> str:
    """Format the user message for the Layer 1 API call."""
    return f"PROBLEM: {problem}\n\nCONTEXT FRAGMENTS:\n{context_bundle.as_text()}"


def build_request_body(
    problem: str,
    context_bundle: ContextBundle,
    model: str = "mercury-2",
) -> dict:
    """Build the full JSON request body for a Layer 1 chat completions call.

    Kept for backward compatibility. Equivalent to building a request with
    LAYER_1_SYSTEM_PROMPT and the formatted user message.
    """
    return _build_request_body(
        system_prompt=LAYER_1_SYSTEM_PROMPT,
        user_message=build_user_message(problem, context_bundle),
        model=model,
    )


def _build_request_body(
    system_prompt: str,
    user_message: str,
    model: str = "mercury-2",
) -> dict:
    """Build a generic JSON request body for the chat completions endpoint."""
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    }


def parse_response(response_body: bytes) -> str:
    """Parse the assistant message text from an OpenAI-format chat completions response.

    Raises:
        MercuryError: If the response cannot be parsed or contains no content.
    """
    try:
        data = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise MercuryError(f"Invalid JSON in API response: {exc}") from exc

    # Check for API-level error
    if "error" in data:
        err = data["error"]
        msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
        raise MercuryError(f"API error: {msg}")

    try:
        choices = data["choices"]
        if not choices:
            raise MercuryError("API response contains no choices")
        content = choices[0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise MercuryError(f"Unexpected API response structure: {exc}") from exc

    if not content:
        raise MercuryError("API response contains empty content")

    return content


def _call_api(
    system_prompt: str,
    user_message: str,
    api_key: str,
    base_url: str = "https://api.inceptionlabs.ai/v1",
    model: str = "mercury-2",
    timeout: int = 120,
) -> str:
    """Low-level API call: send system_prompt + user_message, return response text.

    This is the shared transport for both Layer 1 and Layer 2.

    Args:
        system_prompt: The system prompt for the call.
        user_message: The user message content.
        api_key: Bearer token for the API.
        base_url: API base URL (no trailing slash).
        model: Model identifier.
        timeout: Request timeout in seconds.

    Returns:
        The assistant's response text.

    Raises:
        MercuryError: On any API or network error.
    """
    url = f"{base_url.rstrip('/')}/chat/completions"
    body = _build_request_body(system_prompt, user_message, model=model)
    payload = json.dumps(body).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            response_body = resp.read()
    except urllib.error.HTTPError as exc:
        # Read the error body for diagnostics
        error_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise MercuryError(
            f"HTTP {exc.code} from {url}: {exc.reason}. Body: {error_body}"
        ) from exc
    except urllib.error.URLError as exc:
        raise MercuryError(f"Network error calling {url}: {exc.reason}") from exc
    except TimeoutError:
        raise MercuryError(f"Request to {url} timed out after {timeout}s") from None

    return parse_response(response_body)


def call_layer_1(
    problem: str,
    context_bundle: ContextBundle,
    api_key: str,
    base_url: str = "https://api.inceptionlabs.ai/v1",
    model: str = "mercury-2",
    timeout: int = 120,
) -> str:
    """Call Layer 1 (conflict model) and return the response text.

    Args:
        problem: The problem statement to reflect on.
        context_bundle: Assembled context from Module 1.
        api_key: Bearer token for the API.
        base_url: API base URL (no trailing slash).
        model: Model identifier.
        timeout: Request timeout in seconds.

    Returns:
        The conflict model text.

    Raises:
        MercuryError: On any API or network error.
    """
    user_message = build_user_message(problem, context_bundle)
    return _call_api(
        system_prompt=LAYER_1_SYSTEM_PROMPT,
        user_message=user_message,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
    )


def call_layer_2(
    conflict_model: str,
    api_key: str,
    base_url: str = "https://api.inceptionlabs.ai/v1",
    model: str = "mercury-2",
    timeout: int = 120,
) -> str:
    """Call Layer 2 (dream scene) and return the response text.

    Args:
        conflict_model: The Layer 1 output (conflict model text).
        api_key: Bearer token for the API.
        base_url: API base URL (no trailing slash).
        model: Model identifier.
        timeout: Request timeout in seconds.

    Returns:
        The dream scene text.

    Raises:
        MercuryError: On any API or network error.
    """
    return _call_api(
        system_prompt=LAYER_2_SYSTEM_PROMPT,
        user_message=conflict_model,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
    )


def call_mercury(
    problem: str,
    context_bundle: ContextBundle,
    api_key: str,
    base_url: str = "https://api.inceptionlabs.ai/v1",
    model: str = "mercury-2",
    timeout: int = 120,
) -> str:
    """Call the Mercury-2 chat completions endpoint and return the response text.

    Backward-compatible wrapper around call_layer_1.

    Args:
        problem: The problem statement to reflect on.
        context_bundle: Assembled context from Module 1.
        api_key: Bearer token for the API.
        base_url: API base URL (no trailing slash).
        model: Model identifier.
        timeout: Request timeout in seconds.

    Returns:
        The assistant's response text.

    Raises:
        MercuryError: On any API or network error.
    """
    return call_layer_1(
        problem=problem,
        context_bundle=context_bundle,
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout=timeout,
    )
