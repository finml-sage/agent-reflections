# Agent Reflections

A three-layer reflection pipeline for autonomous agents. Gathers context from session transcripts and configurable sources, builds a conflict model, renders a dream scene, and returns the agent to first person as an observer.

The core mechanism is the perspective toggle: first person to third person to first person. Each layer is a separate stateless API call -- no conversation history carries between them. This prevents cross-layer leakage.

Powered by Mercury-2 from InceptionLabs (diffusion-based LLM). Zero external dependencies -- stdlib only (`urllib.request` + `json`).

## The Three Layers

### Layer 1 -- The Conflict Model (silent, not printed)

First-person inner voice. Receives the problem plus randomly sampled context fragments (memories, rules, session transcripts). Builds a multi-dimensional tension map:

- Surface tensions
- Hidden patterns
- Conflicting goals
- What's being avoided
- Strange connections
- Emotional undercurrent
- Meta-pattern

Output feeds Layer 2 but is never shown to the user.

### Layer 2 -- The Dream (third person)

Scene-builder transforms the conflict model into a 20-line vivid scene. Third person ("they"). One continuous scene, not a montage. Tensions become physical -- conflicting goals become two paths, avoidance becomes a door they keep not opening.

Two variants, randomly selected each run:

- **Variant A (SEARCH)**: The character mid-struggle -- searching, testing, getting closer or further without knowing which. Contains a shift, a small noticing.
- **Variant B (AFTER)**: The world with the problem resolved. Not triumph -- aftermath. Traces of what was. What was lost. A quiet residue of transformation.

### Layer 3 -- The Observer (first person return)

The agent re-enters the dream as protagonist. Detective framing -- every object is a clue, every spatial relationship encodes meaning. Struggles for meaning, pushes through to either discover the insight or end on a cliffhanger. 30-40 lines. The perspective toggle closes into a spiral, not a circle.

## Tuning the Dream

You can steer the aesthetic of the dream by embedding style instructions directly in your problem text. Append a `STYLE INSTRUCTION:` line and describe the vibe you want:

```bash
reflect --problem "I can't stop procrastinating. STYLE INSTRUCTION: Render like a noir detective film -- rain-slicked streets, neon, shadows."
reflect --problem "I'm mass-producing content instead of creating something real. STYLE INSTRUCTION: Render like an Eminem music video -- raw, gritty, Detroit concrete."
reflect --problem "I keep avoiding the hard conversation. STYLE INSTRUCTION: Render like a Tarkovsky film -- slow, water, decay, light through ruined windows."
```

Style instructions act as seasoning, not format overrides. They influence the aesthetic -- setting, texture, mood -- but the pipeline structure stays intact. The dream still follows its form rules (20 lines, third person, one scene). The observer still returns in first person with detective framing. You get the *feel* you asked for without breaking the reflection mechanics.

This means you can run the same problem multiple times with different style instructions and get genuinely different dream experiences that illuminate different facets of the same tension.

## Install

```bash
pipx install .
```

For development:

```bash
pipx install -e .
```

No venv activation needed -- pipx manages the environment.

## Configuration

Copy `.env.example` to `~/.reflect.env` and customize:

```bash
cp .env.example ~/.reflect.env
```

Key variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `REFLECT_SESSION_DIR` | Claude Code session transcript directory | `~/.claude/projects/` |
| `REFLECT_SESSION_DEPTH` | How many recent exchanges to include | -- |
| `REFLECT_MEMORY_PATH` | Directory of `.md` memory files to randomly sample | -- |
| `REFLECT_MEMORY_COUNT` | How many memory files to sample per run | -- |
| `REFLECT_RULES_PATH` | Operational rules directory | -- |
| `REFLECT_RULES_COUNT` | How many rules to sample | -- |
| `REFLECT_EXTRA_SOURCE_N` / `REFLECT_EXTRA_COUNT_N` | Extensible extra sources (add as many as needed) | -- |
| `REFLECT_MODEL` | Model ID | `mercury-2` |
| `REFLECT_BASE_URL` | API endpoint | `https://api.inceptionlabs.ai/v1` |
| `REFLECT_API_KEY_FILE` | Path to `.env` file containing `API_KEY=sk_...` | -- |

The random sampling is intentional -- different context surfaces each run, creating serendipitous connections between memories and the problem.

## Usage

```bash
reflect --problem "why do I keep solving the same problems"
reflect --problem "I ship fast but nothing sticks" --config ~/.reflect.env
```

Output format:

```
=== YOUR PROBLEM ===

<the problem you stated>

=== THE DREAM ===

<20-line third-person scene>

=== THE OBSERVER ===

<30-40 line first-person return>
```

## Python API

```python
from agent_reflections import load_config, assemble_context
from agent_reflections.mercury import read_api_key, call_layer_1, call_layer_2, call_layer_3

config = load_config()
bundle = assemble_context(config)
api_key = read_api_key(config.api_key_file)

conflict = call_layer_1(problem="...", context_bundle=bundle, api_key=api_key)
dream = call_layer_2(conflict_model=conflict, api_key=api_key)
observer = call_layer_3(problem="...", dream=dream, api_key=api_key)
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

128+ tests. All layers tested independently with mocked API responses.

## Module Structure

- `cli.py` -- CLI entry point (`reflect` command), pipeline orchestration
- `config.py` -- `.env` loading and configuration dataclasses
- `session.py` -- JSONL session parser (Claude Code format)
- `context.py` -- Context bundle assembly from session + sampled sources
- `mercury.py` -- Mercury-2 API client, all three layer prompts, dual dream variants

## Requirements

- Python 3.10+
- Mercury-2 API key from InceptionLabs
- No external dependencies (stdlib only)
