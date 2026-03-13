# Agent Reflections

A reflection pipeline for autonomous agents. Gathers context from session transcripts and configurable sources, assembles it into a structured bundle for downstream processing.

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

Works with zero config (sensible defaults). Session directory defaults to `~/.claude/projects/`.

## Usage

```bash
reflect --problem "how can I order a hamburger"
reflect --problem "why is my pipeline slow" --config /path/to/.env
```

### Python API

```python
from agent_reflections import load_config, assemble_context

config = load_config()  # reads ~/.reflect.env
bundle = assemble_context(config)
print(bundle.as_text())
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Module Structure

- `cli.py` -- CLI entry point (`reflect` command)
- `config.py` -- `.env` loading and configuration dataclasses
- `session.py` -- JSONL session parser (Claude Code format)
- `context.py` -- Context bundle assembly from session + sampled sources
