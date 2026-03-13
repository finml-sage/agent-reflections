# Agent Reflections

A reflection pipeline for autonomous agents. Gathers context from session transcripts and configurable sources, assembles it into a structured bundle for downstream processing.

## Setup

```bash
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `~/.reflect.env` and customize:

```bash
cp .env.example ~/.reflect.env
```

Works with zero config (sensible defaults). Session directory defaults to `~/.claude/projects/`.

## Usage

```python
from agent_reflections import load_config, assemble_context

config = load_config()  # reads ~/.reflect.env
bundle = assemble_context(config)
print(bundle.as_text())
```

## Tests

```bash
pytest
```

## Module Structure

- `config.py` -- `.env` loading and configuration dataclasses
- `session.py` -- JSONL session parser (Claude Code format)
- `context.py` -- Context bundle assembly from session + sampled sources
