"""CLI entry point for the reflect command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .context import assemble_context
from .mercury import MercuryError, call_layer_1, call_layer_2, read_api_key


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the reflect CLI."""
    parser = argparse.ArgumentParser(
        prog="reflect",
        description="Reflection pipeline for autonomous agents.",
    )
    parser.add_argument(
        "--problem",
        required=True,
        help="The problem or topic to reflect on.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to .env config file (default: ~/.reflect.env).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the reflect CLI command."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)

    # Assemble context (Module 1)
    try:
        bundle = assemble_context(config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    # Read API key
    if config.api_key_file is None:
        print("Error: REFLECT_API_KEY_FILE not configured.", file=sys.stderr)
        raise SystemExit(1)
    try:
        api_key = read_api_key(config.api_key_file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    # Layer 1: Conflict model (silent — output not printed)
    try:
        conflict_model = call_layer_1(
            problem=args.problem,
            context_bundle=bundle,
            api_key=api_key,
            base_url=config.base_url,
            model=config.model,
        )
    except MercuryError as exc:
        print(f"Error (Layer 1): {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    # Layer 2: Dream scene (printed to stdout)
    try:
        dream = call_layer_2(
            conflict_model=conflict_model,
            api_key=api_key,
            base_url=config.base_url,
            model=config.model,
        )
    except MercuryError as exc:
        print(f"Error (Layer 2): {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    print(dream)


if __name__ == "__main__":
    main()
