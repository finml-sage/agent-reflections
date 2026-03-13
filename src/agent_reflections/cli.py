"""CLI entry point for the reflect command."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config
from .context import assemble_context


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

    try:
        bundle = assemble_context(config)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from None

    print(f"Problem: {args.problem}\n")
    print(bundle.as_text())
    print("\n--- Layer 1/2/3 pipeline not yet wired (Module 2) ---")


if __name__ == "__main__":
    main()
