"""Command line for the staged LM experiment: one command per stage.

Each stage prints the files it wrote and the command to run next. Errors that the user can fix
(bad config, low memory, missing earlier stage) exit with code 2 and a message saying what to do.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from .config import LMConfig, LMConfigError, load_config
from .device import MemoryGuardError


def _setup_stages() -> dict[str, tuple[Callable[[LMConfig, Path, argparse.Namespace], list[Path]], str]]:
    from . import setup

    return {
        "doctor": (lambda c, o, a: [setup.doctor(c, o)], "fetch"),
        "fetch": (lambda c, o, a: [setup.fetch_stage(c, o)], "speedtest"),
        "speedtest": (lambda c, o, a: [setup.speedtest(c, o, use_fallback=a.fallback)], "data"),
        "setup": (lambda c, o, a: [setup.fetch_stage(c, o), setup.speedtest(c, o, use_fallback=a.fallback)], "data"),
    }


def stages() -> dict:
    """Stage name -> (function(config, out_dir, args) -> written paths, next stage)."""
    return _setup_stages()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m capsplit.harness.lm", description="Staged LM capability/reliability experiment")
    parser.add_argument("stage", choices=sorted(stages()), help="stage to run")
    parser.add_argument("config", help="path to an LM experiment config (JSON)")
    parser.add_argument("--out", default="results", help="results root (default: results)")
    parser.add_argument("--fallback", action="store_true", help="use model.fallback_id instead of model.id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        out = Path(args.out) / config.name
        out.mkdir(parents=True, exist_ok=True)
        run, next_stage = stages()[args.stage]
        written = run(config, out, args)
    except (LMConfigError, MemoryGuardError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for path in written:
        print(f"wrote {path}")
    if next_stage:
        print(f"next: python -m capsplit.harness.lm {next_stage} {args.config} --out {args.out}")
    return 0
