"""Command line: `python -m capsplit.harness {run,report,list}`."""
from __future__ import annotations

import argparse
import importlib
import sys
from typing import Sequence


def _run(args: argparse.Namespace) -> int:
    from .config import ConfigError
    from .runner import run_experiment

    try:
        out = run_experiment(args.config, out_root=args.out, workers=args.workers, force=args.force)
    except (ConfigError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(out)
    return 0


def _report(args: argparse.Namespace) -> int:
    try:
        build_report = importlib.import_module("capsplit.harness.report").build_report
    except ModuleNotFoundError as error:
        if error.name in ("capsplit.harness.report",):
            print("error: the report generator (capsplit.harness.report, task T6) is not available yet", file=sys.stderr)
            return 2
        raise
    result = build_report(args.results_dir)
    if result is not None:
        print(result)
    return 0


def _list(args: argparse.Namespace) -> int:
    from .registry import BACKENDS, BASES, POLICIES, TASKS, TRAINERS

    for title, registry in (("tasks", TASKS), ("bases", BASES), ("policies", POLICIES), ("trainers", TRAINERS), ("backends", BACKENDS)):
        print(f"{title}: {', '.join(registry.names()) or '(none)'}")
    return 0


def _verdicts(args: argparse.Namespace) -> int:
    from .verdicts import build_verdicts

    try:
        print(build_verdicts(args.results_dir))
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m capsplit.harness", description="Capability vs reliability experiment harness")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an experiment config")
    run.add_argument("config", help="path to a JSON experiment config")
    run.add_argument("--out", default="results", help="results root (default: results)")
    run.add_argument("--workers", type=int, default=1, help="parallel processes (default: 1)")
    run.add_argument("--force", action="store_true", help="re-run runs whose metrics already exist and rebuild bases")
    run.set_defaults(func=_run)

    report = sub.add_parser("report", help="build the report for a results folder")
    report.add_argument("results_dir", help="<out>/<experiment name>")
    report.set_defaults(func=_report)

    verdicts = sub.add_parser("verdicts", help="apply the pre-registered analysis (docs/preregistration.md)")
    verdicts.add_argument("results_dir", help="<out>/<experiment name>")
    verdicts.set_defaults(func=_verdicts)

    lst = sub.add_parser("list", help="list registered plug-ins")
    lst.set_defaults(func=_list)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))
