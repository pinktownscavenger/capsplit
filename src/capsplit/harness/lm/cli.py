"""Command line for the staged LM experiment: one command per stage.

Each stage prints the files it wrote and the command to run next. Errors that the user can fix
(bad config, low memory, missing earlier stage) exit with code 2 and a message saying what to do.

Stages: doctor, fetch, speedtest, setup (fetch + speedtest), data, base, train, measure, report.
Long stages should run under `caffeinate -i` (DECISIONS L13) so the machine does not sleep.
"""
from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .config import LMConfig, LMConfigError, load_config
from .device import DeviceError, MemoryGuardError
from .report import ReportError
from .storage import append_jsonl

StageFn = Callable[[LMConfig, Path, argparse.Namespace], list[Path]]


def _setup_stages() -> dict[str, tuple[StageFn, str]]:
    from . import setup

    return {
        "doctor": (lambda c, o, a: [setup.doctor(c, o)], "fetch"),
        "fetch": (lambda c, o, a: [setup.fetch_stage(c, o)], "speedtest"),
        "speedtest": (lambda c, o, a: [setup.speedtest(c, o, use_fallback=a.fallback)], "data"),
        "setup": (lambda c, o, a: [setup.fetch_stage(c, o), setup.speedtest(c, o, use_fallback=a.fallback)], "data"),
    }


def _factory(config: LMConfig, args: argparse.Namespace) -> tuple[object, object]:
    from . import stages as st

    device = st.resolve_device(config)
    return device, st.default_policy_factory(config, device, use_fallback=args.fallback)


def _experiment_stages() -> dict[str, tuple[StageFn, str]]:
    from . import stages as st

    def data(c: LMConfig, o: Path, a: argparse.Namespace) -> list[Path]:
        return st.run_data(c, o, device="cpu")

    def base(c: LMConfig, o: Path, a: argparse.Namespace) -> list[Path]:
        device, factory = _factory(c, a)
        return st.run_base(c, o, device=device, factory=factory, use_fallback=a.fallback)  # type: ignore[arg-type]

    def train(c: LMConfig, o: Path, a: argparse.Namespace) -> list[Path]:
        device, factory = _factory(c, a)
        return st.run_train(c, o, device=device, factory=factory, use_fallback=a.fallback)  # type: ignore[arg-type]

    def measure(c: LMConfig, o: Path, a: argparse.Namespace) -> list[Path]:
        device, factory = _factory(c, a)
        return st.run_measure(c, o, device=device, factory=factory,  # type: ignore[arg-type]
                              all_checkpoints=a.all_checkpoints, use_fallback=a.fallback)

    def report(c: LMConfig, o: Path, a: argparse.Namespace) -> list[Path]:
        return st.run_report(c, o)

    def verdicts(c: LMConfig, o: Path, a: argparse.Namespace) -> list[Path]:
        from .verdicts import build_verdicts

        return [build_verdicts(o, Path(a.prereg))]

    return {"data": (data, "base"), "base": (base, "train"), "train": (train, "measure"),
            "measure": (measure, "report"), "report": (report, "verdicts"), "verdicts": (verdicts, "")}


STAGE_NAMES = ("doctor", "fetch", "speedtest", "setup", "data", "base", "train", "measure", "report", "verdicts")


def stages() -> dict[str, tuple[StageFn, str]]:
    """Stage name -> (function(config, out_dir, args) -> written paths, next stage)."""
    return {**_setup_stages(), **_experiment_stages()}


def record_timing(out: Path, stage: str, seconds: float) -> Path:
    """Append this invocation's wall time to `<out>/timing/<stage>.jsonl`.

    One line per successful invocation, so a stage that was interrupted and resumed has several
    lines; the report sums them. Durations are measured here, never typed into documents.
    """
    path = out / "timing" / f"{stage}.jsonl"
    append_jsonl(path, [{"stage": stage, "wall_seconds": round(seconds, 3)}])
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m capsplit.harness.lm", description="Staged LM capability/reliability experiment")
    parser.add_argument("stage", choices=STAGE_NAMES, help="stage to run")
    parser.add_argument("config", help="path to an LM experiment config (JSON)")
    parser.add_argument("--out", default="results", help="results root (default: results)")
    parser.add_argument("--fallback", action="store_true", help="use model.fallback_id instead of model.id")
    parser.add_argument("--prereg", default="configs/lm_prereg.json",
                        help="verdict thresholds for the verdicts stage (default: configs/lm_prereg.json)")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="measure: also measure intermediate checkpoints (default: final only)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .stages import StageError

    try:
        config = load_config(args.config)
        out = Path(args.out) / config.name
        out.mkdir(parents=True, exist_ok=True)
        run, next_stage = stages()[args.stage]
        started = time.monotonic()
        written = run(config, out, args)
        written.append(record_timing(out, args.stage, time.monotonic() - started))
    except (LMConfigError, MemoryGuardError, DeviceError, StageError, ReportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except OSError as error:  # model files not in the cache, network down, unreadable file
        if args.stage in ("fetch", "setup"):
            hint = "Downloading the model failed: check your network connection, then re-run this stage."
        else:
            hint = ("If the model files are missing: run the fetch stage first "
                    f"(python -m capsplit.harness.lm fetch {args.config} --out {args.out}); "
                    "if fetch fails, check your network connection.")
        print(f"error: {type(error).__name__}: {error}\n{hint}", file=sys.stderr)
        return 2
    except (KeyError, ValueError) as error:  # unknown trainer / generator, bad trainer params
        print(f"error: {error}", file=sys.stderr)
        return 2
    for path in written:
        print(f"wrote {path}")
    if next_stage:
        flags = " --fallback" if args.fallback else ""
        print(f"next: python -m capsplit.harness.lm {next_stage} {args.config} --out {args.out}{flags}")
    else:
        print("next: done (open the report)")
    return 0
