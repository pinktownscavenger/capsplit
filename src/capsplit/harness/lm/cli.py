"""Command line for the staged LM experiment: one command per stage.

Each stage prints the files it wrote and the command to run next. Errors that the user can fix
(bad config, low memory, missing earlier stage) exit with code 2 and a message saying what to do.

Stages: doctor, fetch, speedtest, setup (fetch + speedtest), data, base, train, measure, report,
verdicts, discovery (descriptive, DECISIONS L22), the optional post-hoc `sensitivity` (L21), and the
exploratory function-space layer `kl` / `kl-report` (L23-L31), which read `<out>/<name>/` and write
only `<out>/<name>_kl/` (their timing lines included).
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
from .kl import KLConfigError
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
        written = st.run_report(c, o)
        rep = o / "report"
        extra = [rep / "summary.json", *sorted((rep / "tables").glob("*")), *sorted((rep / "figures").glob("*"))]
        return written + [p for p in extra if p.is_file() and p not in written]

    def verdicts(c: LMConfig, o: Path, a: argparse.Namespace) -> list[Path]:
        from .verdicts import build_verdicts

        path = build_verdicts(o, Path(a.prereg))
        return [path, path.with_suffix(".md")]

    def sensitivity(c: LMConfig, o: Path, a: argparse.Namespace) -> list[Path]:
        from .sensitivity import build_sensitivity

        path = build_sensitivity(o, Path(a.prereg))
        return [path, path.with_suffix(".md")]

    def discovery(c: LMConfig, o: Path, a: argparse.Namespace) -> list[Path]:
        from .discovery import build_discovery

        path = build_discovery(o)
        return [path, path.with_suffix(".md")]

    def kl_stage(c: LMConfig, o: Path, a: argparse.Namespace) -> list[Path]:
        from .kl import load_kl_config, run_kl

        device, factory = _factory(c, a)
        return run_kl(c, load_kl_config(a.kl_config), o, kl_dir(o), device=device, factory=factory,  # type: ignore[arg-type]
                      use_fallback=a.fallback)

    def kl_report(c: LMConfig, o: Path, a: argparse.Namespace) -> list[Path]:
        from .kl import build_kl_report, load_kl_config

        path = build_kl_report(o, kl_dir(o), load_kl_config(a.kl_config))
        rep = path.parent
        return [path, rep / "report.md", *sorted((rep / "tables").glob("*")), *sorted((rep / "figures").glob("*"))]

    return {"data": (data, "base"), "base": (base, "train"), "train": (train, "measure"),
            "measure": (measure, "report"), "report": (report, "verdicts"), "verdicts": (verdicts, "discovery"),
            "discovery": (discovery, "sensitivity"), "sensitivity": (sensitivity, ""),
            "kl": (kl_stage, "kl-report"), "kl-report": (kl_report, "")}


STAGE_NAMES = ("doctor", "fetch", "speedtest", "setup", "data", "base", "train", "measure", "report", "verdicts",
               "discovery", "sensitivity", "kl", "kl-report")
KL_STAGES = ("kl", "kl-report")


def kl_dir(out: Path) -> Path:
    """`<out>/<name>_kl`: the function-space layer's own folder next to the run it reads (DECISIONS L24)."""
    return out.parent / f"{out.name}_kl"


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
                        help="verdict thresholds for the verdicts and sensitivity stages (default: configs/lm_prereg.json)")
    parser.add_argument("--kl-config", default="configs/lm_kl.json",
                        help="kl and kl-report stages: the KL layer's config (default: configs/lm_kl.json)")
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
        timing_out = kl_dir(out) if args.stage in KL_STAGES else out
        written.append(record_timing(timing_out, args.stage, time.monotonic() - started))
    except (LMConfigError, KLConfigError, MemoryGuardError, DeviceError, StageError, ReportError) as error:
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
