"""Run every pilot experiment. Results go under results/, which is git-ignored.

Run: python experiments/run_pilot.py [--quick]
"""
import argparse
import time

import pilot_estimator
import pilot_rq1
import pilot_rq2
import pilot_rq4
import pilot_seq
import pilot_smoke


def experiments(quick: bool) -> list:
    return [
        ("pilot_smoke", pilot_smoke.main),
        ("pilot_rq1", lambda: pilot_rq1.main(quick)),
        ("pilot_rq2", lambda: pilot_rq2.main(quick)),
        ("pilot_estimator", pilot_estimator.main),
        ("pilot_seq", pilot_seq.main),
        ("pilot_rq4", lambda: pilot_rq4.main(quick)),
    ]


def main(quick: bool = False) -> None:
    start = time.perf_counter()
    for name, run in experiments(quick):
        print(f"\n{'=' * 20} {name} {'=' * 20}", flush=True)
        began = time.perf_counter()
        run()
        print(f"[{name}: {time.perf_counter() - began:.0f} s]", flush=True)
    print(f"\nAll experiments finished in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="fewer trials, for a check under 10 minutes")
    main(parser.parse_args().quick)
