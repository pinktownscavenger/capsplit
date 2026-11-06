"""RQ1 on a 4x4 grid: do planted cases land in their predicted 2x2 cell?

Run: python experiments/phase1_rq1.py

Two bases. "easy": sharpening helps (the modal behaviour is correct).
"decoy": an extra bonus on bumping the wall at the start makes the modal
behaviour wrong, so sharpening hurts and only a genuinely different pi1 can win.
"""
import numpy as np

from capsplit import excess_score_families, label_2x2, sequence_temper, split
from capsplit.core import success_prob
from capsplit.worlds import (
    GridWorld,
    correct_paths,
    enumerate_paths,
    per_step_family,
    policy_from_routes,
)

WORLD = GridWorld(n=4, start=(0, 0), goal=(3, 3), H=6, walls=frozenset({(1, 1)}))
R_COMMON = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3)]
R_MID = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3)]
R_RARE = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2), (3, 3)]
DECOY = [(0, 0), (0, 0)]  # bump the top wall: stay at the start
ROUTES = [R_COMMON, R_MID, R_RARE, DECOY]
BETAS = np.exp(np.linspace(np.log(0.25), np.log(8), 25))


def lg(common, mid, rare, decoy=0.0):
    return policy_from_routes(WORLD, ROUTES, [common, mid, rare, decoy])


def report(name, lg0, lg1, n, q1_override=None):
    q0 = enumerate_paths(WORLD, lg0)
    q1 = q1_override(q0) if q1_override else enumerate_paths(WORLD, lg1)
    correct = correct_paths(WORLD, q0) | correct_paths(WORLD, q1)
    s = split(q0, q1, correct, n)
    fams = [lambda b: sequence_temper(q0, b), per_step_family(WORLD, lg0)]
    e = excess_score_families(q1, correct, n, fams, BETAS)
    print(f"{name:32s} N={n:<3d} J0={success_prob(q0, correct):.3f} J1={success_prob(q1, correct):.3f} "
          f"d_rel={s['delta_rel']:+.4f} d_cap={s['delta_cap']:+.4f} E_N={e:+.4f} "
          f"-> {label_2x2(s['delta_cap'], e)}")


def show_base(lg0, n):
    q0 = enumerate_paths(WORLD, lg0)
    print(f"  base prob of routes (1/N={1/n}): "
          + ", ".join(f"{q0.get(tuple(r), 0.0):.4f}" for r in (R_COMMON, R_MID, R_RARE)))


print("EASY base (sharpening helps)")
easy = lg(2.5, 2.0, 0.0)
show_base(easy, 20)
report("(a) per-step sharpening b=3", easy, None, 20, lambda q0: enumerate_paths(WORLD, easy, 3.0))
report("(a') sequence sharpening b=3", easy, None, 20, lambda q0: sequence_temper(q0, 3.0))

print("DECOY base (sharpening hurts)")
for n in (4, 10, 20):
    dec = lg(2.5, 2.0, 0.0, decoy=3.5)
    show_base(dec, n)
    report("(a) per-step sharpening b=3", dec, None, n, lambda q0: enumerate_paths(WORLD, dec, 3.0))
    report("(b) reweight reachable routes", dec, lg(0.5, 4.0, 0.0, decoy=0.0), n)
    report("(c) boost rare route", dec, lg(2.5, 2.0, 3.0, decoy=0.0), n)
