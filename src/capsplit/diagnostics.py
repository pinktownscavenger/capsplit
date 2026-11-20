"""The practical pass@k rule (D10) and how often it agrees with the 2x2 label."""
import numpy as np

_SHARPENING_CELLS = {"pure_sharpening", "compounded_sharpening"}


def passk_curve(js, ks) -> np.ndarray:
    """Mean over prompts of 1 - (1 - J_i)^k, one value per k."""
    js = np.asarray(js, dtype=float)
    ks = np.asarray(ks)
    return np.mean(1.0 - (1.0 - js[:, None]) ** ks, axis=0)


def passk_rule(js0, js1, ks, k_large: int, tol: float = 1e-9) -> str:
    """Classify a run from its curves. Differences within tol are ties, since curves saturate at large k."""
    ks = np.asarray(ks)
    difference = passk_curve(js1, ks) - passk_curve(js0, ks)
    below = difference < -tol
    if below[ks >= k_large].any():
        return "sharpening"
    if not below.any() and (difference > tol).any():
        return "capability"
    return "undetermined"


def _rates(rules: list[str], cells: list[str]) -> dict:
    runs = len(rules)
    primary = sum(
        (rule == "capability" and cell == "capability") or (rule == "sharpening" and cell != "capability")
        for rule, cell in zip(rules, cells)
    )
    secondary = sum(
        (rule == "capability" and cell == "capability") or (rule == "sharpening" and cell in _SHARPENING_CELLS)
        for rule, cell in zip(rules, cells)
    )
    return {
        "runs": runs,
        "primary": primary / runs,
        "secondary": secondary / runs,
        "undetermined_rate": rules.count("undetermined") / runs,
    }


def score_rule(rule_outputs: list[str], cells: list[str]) -> dict:
    """Agreement between rule outputs and 2x2 cells, overall and per cell."""
    if len(rule_outputs) != len(cells):
        raise ValueError(f"got {len(rule_outputs)} rule outputs for {len(cells)} cells")
    overall = _rates(rule_outputs, cells)
    per_cell = {}
    for cell in sorted(set(cells)):
        rules_in_cell = [rule for rule, other in zip(rule_outputs, cells) if other == cell]
        per_cell[cell] = _rates(rules_in_cell, [cell] * len(rules_in_cell))
    return {
        "primary": overall["primary"],
        "secondary": overall["secondary"],
        "undetermined_rate": overall["undetermined_rate"],
        "per_cell": per_cell,
    }


def srr_ndr(q0, q1, eps: float) -> dict:
    """SRR and NDR (The Invisible Leash, Def. 3.3) for one prompt, from exact solution probabilities.

    UNVERIFIED: follows docs/notes/invisible-leash-scores.md, which a person has not yet checked
    against the paper. Counts correct solutions, not mass, with the note's strict `> eps`:
    P = kept (q0 > eps and q1 > eps), S = shrunk (q0 > eps, q1 <= eps), E = expanded
    (q0 <= eps, q1 > eps). SRR = P / (P + S) and NDR = E / (P + E), None when the denominator is 0.
    """
    q0, q1 = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    in0, in1 = q0 > eps, q1 > eps
    p, s, e = int((in0 & in1).sum()), int((in0 & ~in1).sum()), int((~in0 & in1).sum())
    return {"P": p, "S": s, "E": e, "srr": p / (p + s) if p + s else None, "ndr": e / (p + e) if p + e else None}
