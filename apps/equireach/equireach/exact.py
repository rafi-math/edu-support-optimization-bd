from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from equireach.constants import (
    MIP_REL_GAP,
    SOLVER_EXACT,
    WARNING_TIME_LIMIT,
)
from equireach.mathutil import count_floor_from_share, warning
from equireach.models import Client, Rule, SolveResult


def group_mask(clients: list[Client], attribute: str, level: str) -> np.ndarray:
    return np.array(
        [str(c.attributes.get(attribute)) == str(level) for c in clients],
        dtype=bool,
    )


def required_count(rule: Rule, k: int, unweighted_exactly: bool) -> int | None:
    if rule.min_count is not None:
        return int(rule.min_count)
    if unweighted_exactly and rule.min_share_pct is not None:
        return count_floor_from_share(rule.min_share_pct / 100.0, k)
    return None


def solve_mode_a_exact(
    clients: list[Client],
    v: np.ndarray,
    a: np.ndarray,
    k: int,
    coverage_mode: str,
    rules: list[Rule],
    time_limit_s: float,
) -> SolveResult:
    n = len(clients)
    selected = [False] * n
    warnings: list[dict] = []
    if n == 0 or k <= 0:
        return SolveResult(
            selected=selected,
            status="infeasible",
            solver=SOLVER_EXACT,
            proven_optimal=False,
            mip_gap=None,
            reasons=["selected_by_solver"] * n,
            warnings=warnings,
        )

    constraints = []
    ones = np.ones((1, n), dtype=float)
    lower = float(k) if coverage_mode == "exactly_K" else 0.0
    constraints.append(LinearConstraint(ones, np.array([lower]), np.array([float(k)])))

    unweighted_exactly = coverage_mode == "exactly_K" and np.allclose(a, 1.0)
    for rule in rules:
        mask = group_mask(clients, rule.attribute, rule.level).astype(float)
        count = required_count(rule, k, unweighted_exactly)
        if count is not None:
            if count > 0:
                constraints.append(
                    LinearConstraint(
                        mask.reshape(1, -1),
                        np.array([float(count)]),
                        np.array([np.inf]),
                    )
                )
        elif rule.min_share_pct:
            frac = rule.min_share_pct / 100.0
            coef = (mask - frac) * a
            constraints.append(
                LinearConstraint(coef.reshape(1, -1), np.array([0.0]), np.array([np.inf]))
            )
        if rule.max_share_pct is not None:
            cap = rule.max_share_pct / 100.0
            coef = (mask - cap) * a
            constraints.append(
                LinearConstraint(
                    coef.reshape(1, -1), np.array([-np.inf]), np.array([0.0])
                )
            )

    result = milp(
        c=-np.asarray(v, dtype=float),
        constraints=constraints,
        integrality=np.ones(n, dtype=int),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        options={"time_limit": float(time_limit_s), "mip_rel_gap": MIP_REL_GAP},
    )

    status_code = int(result.status)
    mip_gap = getattr(result, "mip_gap", None)
    if mip_gap is not None:
        mip_gap = float(mip_gap)

    if result.x is None:
        if status_code == 1:
            warnings.append(
                warning(WARNING_TIME_LIMIT, "time limit reached before a feasible solution was found")
            )
        return SolveResult(
            selected=selected,
            status="infeasible",
            solver=SOLVER_EXACT,
            proven_optimal=False,
            mip_gap=mip_gap,
            reasons=["selected_by_solver"] * n,
            warnings=warnings,
        )

    chosen = np.rint(result.x).astype(int).astype(bool)
    proven = status_code == 0
    if coverage_mode == "exactly_K" and int(chosen.sum()) != k:
        # Numerical leftover from HiGHS; do not claim a proof.
        chosen = np.zeros(n, dtype=bool)
        order = np.argsort(-np.asarray(result.x, dtype=float))
        chosen[order[:k]] = True
        proven = False

    if status_code == 1:
        warnings.append(
            warning(WARNING_TIME_LIMIT, "time limit reached; returning the best feasible solution")
        )

    solve_status = "ok" if status_code in (0, 1) else "infeasible"
    if status_code == 2:
        solve_status = "infeasible"
        chosen = np.zeros(n, dtype=bool)
        proven = False

    return SolveResult(
        selected=chosen.tolist(),
        status=solve_status,
        solver=SOLVER_EXACT,
        proven_optimal=bool(proven),
        mip_gap=0.0 if proven else mip_gap,
        reasons=["selected_by_solver"] * n,
        warnings=warnings,
    )
