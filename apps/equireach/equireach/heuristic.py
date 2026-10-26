from __future__ import annotations

import numpy as np

from equireach.constants import SOLVER_HEURISTIC, WARNING_RULE_UNMET
from equireach.exact import group_mask, required_count
from equireach.mathutil import warning
from equireach.models import Client, Rule, SolveResult


def _share(selected: np.ndarray, mask: np.ndarray, a: np.ndarray) -> float:
    denom = float((a * selected).sum())
    if denom <= 0:
        return 0.0
    return float((a * selected * mask).sum() / denom)


def _rule_ok(
    selected: np.ndarray,
    clients: list[Client],
    rule: Rule,
    a: np.ndarray,
    k: int,
    unweighted_exactly: bool,
) -> bool:
    mask = group_mask(clients, rule.attribute, rule.level)
    count = required_count(rule, k, unweighted_exactly)
    if count is not None:
        if int((selected & mask).sum()) < count:
            return False
    elif rule.min_share_pct:
        if _share(selected, mask, a) + 1e-12 < rule.min_share_pct / 100.0:
            return False
    if rule.max_share_pct is not None:
        if _share(selected, mask, a) - 1e-12 > rule.max_share_pct / 100.0:
            return False
    return True


def _met_set(
    selected: np.ndarray,
    clients: list[Client],
    rules: list[Rule],
    a: np.ndarray,
    k: int,
    unweighted_exactly: bool,
) -> set[str]:
    return {
        rule.rule_id
        for rule in rules
        if _rule_ok(selected, clients, rule, a, k, unweighted_exactly)
    }


def solve_mode_a_heuristic(
    clients: list[Client],
    v: np.ndarray,
    a: np.ndarray,
    k: int,
    coverage_mode: str,
    rules: list[Rule],
) -> SolveResult:
    n = len(clients)
    warnings: list[dict] = []
    reasons = ["top_priority"] * n
    if n == 0 or k <= 0:
        return SolveResult(
            selected=[False] * n,
            status="infeasible",
            solver=SOLVER_HEURISTIC,
            proven_optimal=False,
            mip_gap=None,
            reasons=reasons,
            warnings=warnings,
        )

    ids = np.array([c.client_id for c in clients])
    order = np.lexsort((ids, -np.asarray(v, dtype=float)))
    selected = np.zeros(n, dtype=bool)
    take = min(k, n)
    selected[order[:take]] = True
    unweighted_exactly = coverage_mode == "exactly_K" and np.allclose(a, 1.0)

    def try_repair(rule: Rule) -> bool:
        mask = group_mask(clients, rule.attribute, rule.level)
        changed = False
        while not _rule_ok(selected, clients, rule, a, k, unweighted_exactly):
            met_before = _met_set(selected, clients, rules, a, k, unweighted_exactly)
            out_idx = np.flatnonzero(selected & ~mask)
            in_idx = np.flatnonzero(~selected & mask)
            if len(out_idx) == 0 or len(in_idx) == 0:
                return changed
            out_idx = out_idx[np.lexsort((ids[out_idx], v[out_idx]))]
            in_idx = in_idx[np.lexsort((ids[in_idx], -v[in_idx]))]
            swapped = False
            for i_out in out_idx:
                for i_in in in_idx:
                    trial = selected.copy()
                    trial[i_out] = False
                    trial[i_in] = True
                    still_ok = True
                    for other in rules:
                        if other.rule_id == rule.rule_id:
                            continue
                        if other.rule_id in met_before and not _rule_ok(
                            trial, clients, other, a, k, unweighted_exactly
                        ):
                            still_ok = False
                            break
                    if not still_ok:
                        continue
                    improved = not _rule_ok(
                        selected, clients, rule, a, k, unweighted_exactly
                    ) and (
                        _share(trial, mask, a) >= _share(selected, mask, a) - 1e-15
                        or _rule_ok(trial, clients, rule, a, k, unweighted_exactly)
                    )
                    if unweighted_exactly:
                        improved = int((trial & mask).sum()) >= int((selected & mask).sum())
                    if not improved:
                        continue
                    selected[i_out] = False
                    selected[i_in] = True
                    reasons[int(i_in)] = f"swapped_in_for_rule:{rule.rule_id}"
                    swapped = True
                    changed = True
                    break
                if swapped:
                    break
            if not swapped:
                return changed
        return changed

    for rule in rules:
        try_repair(rule)
    for rule in rules:
        try_repair(rule)
    moved = True
    passes = 0
    while moved and passes < n:
        moved = False
        passes += 1
        for rule in rules:
            if try_repair(rule):
                moved = True

    unmet = [
        rule
        for rule in rules
        if not _rule_ok(selected, clients, rule, a, k, unweighted_exactly)
    ]
    status = "ok"
    unmet_id = None
    if unmet:
        status = "partial"
        unmet_id = unmet[0].rule_id
        mask = group_mask(clients, unmet[0].attribute, unmet[0].level)
        achieved = 100.0 * _share(selected, mask, a)
        need = unmet[0].min_share_pct
        if need is None and unmet[0].min_count is not None and take:
            need = 100.0 * unmet[0].min_count / take
        warnings.append(
            warning(
                WARNING_RULE_UNMET,
                f"rule {unmet_id} unmet (achieved {achieved:.4g}% against {need}%)",
            )
        )

    return SolveResult(
        selected=selected.tolist(),
        status=status,
        solver=SOLVER_HEURISTIC,
        proven_optimal=False,
        mip_gap=None,
        reasons=reasons,
        warnings=warnings,
        unmet_rule_id=unmet_id,
    )
