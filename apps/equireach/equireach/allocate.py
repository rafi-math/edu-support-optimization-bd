from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np

from equireach.constants import (
    ENGINE_VERSION,
    EXACT_CLIENT_LIMIT,
    PREVIEW_LIMIT,
    SOLVER_EXACT,
    SOLVER_HEURISTIC,
    TAGLINE,
    TOOL_NAME,
    WARNING_AT_MOST_K,
    WARNING_CLAMPED,
    WARNING_EXACT_DISABLED,
    WARNING_RULE_UNMET,
)
from equireach.exact import group_mask, required_count, solve_mode_a_exact
from equireach.heuristic import solve_mode_a_heuristic
from equireach.mathutil import resolve_coverage_k, warning
from equireach.models import Client, Params, Rule, Scenario, SolveResult
from equireach.validation import exclusive_share_sum_impossible, parse_scenario


def _weights(clients: list[Client], params: Params) -> tuple[np.ndarray, np.ndarray]:
    priority = np.array([c.priority_score for c in clients], dtype=float)
    weight = np.array([c.weight for c in clients], dtype=float)
    if params.objective_weighting == "weighted":
        return priority * weight, weight
    return priority, np.ones(len(clients), dtype=float)


def _equity_compliance(
    selected: np.ndarray,
    clients: list[Client],
    rules: list[Rule],
    a: np.ndarray,
    k: int,
    unweighted_exactly: bool,
) -> list[dict]:
    rows = []
    denom = float((a * selected).sum())
    for rule in rules:
        mask = group_mask(clients, rule.attribute, rule.level)
        achieved = 0.0 if denom <= 0 else 100.0 * float((a * selected * mask).sum() / denom)
        min_pct = rule.min_share_pct
        if min_pct is None and rule.min_count is not None and k:
            min_pct = 100.0 * rule.min_count / k
        met = True
        count = required_count(rule, k, unweighted_exactly)
        if count is not None:
            met = int((selected & mask).sum()) >= count
        elif rule.min_share_pct:
            met = achieved + 1e-9 >= rule.min_share_pct
        if rule.max_share_pct is not None and achieved - 1e-9 > rule.max_share_pct:
            met = False
        rows.append(
            {
                "attribute": rule.attribute,
                "level": rule.level,
                "min_pct": float(min_pct if min_pct is not None else 0.0),
                "achieved_pct": achieved,
                "met": bool(met),
            }
        )
    return rows


def _need_capture_pct(clients: list[Client], selected: np.ndarray) -> Optional[float]:
    flags = [c.need_flag for c in clients]
    if all(flag is None for flag in flags):
        return None
    need = np.array([0 if f is None else int(f) for f in flags], dtype=float)
    total = float(need.sum())
    if total <= 0:
        return 0.0
    return 100.0 * float((need * selected).sum() / total)


def _assignment_row(client: Client, reason: str) -> dict[str, Any]:
    row = client.to_record()
    row.update(
        {
            "assigned_center_id": None,
            "assigned_center_label": None,
            "distance_km": None,
            "selection_reason": reason,
        }
    )
    return row


def _numeric_summary_delta(current: dict, baseline: dict) -> dict:
    delta = {}
    for key, value in current.items():
        other = baseline.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if isinstance(other, (int, float)) and not isinstance(other, bool):
                delta[key] = value - other
        elif key == "need_capture_pct":
            if value is None and other is None:
                delta[key] = 0.0
            elif isinstance(value, (int, float)) and isinstance(other, (int, float)):
                delta[key] = value - other
    return delta


def _build_result(
    scenario: Scenario,
    solve: SolveResult,
    k: int,
    eligible_count: int,
    extra_warnings: list[dict],
) -> dict[str, Any]:
    clients = scenario.clients
    n = len(clients)
    selected = np.array(solve.selected, dtype=bool)
    v, a = _weights(clients, scenario.params)
    unweighted_exactly = (
        scenario.params.coverage_mode == "exactly_K" and np.allclose(a, 1.0)
    )
    served_idx = np.flatnonzero(selected)
    served_count = int(selected.sum())
    captured = float(v[selected].sum()) if n else 0.0
    mean_priority = (
        float(np.mean([clients[i].priority_score for i in served_idx]))
        if served_count
        else 0.0
    )
    shortfall = max(0, k - served_count)
    coverage_pct = 0.0 if eligible_count == 0 else 100.0 * served_count / eligible_count
    equity = _equity_compliance(
        selected, clients, scenario.rules, a, k, unweighted_exactly
    )
    status = solve.status
    warnings = list(extra_warnings) + list(solve.warnings)
    if status == "ok" and any(not row["met"] for row in equity):
        status = "partial" if solve.solver == SOLVER_HEURISTIC else "infeasible"
        for row, rule in zip(equity, scenario.rules):
            if not row["met"]:
                warnings.append(
                    warning(
                        WARNING_RULE_UNMET,
                        f"rule {rule.rule_id} unmet (achieved {row['achieved_pct']:.4g}% against {row['min_pct']}%)",
                    )
                )
                break

    assignments = [
        _assignment_row(clients[i], solve.reasons[i] if i < len(solve.reasons) else "selected_by_solver")
        for i in served_idx
    ]
    assignments.sort(key=lambda row: (-row["priority_score"], row["client_id"]))
    unserved = [
        {"client_id": clients[i].client_id, "state": "eligible_unserved"}
        for i in range(n)
        if not selected[i]
    ]
    preview = assignments[:PREVIEW_LIMIT]
    summary = {
        "served_count": served_count,
        "coverage_target": k,
        "coverage_met": shortfall == 0 and status != "infeasible",
        "shortfall": shortfall if status != "infeasible" else k,
        "eligible_count": eligible_count,
        "ineligible_count": 0,
        "coverage_pct": coverage_pct,
        "total_priority_captured": captured if status != "infeasible" else 0.0,
        "mean_priority_served": mean_priority if status != "infeasible" else 0.0,
        "need_capture_pct": _need_capture_pct(clients, selected) if status != "infeasible" else _need_capture_pct(clients, np.zeros(n, dtype=bool)),
        "equity_compliance": equity,
        "cost_total": None,
        "budget_used_pct": None,
        "centers_opened": 0,
        "center_ids": [],
        "avg_travel_km": None,
        "max_travel_km_used": None,
        "p90_travel_km": None,
        "solver": solve.solver,
        "proven_optimal": bool(solve.proven_optimal and status == "ok"),
        "mip_gap": solve.mip_gap,
        "status": status,
    }
    if status == "infeasible":
        assignments = []
        preview = []
        unserved = [{"client_id": c.client_id, "state": "eligible_unserved"} for c in clients]
        summary["served_count"] = 0
        summary["coverage_met"] = False
        summary["total_priority_captured"] = 0.0
        summary["mean_priority_served"] = 0.0
        summary["coverage_pct"] = 0.0

    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "status": status,
        "solver": solve.solver,
        "proven_optimal": summary["proven_optimal"],
        "mip_gap": solve.mip_gap,
        "engine_version": ENGINE_VERSION,
        "provenance": {
            "tool": TOOL_NAME,
            "tagline": TAGLINE,
            "source": scenario.params.source,
            "seed": scenario.params.seed,
            "mode": "selection",
            "created_utc": created,
        },
        "inputs_echo": {
            "params": scenario.params.to_record(resolved_k=k),
            "rules": [rule.to_record() for rule in scenario.rules],
        },
        "summary": summary,
        "centers": [],
        "assignments": assignments,
        "unserved": unserved,
        "preview": preview,
        "baseline_comparison": None,
        "warnings": warnings,
    }


def _solve(scenario: Scenario, k: int, rules: list[Rule]) -> tuple[SolveResult, list[dict]]:
    clients = scenario.clients
    params = scenario.params
    extra = []
    v, a = _weights(clients, params)
    use_exact = bool(params.exact)
    if use_exact and len(clients) > params.exact_client_limit:
        use_exact = False
        extra.append(
            warning(
                WARNING_EXACT_DISABLED,
                f"exact mode disabled because the instance has {len(clients)} clients "
                f"(limit {params.exact_client_limit})",
            )
        )
    if use_exact:
        solve = solve_mode_a_exact(
            clients, v, a, k, params.coverage_mode, rules, params.time_limit_s
        )
    else:
        solve = solve_mode_a_heuristic(
            clients, v, a, k, params.coverage_mode, rules
        )
        solve.proven_optimal = False
        solve.solver = SOLVER_HEURISTIC
    return solve, extra


def allocate_scenario(scenario: Scenario, *, include_baseline: Optional[bool] = None) -> dict[str, Any]:
    params = scenario.params
    n = len(scenario.clients)
    warnings: list[dict] = []
    if n == 0:
        empty = SolveResult(
            selected=[],
            status="infeasible",
            solver=SOLVER_EXACT if params.exact else SOLVER_HEURISTIC,
            proven_optimal=False,
            mip_gap=None,
            reasons=[],
            warnings=[],
        )
        return _build_result(scenario, empty, 0, 0, warnings)

    k = resolve_coverage_k(params.coverage_target, n)
    if k > n:
        warnings.append(
            warning(WARNING_CLAMPED, f"coverage target {k} exceeds {n} eligible clients; clamped")
        )
        k = n
    if k < 1:
        k = 1 if n >= 1 else 0
    if params.coverage_mode == "at_most_K":
        warnings.append(
            warning(
                WARNING_AT_MOST_K,
                "at_most_K is in use; a floor can shrink coverage instead of changing who is served",
            )
        )

    impossible = exclusive_share_sum_impossible(scenario.rules)
    if impossible:
        empty = SolveResult(
            selected=[False] * n,
            status="infeasible",
            solver=SOLVER_EXACT if params.exact else SOLVER_HEURISTIC,
            proven_optimal=False,
            mip_gap=None,
            reasons=["selected_by_solver"] * n,
            warnings=[warning(WARNING_RULE_UNMET, impossible)],
        )
        return _build_result(scenario, empty, k, n, warnings)

    solve, extra = _solve(scenario, k, scenario.rules)
    result = _build_result(scenario, solve, k, n, warnings + extra)

    compare = scenario.flags.compare_baseline if include_baseline is None else include_baseline
    if compare:
        baseline_scenario = Scenario(
            params=params,
            clients=scenario.clients,
            rules=[],
            centers=scenario.centers,
            flags=scenario.flags,
        )
        baseline_solve, baseline_extra = _solve(baseline_scenario, k, [])
        baseline = _build_result(baseline_scenario, baseline_solve, k, n, baseline_extra)
        result["baseline_comparison"] = {
            "baseline_summary": baseline["summary"],
            "delta": _numeric_summary_delta(result["summary"], baseline["summary"]),
        }
    return result


def allocate(payload: dict[str, Any]) -> dict[str, Any]:
    """Mode A entry point. FastAPI can call this with a request dict."""
    scenario = parse_scenario(payload)
    return allocate_scenario(scenario)
