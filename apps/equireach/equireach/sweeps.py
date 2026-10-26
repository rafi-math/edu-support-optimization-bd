from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Optional

from equireach.allocate import allocate_scenario
from equireach.constants import (
    INTERACTION_EPS_FLOOR,
    INTERACTION_EPS_FRAC,
    NO_COST_THRESHOLD,
    PRICE_SWEEP_STEPS,
    WARNING_SWEEP_PARTIAL,
)
from equireach.errors import EquiReachValidationError
from equireach.mathutil import warning
from equireach.models import Params, Rule, Scenario
from equireach.validation import parse_scenario


def _force_exactly_k(params: Params) -> Params:
    clone = deepcopy(params)
    clone.coverage_mode = "exactly_K"
    return clone


def _with_rules(scenario: Scenario, rules: list[Rule]) -> Scenario:
    return Scenario(
        params=_force_exactly_k(scenario.params),
        clients=scenario.clients,
        rules=rules,
        centers=scenario.centers,
        flags=scenario.flags,
    )


def _objective(result: dict[str, Any]) -> float:
    if result.get("status") == "infeasible":
        return float("-inf")
    return float(result["summary"]["total_priority_captured"])


def _need(result: dict[str, Any]) -> Optional[float]:
    return result["summary"].get("need_capture_pct")


def _achieved_for_rule(result: dict[str, Any], rule: Rule) -> Optional[float]:
    for row in result["summary"].get("equity_compliance") or []:
        if row.get("attribute") == rule.attribute and str(row.get("level")) == str(rule.level):
            return float(row["achieved_pct"])
    return None


def _run(scenario: Scenario, rules: list[Rule]) -> dict[str, Any]:
    return allocate_scenario(_with_rules(scenario, rules), include_baseline=False)


def _unconstrained(scenario: Scenario) -> dict[str, Any]:
    return _run(scenario, [])


def _replace_min(rule: Rule, min_pct: float) -> Rule:
    clone = deepcopy(rule)
    clone.min_share_pct = min_pct
    clone.min_count = None
    return clone


def price_of_fairness(
    scenario: Scenario,
    rule_id: str,
    *,
    steps: int = PRICE_SWEEP_STEPS,
    others: str = "off",
    deadline: Optional[float] = None,
) -> tuple[list[dict], list[dict]]:
    target = next((r for r in scenario.rules if r.rule_id == rule_id), None)
    if target is None:
        raise EquiReachValidationError(f"rule_id '{rule_id}' was not found")
    other_rules = [r for r in scenario.rules if r.rule_id != rule_id]
    keep_others = others == "current"

    zero_rules = [_replace_min(target, 0.0)]
    if keep_others:
        zero_rules = [_replace_min(target, 0.0)] + other_rules
    zero_result = _run(scenario, zero_rules)
    zero_obj = _objective(zero_result)
    rows: list[dict] = []
    warnings: list[dict] = []

    n_steps = max(2, int(steps))
    candidates = [i * 100.0 / (n_steps - 1) for i in range(n_steps)]
    for min_pct in candidates:
        if deadline is not None and time.monotonic() >= deadline:
            warnings.append(
                warning(WARNING_SWEEP_PARTIAL, "sweep stopped because sweep_time_limit_s was reached")
            )
            break
        active = [_replace_min(target, min_pct)]
        if keep_others:
            active = [_replace_min(target, min_pct)] + other_rules
        result = _run(scenario, active)
        if result["status"] == "infeasible":
            break
        obj = _objective(result)
        loss = 0.0 if min_pct == 0.0 else max(0.0, zero_obj - obj)
        if min_pct == 0.0:
            loss = 0.0
            obj = zero_obj
        rows.append(
            {
                "rule_id": rule_id,
                "min_pct": min_pct,
                "objective": obj,
                "priority_loss_vs_zero": loss,
                "need_capture_pct": _need(result),
                "served_count": result["summary"]["served_count"],
                "achieved_share_pct": _achieved_for_rule(result, target),
                "status": result["status"],
                "proven_optimal": result["proven_optimal"],
            }
        )
    return rows, warnings


def interaction_readout(
    scenario: Scenario,
    *,
    deadline: Optional[float] = None,
) -> tuple[Optional[dict], list[dict]]:
    rules = scenario.rules
    if len(rules) < 2:
        return None, []
    warnings: list[dict] = []
    if deadline is not None and time.monotonic() >= deadline:
        warnings.append(
            warning(WARNING_SWEEP_PARTIAL, "sweep stopped because sweep_time_limit_s was reached")
        )
        return None, warnings

    unconstrained = _unconstrained(scenario)
    z0 = _objective(unconstrained)
    both = _run(scenario, rules)
    z_both = _objective(both)
    cost_both = max(0.0, z0 - z_both)
    cost_each = []
    for rule in rules:
        if deadline is not None and time.monotonic() >= deadline:
            warnings.append(
                warning(WARNING_SWEEP_PARTIAL, "sweep stopped because sweep_time_limit_s was reached")
            )
            return None, warnings
        one = _run(scenario, [rule])
        cost_each.append(
            {
                "rule_id": rule.rule_id,
                "cost": max(0.0, z0 - _objective(one)),
            }
        )
    interaction = cost_both - sum(item["cost"] for item in cost_each)
    if cost_both <= NO_COST_THRESHOLD:
        label = "NO_COST"
        pct = 0.0
    else:
        epsilon = max(INTERACTION_EPS_FLOOR, INTERACTION_EPS_FRAC * cost_both)
        if interaction < -epsilon:
            label = "SUBADDITIVE"
        elif interaction > epsilon:
            label = "SUPERADDITIVE"
        else:
            label = "ADDITIVE"
        pct = 100.0 * interaction / cost_both
    return {
        "rule_ids": [rule.rule_id for rule in rules],
        "cost_both": cost_both,
        "cost_each": cost_each,
        "interaction": interaction,
        "interaction_pct_of_cost_both": pct,
        "epsilon_pct": 100.0 * INTERACTION_EPS_FRAC,
        "label": label,
    }, warnings


def run_sweep(
    payload: dict[str, Any],
    *,
    kind: str,
    rule_id: Optional[str] = None,
    steps: int = PRICE_SWEEP_STEPS,
    others: str = "off",
    sweep_time_limit_s: Optional[float] = None,
) -> dict[str, Any]:
    """Sweep entry point. FastAPI can wrap this for POST /api/sweep."""
    if "scenario" in payload and "params" not in payload:
        scenario_payload = payload["scenario"]
        kind = payload.get("kind", kind)
        rule_id = payload.get("rule_id", rule_id)
        steps = int(payload.get("steps", steps))
        others = payload.get("others", others)
        sweep_time_limit_s = payload.get("sweep_time_limit_s", sweep_time_limit_s)
    else:
        scenario_payload = payload
    scenario = parse_scenario(scenario_payload)
    limit = (
        float(sweep_time_limit_s)
        if sweep_time_limit_s is not None
        else float(scenario.params.sweep_time_limit_s)
    )
    deadline = time.monotonic() + limit
    warnings: list[dict] = []
    price_rows = None
    readout = None

    if kind not in {"price", "interaction", "both"}:
        raise EquiReachValidationError("kind must be price, interaction, or both")
    if kind in {"price", "both"}:
        if not rule_id:
            raise EquiReachValidationError("rule_id is required for price sweeps")
        price_rows, extra = price_of_fairness(
            scenario, rule_id, steps=steps, others=others, deadline=deadline
        )
        warnings.extend(extra)
    if kind in {"interaction", "both"}:
        readout, extra = interaction_readout(scenario, deadline=deadline)
        warnings.extend(extra)
        if readout is None and kind == "interaction" and len(scenario.rules) < 2:
            raise EquiReachValidationError("interaction needs 2 or more rules")

    return {
        "price_of_fairness": price_rows,
        "interaction_readout": readout,
        "warnings": warnings,
    }
