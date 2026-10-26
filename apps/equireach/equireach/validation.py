from __future__ import annotations

from typing import Any, Iterable, Optional

import pandas as pd

from equireach.constants import RESERVED_CLIENT_FIELDS
from equireach.errors import EquiReachValidationError
from equireach.models import Client, Flags, Params, Rule, Scenario


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return False


def _as_str(value: Any) -> Optional[str]:
    if _is_empty(value):
        return None
    text = str(value).strip()
    return text if text else None


def _parse_float(value: Any, field: str, row: int) -> tuple[Optional[float], Optional[dict]]:
    if _is_empty(value):
        return None, None
    try:
        return float(value), None
    except (TypeError, ValueError):
        return None, {
            "row": row,
            "field": field,
            "reason": "must be numeric",
        }


def _parse_int01(value: Any, field: str, row: int) -> tuple[Optional[int], Optional[dict]]:
    if _is_empty(value):
        return None, None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, {"row": row, "field": field, "reason": "must be 0 or 1"}
    if number not in (0, 1):
        return None, {"row": row, "field": field, "reason": "must be 0 or 1"}
    return int(number), None


def _row_is_empty(raw: dict[str, Any]) -> bool:
    if not raw:
        return True
    for value in raw.values():
        if isinstance(value, dict):
            if not _row_is_empty(value):
                return False
        elif not _is_empty(value):
            return False
    return True


def _attributes_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    nested = raw.get("attributes")
    attrs: dict[str, Any] = {}
    if isinstance(nested, dict):
        for key, value in nested.items():
            if not _is_empty(value):
                attrs[str(key)] = value if not isinstance(value, str) else value.strip()
    for key, value in raw.items():
        if key in RESERVED_CLIENT_FIELDS:
            continue
        if not _is_empty(value):
            attrs[str(key)] = value if not isinstance(value, str) else value.strip()
    return attrs


def validate_clients(
    rows: Iterable[dict[str, Any]],
    *,
    mode: str = "selection",
    start_row: int = 1,
) -> dict[str, Any]:
    """Validate client rows. `mode` is accepted for the future siting path."""
    del mode
    received = 0
    dropped_empty = 0
    errors: list[dict] = []
    clients: list[Client] = []
    seen_ids: dict[str, int] = {}
    attribute_levels: dict[str, set] = {}

    for offset, raw in enumerate(rows):
        row_number = start_row + offset
        received += 1
        if not isinstance(raw, dict) or _row_is_empty(raw):  # drop blank records, do not score them
            dropped_empty += 1
            continue

        row_errors: list[dict] = []
        client_id = _as_str(raw.get("client_id"))
        if client_id is None:
            row_errors.append(
                {"row": row_number, "field": "client_id", "reason": "is required"}
            )
        elif client_id in seen_ids:
            row_errors.append(
                {
                    "row": row_number,
                    "field": "client_id",
                    "reason": f"duplicate of row {seen_ids[client_id]}",
                }
            )
        else:
            seen_ids[client_id] = row_number

        score, score_err = _parse_float(raw.get("priority_score"), "priority_score", row_number)
        if raw.get("priority_score") is None or _is_empty(raw.get("priority_score")):
            row_errors.append(
                {"row": row_number, "field": "priority_score", "reason": "is required"}
            )
        elif score_err:
            row_errors.append(score_err)
        elif score is None or score < 0 or score > 1:
            row_errors.append(
                {
                    "row": row_number,
                    "field": "priority_score",
                    "reason": "must be between 0 and 1",
                }
            )

        weight_raw = raw.get("weight", 1)
        if _is_empty(weight_raw):
            weight = 1.0
            weight_err = None
        else:
            weight, weight_err = _parse_float(weight_raw, "weight", row_number)
        if weight_err:
            row_errors.append(weight_err)
        elif weight is None or weight <= 0:
            row_errors.append(
                {"row": row_number, "field": "weight", "reason": "must be greater than 0"}
            )

        need_flag, need_err = _parse_int01(raw.get("need_flag"), "need_flag", row_number)
        if need_err:
            row_errors.append(need_err)

        if row_errors:
            errors.extend(row_errors)
            continue

        attrs = _attributes_from_raw(raw)
        for key, value in attrs.items():
            attribute_levels.setdefault(key, set()).add(value)
        clients.append(
            Client(
                client_id=client_id or "",
                priority_score=float(score),
                weight=float(weight if weight is not None else 1.0),
                label=_as_str(raw.get("label")),
                lat=None if _is_empty(raw.get("lat")) else raw.get("lat"),
                lng=None if _is_empty(raw.get("lng")) else raw.get("lng"),
                attributes=attrs,
                need_flag=need_flag,
                source_row=row_number,
            )
        )

    return {
        "ok": len(errors) == 0,
        "clients": {
            "received": received,
            "valid": len(clients),
            "dropped_empty": dropped_empty,
            "errors": errors,
        },
        "centers": {"received": 0, "valid": 0, "dropped_empty": 0, "errors": []},
        "attribute_hints": {k: sorted(v, key=str) for k, v in attribute_levels.items()},
        "parsed_clients": clients,
    }


def load_clients_csv(path: str) -> dict[str, Any]:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = frame.to_dict(orient="records")
    # Header is line 1; first data row is line 2.
    return validate_clients(rows, start_row=2)


def _parse_rule(raw: dict[str, Any], index: int) -> Rule:
    if not isinstance(raw, dict):
        raise EquiReachValidationError("each rule must be an object")
    rule_id = _as_str(raw.get("rule_id")) or f"r{index + 1}"
    attribute = _as_str(raw.get("attribute"))
    level = raw.get("level")
    if attribute is None:
        raise EquiReachValidationError("rule.attribute is required")
    if _is_empty(level):
        raise EquiReachValidationError("rule.level is required")
    min_share = raw.get("min_share_pct")
    max_share = raw.get("max_share_pct")
    min_count = raw.get("min_count")
    if min_share is not None and not _is_empty(min_share):
        try:
            min_share = float(min_share)
        except (TypeError, ValueError) as exc:
            raise EquiReachValidationError("min_share_pct must be numeric") from exc
        if min_share < 0 or min_share > 100:
            raise EquiReachValidationError("min_share_pct must be between 0 and 100")
    else:
        min_share = None
    if max_share is not None and not _is_empty(max_share):
        try:
            max_share = float(max_share)
        except (TypeError, ValueError) as exc:
            raise EquiReachValidationError("max_share_pct must be numeric") from exc
        if max_share < 0 or max_share > 100:
            raise EquiReachValidationError("max_share_pct must be between 0 and 100")
    else:
        max_share = None
    if min_count is not None and not _is_empty(min_count):
        try:
            min_count = int(min_count)
        except (TypeError, ValueError) as exc:
            raise EquiReachValidationError("min_count must be an integer") from exc
        if min_count < 0:
            raise EquiReachValidationError("min_count must be >= 0")
    else:
        min_count = None
    if min_share is None and min_count is None:
        min_share = 0.0
    return Rule(
        rule_id=rule_id,
        attribute=attribute,
        level=str(level),
        min_share_pct=min_share,
        max_share_pct=max_share,
        min_count=min_count,
    )


def _parse_params(raw: Optional[dict[str, Any]]) -> Params:
    raw = raw or {}
    coverage_mode = raw.get("coverage_mode") or "exactly_K"
    if coverage_mode not in {"exactly_K", "at_most_K"}:
        raise EquiReachValidationError("coverage_mode must be exactly_K or at_most_K")
    weighting = raw.get("objective_weighting") or "unweighted"
    if weighting not in {"unweighted", "weighted"}:
        raise EquiReachValidationError(
            "objective_weighting must be unweighted or weighted"
        )
    mode = raw.get("mode") or "selection"
    coverage_target = raw.get("coverage_target")
    if coverage_target is None:
        raise EquiReachValidationError("coverage_target is required")
    if isinstance(coverage_target, dict):
        value = coverage_target.get("value")
        if value is None:
            raise EquiReachValidationError("coverage_target.value is required")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise EquiReachValidationError("coverage_target.value must be numeric") from exc
        if numeric < 1 and coverage_target.get("type", "count") == "count":
            raise EquiReachValidationError("coverage_target must be >= 1")
    else:
        try:
            numeric = float(coverage_target)
        except (TypeError, ValueError) as exc:
            raise EquiReachValidationError("coverage_target must be numeric") from exc
        if numeric < 1:
            raise EquiReachValidationError("coverage_target must be >= 1")
    time_limit = raw.get("time_limit_s", 20)
    try:
        time_limit = float(time_limit)
    except (TypeError, ValueError) as exc:
        raise EquiReachValidationError("time_limit_s must be numeric") from exc
    return Params(
        mode=str(mode),
        coverage_mode=coverage_mode,
        coverage_target=coverage_target,
        objective_weighting=weighting,
        exact=bool(raw.get("exact", True)),
        time_limit_s=time_limit,
        source=str(raw.get("source") or "imported"),
        seed=raw.get("seed"),
        sweep_time_limit_s=float(raw.get("sweep_time_limit_s", 55)),
        exact_client_limit=int(raw.get("exact_client_limit", 20_000)),
    )


def parse_scenario(payload: dict[str, Any]) -> Scenario:
    if not isinstance(payload, dict):
        raise EquiReachValidationError("scenario must be an object")
    params = _parse_params(payload.get("params"))
    raw_clients = payload.get("clients") or []
    if not isinstance(raw_clients, list):
        raise EquiReachValidationError("clients must be a list")
    report = validate_clients(raw_clients, mode=params.mode)
    if report["clients"]["errors"]:
        raise EquiReachValidationError(
            "one or more client rows are invalid",
            details=report["clients"]["errors"],
        )
    clients: list[Client] = report["parsed_clients"]
    raw_rules = payload.get("rules") or []
    if not isinstance(raw_rules, list):
        raise EquiReachValidationError("rules must be a list")
    if len(raw_rules) > 20:
        raise EquiReachValidationError("at most 20 rules are allowed")
    rules = [_parse_rule(item, i) for i, item in enumerate(raw_rules)]
    _check_rules_exist_on_clients(clients, rules)
    flags_raw = payload.get("flags") or {}
    flags = Flags(compare_baseline=bool(flags_raw.get("compare_baseline", False)))
    centers = payload.get("centers") or []
    if params.mode == "siting" and centers:
        raise EquiReachValidationError(
            "Mode B siting is not implemented in this engine build"
        )
    return Scenario(params=params, clients=clients, rules=rules, centers=list(centers), flags=flags)


def _check_rules_exist_on_clients(clients: list[Client], rules: list[Rule]) -> None:
    for rule in rules:
        found = any(
            str(client.attributes.get(rule.attribute)) == str(rule.level)
            for client in clients
        )
        if not found:
            raise EquiReachValidationError(
                f"rule {rule.rule_id} names attribute '{rule.attribute}' "
                f"level '{rule.level}', which is not present on any client"
            )


def exclusive_share_sum_impossible(rules: list[Rule]) -> Optional[str]:
    by_attr: dict[str, list[Rule]] = {}
    for rule in rules:
        by_attr.setdefault(rule.attribute, []).append(rule)
    for attribute, group in by_attr.items():
        share_sum = 0.0
        levels = []
        for rule in group:
            if rule.min_share_pct:
                share_sum += rule.min_share_pct
                levels.append(rule.level)
        unique_levels = {str(level) for level in levels}
        if len(unique_levels) >= 2 and share_sum > 100 + 1e-9:
            return (
                f"min_share_pct values for mutually exclusive levels of "
                f"'{attribute}' add up to more than 100%"
            )
    return None
