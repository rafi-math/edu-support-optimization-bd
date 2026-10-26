import pytest

from equireach.errors import EquiReachValidationError
from equireach.validation import parse_scenario, validate_clients


def test_bad_priority_and_duplicate_ids_are_rejected():
    report = validate_clients(
        [
            {"client_id": "c1", "priority_score": 1.2, "weight": 1},
            {"client_id": "c1", "priority_score": 0.4, "weight": 0},
            {"client_id": "c2", "priority_score": "nope"},
            {},
        ]
    )
    reasons = {(e["field"], e["reason"]) for e in report["clients"]["errors"]}
    assert ("priority_score", "must be between 0 and 1") in reasons
    assert ("weight", "must be greater than 0") in reasons
    assert ("priority_score", "must be numeric") in reasons
    assert any(e["field"] == "client_id" and "duplicate" in e["reason"] for e in report["clients"]["errors"])
    assert report["clients"]["dropped_empty"] == 1
    assert report["ok"] is False
    assert all(e["row"] >= 1 for e in report["clients"]["errors"])


def test_unknown_rule_level_is_an_error():
    with pytest.raises(EquiReachValidationError) as exc:
        parse_scenario(
            {
                "params": {
                    "mode": "selection",
                    "coverage_mode": "exactly_K",
                    "coverage_target": {"type": "count", "value": 1},
                    "objective_weighting": "unweighted",
                    "exact": True,
                    "time_limit_s": 5,
                },
                "clients": [
                    {
                        "client_id": "c1",
                        "priority_score": 0.5,
                        "weight": 1,
                        "attributes": {"gender": "female"},
                    }
                ],
                "rules": [
                    {
                        "rule_id": "r1",
                        "attribute": "gender",
                        "level": "nonexistent",
                        "min_share_pct": 40,
                    }
                ],
            }
        )
    assert "nonexistent" in exc.value.message


def test_empty_and_zero_inputs_do_not_crash():
    from equireach.allocate import allocate

    empty = allocate(
        {
            "params": {
                "mode": "selection",
                "coverage_mode": "exactly_K",
                "coverage_target": {"type": "count", "value": 1},
                "objective_weighting": "unweighted",
                "exact": True,
                "time_limit_s": 5,
            },
            "clients": [],
            "rules": [],
        }
    )
    assert empty["status"] == "infeasible"
    assert empty["summary"]["served_count"] == 0
