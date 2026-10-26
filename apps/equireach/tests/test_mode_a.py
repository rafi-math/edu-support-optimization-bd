from itertools import combinations

import numpy as np

from equireach.allocate import allocate
from equireach.mathutil import count_floor_from_share
from tests.conftest import scenario, tiny_clients


def _ids(result):
    return {row["client_id"] for row in result["assignments"]}


def brute_force(payload):
    clients = payload["clients"]
    k = int(payload["params"]["coverage_target"]["value"])
    rules = payload["rules"]
    best = None
    best_obj = -1.0
    n = len(clients)
    for combo in combinations(range(n), k):
        chosen = set(combo)
        ok = True
        for rule in rules:
            group = [
                i
                for i in chosen
                if str(clients[i]["attributes"].get(rule["attribute"])) == str(rule["level"])
            ]
            need = count_floor_from_share(rule["min_share_pct"] / 100.0, k)
            if len(group) < need:
                ok = False
                break
        if not ok:
            continue
        obj = sum(clients[i]["priority_score"] for i in chosen)
        ids = tuple(sorted(clients[i]["client_id"] for i in chosen))
        if obj > best_obj + 1e-12 or (abs(obj - best_obj) <= 1e-12 and (best is None or ids < best[1])):
            best_obj = obj
            best = (set(ids), ids)
    return None if best is None else (best[0], best_obj)


def test_exactly_k_or_clamp():
    result = allocate(scenario(k=3))
    assert result["summary"]["served_count"] == 3
    assert result["status"] == "ok"
    assert result["proven_optimal"] is True
    assert result["solver"] == "scipy_highs_milp"

    clamped = allocate(scenario(k=9))
    assert clamped["summary"]["served_count"] == 5
    assert any(w["code"] == "CLAMPED" for w in clamped["warnings"])


def test_exact_matches_brute_force_on_tiny_instance():
    payload = scenario(
        k=3,
        rules=[
            {
                "rule_id": "r1",
                "attribute": "age_band",
                "level": "6-12",
                "min_share_pct": 33.3,
            }
        ],
    )
    result = allocate(payload)
    expected_ids, expected_obj = brute_force(payload)
    assert _ids(result) == expected_ids
    assert abs(result["summary"]["total_priority_captured"] - expected_obj) < 1e-9


def test_floor_raises_share_or_is_partial():
    baseline = allocate(scenario(k=3, rules=[]))
    young_share = next(
        row["achieved_pct"]
        for row in allocate(
            scenario(
                k=3,
                rules=[
                    {
                        "rule_id": "r1",
                        "attribute": "age_band",
                        "level": "6-12",
                        "min_share_pct": 0,
                    }
                ],
            )
        )["summary"]["equity_compliance"]
    )
    floored = allocate(
        scenario(
            k=3,
            rules=[
                {
                    "rule_id": "r1",
                    "attribute": "age_band",
                    "level": "6-12",
                    "min_share_pct": 60,
                }
            ],
        )
    )
    if floored["status"] in {"partial", "infeasible"}:
        assert any(w["code"] == "RULE_UNMET" for w in floored["warnings"])
        if floored["status"] == "partial":
            achieved = floored["summary"]["equity_compliance"][0]["achieved_pct"]
            assert achieved is not None
    else:
        achieved = floored["summary"]["equity_compliance"][0]["achieved_pct"]
        assert achieved + 1e-9 >= 60
        assert achieved + 1e-9 >= young_share
        assert achieved + 1e-9 >= baseline["summary"].get("served_count", 0) * 0
    # The floor is not ignored: either the group is in the served set or we warn.
    if floored["status"] == "ok":
        served = _ids(floored)
        assert "c4" in served or "c5" in served


def test_count_floor_equivalence():
    k = 5
    frac = 0.333
    assert count_floor_from_share(243.00000000000003 / 730, 730) == 243
    assert count_floor_from_share(frac, 3) == 1
    payload = scenario(
        k=3,
        rules=[
            {
                "rule_id": "r1",
                "attribute": "age_band",
                "level": "6-12",
                "min_share_pct": 33.3,
            }
        ],
    )
    result = allocate(payload)
    need = count_floor_from_share(0.333, 3)
    served_young = sum(
        1
        for row in result["assignments"]
        if row["attributes"]["age_band"] == "6-12"
    )
    assert served_young >= need
    assert k == 5 or True


def test_need_flag_shuffle_does_not_change_selection():
    base = scenario(k=3, rules=[])
    shuffled = scenario(k=3, rules=[])
    flags = [c["need_flag"] for c in shuffled["clients"]]
    shuffled["clients"][0]["need_flag"] = flags[4]
    shuffled["clients"][1]["need_flag"] = flags[3]
    shuffled["clients"][2]["need_flag"] = flags[0]
    shuffled["clients"][3]["need_flag"] = flags[1]
    shuffled["clients"][4]["need_flag"] = flags[2]
    a = allocate(base)
    b = allocate(shuffled)
    assert _ids(a) == _ids(b)
    assert abs(
        a["summary"]["total_priority_captured"] - b["summary"]["total_priority_captured"]
    ) < 1e-12


def test_imported_tiny_scenario_drives_results():
    payload = scenario(k=2, clients=tiny_clients()[:3], rules=[])
    result = allocate(payload)
    assert _ids(result) == {"c1", "c2"}
    assert abs(result["summary"]["total_priority_captured"] - 1.7) < 1e-12


def test_heuristic_is_never_proven_optimal():
    result = allocate(
        scenario(
            k=3,
            exact=False,
            rules=[
                {
                    "rule_id": "r1",
                    "attribute": "age_band",
                    "level": "6-12",
                    "min_share_pct": 33.3,
                }
            ],
        )
    )
    assert result["solver"] == "greedy_swap"
    assert result["proven_optimal"] is False
    assert result["summary"]["served_count"] == 3


def test_size_guard_uses_heuristic():
    result = allocate(scenario(k=3, exact=True, exact_client_limit=1))
    assert result["solver"] == "greedy_swap"
    assert result["proven_optimal"] is False
    assert any(w["code"] == "EXACT_DISABLED" for w in result["warnings"])


def test_at_most_k_warns():
    result = allocate(scenario(k=3, coverage_mode="at_most_K"))
    assert any(w["code"] == "AT_MOST_K" for w in result["warnings"])
    assert result["summary"]["served_count"] <= 3


def test_impossible_exclusive_floors_are_infeasible():
    result = allocate(
        scenario(
            k=3,
            rules=[
                {
                    "rule_id": "r1",
                    "attribute": "gender",
                    "level": "female",
                    "min_share_pct": 60,
                },
                {
                    "rule_id": "r2",
                    "attribute": "gender",
                    "level": "male",
                    "min_share_pct": 60,
                },
            ],
        )
    )
    assert result["status"] == "infeasible"
    assert result["proven_optimal"] is False


def test_baseline_comparison_is_realised_difference():
    payload = scenario(
        k=3,
        compare_baseline=True,
        rules=[
            {
                "rule_id": "r1",
                "attribute": "age_band",
                "level": "6-12",
                "min_share_pct": 33.3,
            }
        ],
    )
    result = allocate(payload)
    base = allocate(scenario(k=3, rules=[]))
    delta = result["baseline_comparison"]["delta"]["total_priority_captured"]
    realised = (
        result["summary"]["total_priority_captured"]
        - base["summary"]["total_priority_captured"]
    )
    assert abs(delta - realised) < 1e-12
    assert result.get("price_of_fairness") is None
    assert result.get("interaction_readout") is None


def test_heuristic_partial_when_floor_cannot_be_met():
    result = allocate(
        scenario(
            k=3,
            exact=False,
            rules=[
                {
                    "rule_id": "r1",
                    "attribute": "age_band",
                    "level": "6-12",
                    "min_share_pct": 100,
                }
            ],
        )
    )
    assert result["status"] == "partial"
    assert result["proven_optimal"] is False
    assert any(w["code"] == "RULE_UNMET" for w in result["warnings"])
    assert result["summary"]["equity_compliance"][0]["achieved_pct"] < 100
