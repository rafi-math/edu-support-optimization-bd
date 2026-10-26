from equireach.allocate import allocate
from equireach.sweeps import run_sweep
from tests.conftest import scenario


def test_price_of_fairness_loss_is_zero_at_zero_and_monotone():
    payload = scenario(
        k=3,
        rules=[
            {
                "rule_id": "r1",
                "attribute": "age_band",
                "level": "6-12",
                "min_share_pct": 40,
            }
        ],
    )
    result = run_sweep(payload, kind="price", rule_id="r1", steps=12)
    rows = result["price_of_fairness"]
    assert rows, "expected at least the 0% step"
    assert abs(rows[0]["min_pct"]) < 1e-12
    assert abs(rows[0]["priority_loss_vs_zero"]) < 1e-12
    losses = [row["priority_loss_vs_zero"] for row in rows]
    for earlier, later in zip(losses, losses[1:]):
        assert later + 1e-9 >= earlier
    assert "assignments" not in result


def test_interaction_uses_real_runs_and_epsilon_rule():
    payload = scenario(
        k=3,
        rules=[
            {
                "rule_id": "r1",
                "attribute": "age_band",
                "level": "6-12",
                "min_share_pct": 33.3,
            },
            {
                "rule_id": "r2",
                "attribute": "gender",
                "level": "female",
                "min_share_pct": 33.3,
            },
        ],
    )
    result = run_sweep(payload, kind="both", rule_id="r1", steps=6)
    readout = result["interaction_readout"]
    assert readout is not None
    unconstrained = allocate(scenario(k=3, rules=[]))
    both = allocate(payload)
    z0 = unconstrained["summary"]["total_priority_captured"]
    cost_both = z0 - both["summary"]["total_priority_captured"]
    cost_r1 = z0 - allocate(
        scenario(
            k=3,
            rules=[payload["rules"][0]],
        )
    )["summary"]["total_priority_captured"]
    cost_r2 = z0 - allocate(
        scenario(
            k=3,
            rules=[payload["rules"][1]],
        )
    )["summary"]["total_priority_captured"]
    interaction = cost_both - (cost_r1 + cost_r2)
    assert abs(readout["cost_both"] - cost_both) < 1e-9
    assert abs(readout["interaction"] - interaction) < 1e-9
    assert readout["label"] in {"SUBADDITIVE", "SUPERADDITIVE", "ADDITIVE", "NO_COST"}
    assert readout["epsilon_pct"] == 0.5
    if readout["label"] == "NO_COST":
        assert readout["cost_both"] <= 1e-9
    else:
        assert abs(
            readout["interaction_pct_of_cost_both"]
            - 100.0 * readout["interaction"] / readout["cost_both"]
        ) < 1e-9
