from unittest.mock import patch

import numpy as np

from equireach.allocate import allocate
from equireach.constants import SOLVER_EXACT
from tests.conftest import scenario


class _FakeMilp:
    def __init__(self, status, x, gap=0.05):
        self.status = status
        self.success = status == 0
        self.x = x
        self.fun = -float(np.dot(x, [0.9, 0.8, 0.7, 0.2, 0.1])) if x is not None else None
        self.mip_gap = gap
        self.message = "fake"


def test_proven_optimal_only_when_highs_is_optimal():
    result = allocate(scenario(k=2))
    assert result["solver"] == SOLVER_EXACT
    assert result["proven_optimal"] is True
    assert result["mip_gap"] == 0.0


def test_time_limited_solve_keeps_gap_and_is_not_proven():
    x = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    with patch("equireach.exact.milp", return_value=_FakeMilp(1, x, gap=0.12)):
        result = allocate(scenario(k=2))
    assert result["proven_optimal"] is False
    assert result["solver"] == SOLVER_EXACT
    assert abs(result["mip_gap"] - 0.12) < 1e-12
    assert any(w["code"] == "TIME_LIMIT" for w in result["warnings"])
    assert result["summary"]["served_count"] == 2


def test_heuristic_proven_optimal_is_false():
    result = allocate(scenario(k=2, exact=False))
    assert result["solver"] == "greedy_swap"
    assert result["proven_optimal"] is False
