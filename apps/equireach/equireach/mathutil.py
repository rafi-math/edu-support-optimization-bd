from __future__ import annotations

import math

from equireach.constants import CEIL_EPS


def count_floor_from_share(min_share_frac: float, k: int) -> int:
    """Count-floor used in the validated study: ceil(f * K - 1e-9)."""
    if k <= 0 or min_share_frac <= 0:
        return 0
    return int(math.ceil(min_share_frac * k - CEIL_EPS))


def resolve_coverage_k(coverage_target, n: int) -> int:
    if coverage_target is None:
        raise ValueError("coverage_target is required")
    if isinstance(coverage_target, dict):
        kind = coverage_target.get("type", "count")
        value = coverage_target.get("value")
        if value is None:
            raise ValueError("coverage_target.value is required")
        if kind == "percent":
            return int(math.ceil(float(value) / 100.0 * n - CEIL_EPS))
        return int(math.floor(float(value) + CEIL_EPS))
    return int(math.floor(float(coverage_target) + CEIL_EPS))


def warning(code: str, message: str) -> dict:
    return {"code": code, "message": message}
