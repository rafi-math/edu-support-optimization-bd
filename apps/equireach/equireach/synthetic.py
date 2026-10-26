from __future__ import annotations

import random
from typing import Optional


AGE_BANDS = ["6-12", "13-17"]
GENDERS = ["female", "male"]
DISABILITY = ["yes", "no"]
WEALTH = ["q1", "q2", "q3", "q4", "q5"]


def generate_selection_demo(
    n: int = 80,
    seed: int = 20261018,
    *,
    bbox: Optional[tuple[float, float, float, float]] = None,
) -> dict:
    """Tiny synthetic scenario for later API demo wiring. No real survey rows."""
    rng = random.Random(seed)
    min_lng, min_lat, max_lng, max_lat = bbox or (91.7, 22.2, 92.2, 22.7)
    clients = []
    for i in range(n):
        age = rng.choice(AGE_BANDS)
        wealth = rng.choice(WEALTH)
        age_boost = 0.15 if age == "13-17" else 0.0
        wealth_boost = 0.12 * (5 - int(wealth[1])) / 4.0
        noise = rng.uniform(-0.05, 0.05)
        priority = min(1.0, max(0.0, 0.45 + age_boost + wealth_boost + noise))
        need = 1 if (priority + rng.uniform(-0.2, 0.2)) >= 0.62 else 0
        clients.append(
            {
                "client_id": f"s{i+1:04d}",
                "label": None,
                "lat": rng.uniform(min_lat, max_lat),
                "lng": rng.uniform(min_lng, max_lng),
                "priority_score": round(priority, 4),
                "weight": round(rng.uniform(0.85, 1.15), 3),
                "attributes": {
                    "age_band": age,
                    "gender": rng.choice(GENDERS),
                    "disability": rng.choices(DISABILITY, weights=[1, 9], k=1)[0],
                    "wealth_quintile": wealth,
                },
                "need_flag": need,
            }
        )
    k = max(1, n // 5)
    return {
        "params": {
            "mode": "selection",
            "coverage_mode": "exactly_K",
            "coverage_target": {"type": "count", "value": k},
            "objective_weighting": "unweighted",
            "exact": True,
            "time_limit_s": 20,
            "source": "synthetic",
            "seed": seed,
        },
        "clients": clients,
        "centers": [],
        "rules": [],
        "flags": {"compare_baseline": False},
    }
