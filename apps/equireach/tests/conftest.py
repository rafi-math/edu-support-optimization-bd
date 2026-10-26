from copy import deepcopy


def tiny_clients():
    return [
        {
            "client_id": "c1",
            "label": None,
            "lat": None,
            "lng": None,
            "priority_score": 0.9,
            "weight": 1,
            "attributes": {"age_band": "13-17", "gender": "female"},
            "need_flag": 1,
        },
        {
            "client_id": "c2",
            "label": None,
            "lat": None,
            "lng": None,
            "priority_score": 0.8,
            "weight": 1,
            "attributes": {"age_band": "13-17", "gender": "male"},
            "need_flag": 1,
        },
        {
            "client_id": "c3",
            "label": None,
            "lat": None,
            "lng": None,
            "priority_score": 0.7,
            "weight": 1,
            "attributes": {"age_band": "13-17", "gender": "male"},
            "need_flag": 0,
        },
        {
            "client_id": "c4",
            "label": None,
            "lat": None,
            "lng": None,
            "priority_score": 0.2,
            "weight": 1,
            "attributes": {"age_band": "6-12", "gender": "female"},
            "need_flag": 0,
        },
        {
            "client_id": "c5",
            "label": None,
            "lat": None,
            "lng": None,
            "priority_score": 0.1,
            "weight": 1,
            "attributes": {"age_band": "6-12", "gender": "male"},
            "need_flag": 0,
        },
    ]


def scenario(k=3, rules=None, clients=None, **params):
    payload = {
        "params": {
            "mode": "selection",
            "coverage_mode": params.get("coverage_mode", "exactly_K"),
            "coverage_target": {"type": "count", "value": k},
            "objective_weighting": params.get("objective_weighting", "unweighted"),
            "exact": params.get("exact", True),
            "time_limit_s": params.get("time_limit_s", 20),
            "source": "synthetic",
            "seed": 1,
            "exact_client_limit": params.get("exact_client_limit", 20000),
        },
        "clients": deepcopy(clients or tiny_clients()),
        "centers": [],
        "rules": deepcopy(rules or []),
        "flags": {"compare_baseline": params.get("compare_baseline", False)},
    }
    return payload
