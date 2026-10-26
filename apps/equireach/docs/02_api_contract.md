# EquiReach API contract (file 2 of 3)

**Give this file to:** Cursor, in the same chat as file 1. Lovable does not need it, because file 3 lists the fields the screen reads.
**Job:** the exact shape of every request and response. File 1 says what the engine computes. This file says how it is sent.

## General rules

- JSON over HTTPS, UTF-8. No login. Nothing is stored.
- CORS: allowed origins come from `ALLOWED_ORIGINS`.
- Limits (also returned by `/api/config`): 50,000 clients, 500 centers, 20 rules, 15 MB per request.
- Errors: `{"error": {"code": "...", "message": "...", "details": []}}`. HTTP 400 for invalid input, 413 for a request that is too large, 500 for unexpected failures.
- A scenario that cannot be satisfied is not an error. Return HTTP 200 with `status` set to `infeasible` or `partial`, and a warning.
- A warning is `{"code": "...", "message": "..."}`. Codes: `RULE_UNMET`, `TRAVEL_EXCLUDES`, `BUDGET_BINDS`, `BUDGET_UNBOUNDED`, `EXACT_DISABLED`, `AT_MOST_K`, `TIME_LIMIT`, `SWEEP_PARTIAL`, `CLAMPED`.
- Percent fields are numbers from 0 to 100. Distances are kilometers.

## Endpoints

```text
GET  /health
GET  /api/config
GET  /api/demo?kind=selection|siting&seed=20261018&bbox=minLng,minLat,maxLng,maxLat
POST /api/validate
POST /api/allocate
POST /api/sweep
```

## GET /health

```json
{"ok": true, "engine_version": "0.1.0", "highs": true}
```

## GET /api/config

```json
{
  "tool": "EquiReach",
  "tagline": "Optimizing Resources. Advancing Fairness.",
  "engine_version": "0.1.0",
  "defaults": {
    "coverage_mode": "exactly_K",
    "objective_weighting": "unweighted",
    "exact": true,
    "time_limit_s": 20,
    "max_travel_km": 10,
    "center_mode": "open_exactly_N",
    "n_centers": 5
  },
  "bounds": {
    "max_travel_km": [0.5, 50],
    "min_share_pct": [0, 100],
    "time_limit_s": [1, 60]
  },
  "presets": [
    {"id": "parity", "label": "Population parity (50% for a two-level attribute)"},
    {"id": "proportional", "label": "Proportional (the group's share among all clients)"}
  ],
  "disclaimer_text": "Synthetic demo data unless you imported your own. Coordinates and distances are illustrative.",
  "demo_bbox": [91.7, 22.2, 92.2, 22.7],
  "limits": {"max_clients": 50000, "max_centers": 500, "max_rules": 20, "max_request_mb": 15}
}
```

The `demo_bbox` values are placeholders. Use a generic rural box and let the caller change it.

## GET /api/demo

Returns a full scenario (same shape as the request body of `/api/allocate`) with `params.source` set to `synthetic` and the seed filled in. `kind=selection` returns clients only. `kind=siting` returns clients and centers.

## POST /api/validate

Request: the rows the screen has mapped, with values still raw (numbers may be strings).

```json
{
  "mode": "selection",
  "clients": [
    {"client_id": "c1", "label": null, "lat": null, "lng": null, "priority_score": "0.8", "weight": "1",
     "attributes": {"age_band": "13-17", "gender": "male"}, "need_flag": "1"}
  ],
  "centers": []
}
```

Response:

```json
{
  "ok": true,
  "clients": {"received": 1, "valid": 1, "dropped_empty": 0, "errors": []},
  "centers": {"received": 0, "valid": 0, "dropped_empty": 0, "errors": []},
  "attribute_hints": {"age_band": ["13-17"], "gender": ["male"]}
}
```

Each error is `{"row": 12, "field": "priority_score", "reason": "must be between 0 and 1"}`. `row` counts from 1.

## POST /api/allocate

Request (the scenario). Fields marked null are not used in Mode A.

```json
{
  "params": {
    "mode": "selection",
    "coverage_mode": "exactly_K",
    "coverage_target": {"type": "count", "value": 2},
    "objective_weighting": "unweighted",
    "exact": true,
    "time_limit_s": 20,
    "max_travel_km": null,
    "center_mode": null,
    "n_centers": null,
    "max_centers": null,
    "budget": null,
    "source": "imported",
    "seed": null
  },
  "clients": [
    {"client_id": "c1", "label": null, "lat": null, "lng": null, "priority_score": 0.8, "weight": 1,
     "attributes": {"age_band": "13-17", "gender": "male"}, "need_flag": 1},
    {"client_id": "c2", "label": null, "lat": null, "lng": null, "priority_score": 0.6, "weight": 1,
     "attributes": {"age_band": "6-12", "gender": "female"}, "need_flag": 0},
    {"client_id": "c3", "label": null, "lat": null, "lng": null, "priority_score": 0.5, "weight": 1,
     "attributes": {"age_band": "6-12", "gender": "male"}, "need_flag": 0}
  ],
  "centers": [],
  "rules": [
    {"rule_id": "r1", "attribute": "age_band", "level": "6-12", "min_share_pct": 50, "max_share_pct": null}
  ],
  "flags": {"compare_baseline": true}
}
```

In Mode B, `params.mode` is `siting`, `lat` and `lng` are required on clients and centers, and a center looks like this:

```json
{"center_id": "k1", "label": "Center 1", "lat": 22.4, "lng": 91.9, "capacity": 120, "build_cost": 5000, "operate_cost_per_client": 10, "tags": []}
```

Response (values here are illustrative for the scenario above):

```json
{
  "status": "ok",
  "solver": "scipy_highs_milp",
  "proven_optimal": true,
  "mip_gap": 0.0,
  "engine_version": "0.1.0",
  "provenance": {
    "tool": "EquiReach",
    "tagline": "Optimizing Resources. Advancing Fairness.",
    "source": "imported",
    "seed": null,
    "mode": "selection",
    "created_utc": "2026-10-24T09:00:00Z"
  },
  "inputs_echo": {"params": {}, "rules": []},
  "summary": {
    "served_count": 2,
    "coverage_target": 2,
    "coverage_met": true,
    "shortfall": 0,
    "eligible_count": 3,
    "ineligible_count": 0,
    "coverage_pct": 66.7,
    "total_priority_captured": 1.4,
    "mean_priority_served": 0.7,
    "need_capture_pct": 100.0,
    "equity_compliance": [
      {"attribute": "age_band", "level": "6-12", "min_pct": 50.0, "achieved_pct": 50.0, "met": true}
    ],
    "cost_total": null,
    "budget_used_pct": null,
    "centers_opened": 0,
    "center_ids": [],
    "avg_travel_km": null,
    "max_travel_km_used": null,
    "p90_travel_km": null,
    "solver": "scipy_highs_milp",
    "proven_optimal": true,
    "mip_gap": 0.0,
    "status": "ok"
  },
  "centers": [],
  "assignments": [
    {"client_id": "c1", "label": null, "lat": null, "lng": null, "assigned_center_id": null, "assigned_center_label": null,
     "distance_km": null, "priority_score": 0.8, "weight": 1, "attributes": {"age_band": "13-17", "gender": "male"},
     "need_flag": 1, "selection_reason": "selected_by_solver"},
    {"client_id": "c2", "label": null, "lat": null, "lng": null, "assigned_center_id": null, "assigned_center_label": null,
     "distance_km": null, "priority_score": 0.6, "weight": 1, "attributes": {"age_band": "6-12", "gender": "female"},
     "need_flag": 0, "selection_reason": "selected_by_solver"}
  ],
  "unserved": [
    {"client_id": "c3", "state": "eligible_unserved"}
  ],
  "preview": [],
  "baseline_comparison": {
    "baseline_summary": {},
    "delta": {"total_priority_captured": 0.0, "need_capture_pct": 0.0, "served_count": 0}
  },
  "warnings": []
}
```

Notes:

- `inputs_echo` repeats the parameters and rules actually used, so an export is reproducible. Fill it fully; it is shown empty above to save space.
- `preview` holds the top 50 served by priority (same row shape as `assignments`).
- `baseline_comparison` is null unless `flags.compare_baseline` is true. `baseline_summary` has the same shape as `summary`. `delta` is current minus baseline for each numeric summary field.
- `selection_reason` is one of `selected_by_solver`, `top_priority`, `swapped_in_for_rule:<rule_id>`.
- `state` in `unserved` is `eligible_unserved` or `ineligible`.

## POST /api/sweep

Request: the same scenario as `/api/allocate`, wrapped.

```text
{
  "scenario": <the scenario object above>,
  "kind": "price" | "interaction" | "both",
  "rule_id": "r1",
  "steps": 12,
  "others": "off" | "current"
}
```

`rule_id` is required for `price` and `both`. `interaction` needs 2 or more rules in the scenario. The call stops after `sweep_time_limit_s` (default 55) and returns what it finished with a `SWEEP_PARTIAL` warning.

Response (illustrative values):

```json
{
  "price_of_fairness": [
    {"rule_id": "r1", "min_pct": 0.0, "objective": 1.4, "priority_loss_vs_zero": 0.0, "need_capture_pct": 100.0,
     "served_count": 2, "achieved_share_pct": 50.0, "status": "ok", "proven_optimal": true},
    {"rule_id": "r1", "min_pct": 50.0, "objective": 1.4, "priority_loss_vs_zero": 0.0, "need_capture_pct": 100.0,
     "served_count": 2, "achieved_share_pct": 50.0, "status": "ok", "proven_optimal": true}
  ],
  "interaction_readout": {
    "rule_ids": ["r1", "r2"],
    "cost_both": 0.30,
    "cost_each": [{"rule_id": "r1", "cost": 0.20}, {"rule_id": "r2", "cost": 0.15}],
    "interaction": -0.05,
    "interaction_pct_of_cost_both": -16.7,
    "epsilon_pct": 0.5,
    "label": "SUBADDITIVE"
  },
  "warnings": []
}
```

`interaction_readout` is null when `kind` is `price` or when fewer than 2 rules exist. `label` is one of `SUBADDITIVE`, `SUPERADDITIVE`, `ADDITIVE`, `NO_COST`.

## Contract checks (add to the tests)

- Every response example in this file validates against the pydantic models.
- Unknown fields in a request are ignored. Missing required fields give HTTP 400 with the field name.
- `/api/allocate` never returns `price_of_fairness` or `interaction_readout`. `/api/sweep` never returns `assignments`.
