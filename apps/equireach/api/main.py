from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastapi.responses import RedirectResponse

from equireach import allocate, run_sweep, validate_clients
from equireach.synthetic import generate_selection_demo

app = FastAPI(title="EquiReach Mode A API", version="0.1.0")

DEV = os.getenv("DEV", "0") == "1"
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]

if DEV:
    origins = ["*"]
else:
    origins = ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

MAX_CLIENTS = 20000

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "equireach-mode-a",
        "dev": DEV,
        "cors_origins": origins if not DEV else ["*"],
    }


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "tool": "EquiReach",
        "tagline": "Optimizing Resources. Advancing Fairness.",
        "mode": "selection",
        "validated": True,
        "parity_doc": "apps/equireach/docs/PARITY_VALIDATION.md",
        "limits": {
            "max_clients": MAX_CLIENTS,
            "demo_default_n": 80,
            "sweep_default_steps": 12,
        },
        "privacy": "Public demo uses synthetic data only. Restricted survey data is never accepted by this API.",
    }


class ValidateRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(default_factory=list)
    mode: str = "selection"
    start_row: int = 1


@app.post("/api/validate")
def validate_endpoint(req: ValidateRequest) -> dict[str, Any]:
    if len(req.rows) > MAX_CLIENTS:
        raise HTTPException(status_code=413, detail="Too many rows")

    try:
        return validate_clients(req.rows, mode=req.mode, start_row=req.start_row)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Validation failed: {exc}") from exc


class AllocateRequest(BaseModel):
    payload: dict[str, Any]


@app.post("/api/allocate")
def allocate_endpoint(req: AllocateRequest) -> dict[str, Any]:
    payload = req.payload
    params = payload.get("params", {})

    if isinstance(params, dict) and params.get("mode") == "siting":
        raise HTTPException(status_code=501, detail="Mode B siting is not implemented yet.")

    clients = payload.get("clients")
    if isinstance(clients, list) and len(clients) > MAX_CLIENTS:
        raise HTTPException(status_code=413, detail="Too many clients")

    try:
        result = allocate(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Allocation failed: {exc}") from exc

    if isinstance(result, dict):
        summary = result.get("summary")

        if isinstance(summary, dict):
            if "objective" not in result and "total_priority_captured" in summary:
                result["objective"] = summary["total_priority_captured"]

            if "served_count" not in result and "served_count" in summary:
                result["served_count"] = summary["served_count"]

            for key in ("status", "proven_optimal", "solver"):
                if key not in result and key in summary:
                    result[key] = summary[key]

    return result

class SweepRequest(BaseModel):
    payload: dict[str, Any]
    kind: str = "price"
    rule_id: str | None = None
    steps: int = 12
    others: str = "off"
    sweep_time_limit_s: float | None = None


@app.post("/api/sweep")
def sweep_endpoint(req: SweepRequest) -> dict[str, Any]:
    payload = req.payload
    params = payload.get("params", {})

    if isinstance(params, dict) and params.get("mode") == "siting":
        raise HTTPException(status_code=501, detail="Mode B siting is not implemented yet.")

    clients = payload.get("clients")
    if isinstance(clients, list) and len(clients) > MAX_CLIENTS:
        raise HTTPException(status_code=413, detail="Too many clients")

    if req.kind not in {"price", "interaction", "both"}:
        raise HTTPException(status_code=400, detail="kind must be price, interaction, or both")

    if req.steps < 2 or req.steps > 50:
        raise HTTPException(status_code=400, detail="steps must be between 2 and 50")

    try:
        return run_sweep(
            payload,
            kind=req.kind,
            rule_id=req.rule_id,
            steps=req.steps,
            others=req.others,
            sweep_time_limit_s=req.sweep_time_limit_s,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Sweep failed: {exc}") from exc


@app.get("/api/demo")
def demo(kind: str = "selection", n: int = 80, seed: int = 20261018) -> dict[str, Any]:
    if kind != "selection":
        raise HTTPException(
            status_code=501,
            detail="Only selection demo is implemented. Mode B siting is not implemented yet.",
        )

    if n < 10 or n > 5000:
        raise HTTPException(status_code=400, detail="n must be between 10 and 5000")

    return generate_selection_demo(n=n, seed=seed)
