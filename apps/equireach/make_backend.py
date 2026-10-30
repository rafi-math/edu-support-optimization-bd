from pathlib import Path

root = Path(__file__).resolve().parent
api_dir = root / "api"
tests_dir = root / "tests"
docs_dir = root / "docs"

api_dir.mkdir(parents=True, exist_ok=True)
tests_dir.mkdir(parents=True, exist_ok=True)
docs_dir.mkdir(parents=True, exist_ok=True)

(api_dir / "__init__.py").write_text("", encoding="utf-8", newline="\n")

MAIN_PY = """from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
        return allocate(payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Allocation failed: {exc}") from exc


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
"""

(api_dir / "main.py").write_text(MAIN_PY, encoding="utf-8", newline="\n")

TEST_API_PY = """from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_config():
    r = client.get("/api/config")
    assert r.status_code == 200
    body = r.json()
    assert body["mode"] == "selection"
    assert body["validated"] is True


def test_demo_selection_returns_payload():
    r = client.get("/api/demo", params={"kind": "selection", "n": 30, "seed": 1})
    assert r.status_code == 200
    payload = r.json()
    assert isinstance(payload, dict)
    assert "clients" in payload
    assert "params" in payload
    assert len(payload["clients"]) > 0


def test_allocate_demo_is_proven_optimal():
    payload = client.get("/api/demo", params={"kind": "selection", "n": 30, "seed": 1}).json()
    r = client.post("/api/allocate", json={"payload": payload})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["proven_optimal"] is True
    assert body["served_count"] > 0


def test_validate_demo_rows():
    payload = client.get("/api/demo", params={"kind": "selection", "n": 20, "seed": 2}).json()
    r = client.post("/api/validate", json={"rows": payload["clients"]})
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_price_sweep_with_one_rule():
    payload = client.get("/api/demo", params={"kind": "selection", "n": 40, "seed": 3}).json()
    payload["rules"] = [
        {
            "rule_id": "girls_30",
            "attribute": "gender",
            "level": "female",
            "min_share_pct": 30,
        }
    ]

    r = client.post("/api/sweep", json={"payload": payload, "kind": "price", "steps": 3})
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_siting_demo_not_implemented():
    r = client.get("/api/demo", params={"kind": "siting"})
    assert r.status_code == 501
"""

(tests_dir / "test_api.py").write_text(TEST_API_PY, encoding="utf-8", newline="\n")

req = root / "requirements.txt"
lines = [line.strip() for line in req.read_text(encoding="utf-8").splitlines() if line.strip()]

for pkg in ["fastapi", "uvicorn[standard]", "httpx"]:
    if pkg not in lines:
        lines.append(pkg)

req.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")

DOCKERFILE = """FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY equireach ./equireach
COPY api ./api

ENV PORT=7860
EXPOSE 7860

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
"""

(root / "Dockerfile").write_text(DOCKERFILE, encoding="utf-8", newline="\n")

DEPLOY_MD = """# Deploy EquiReach Mode A API to Hugging Face Spaces

1. Create a new Hugging Face Space named `equireach-api`.
2. Choose SDK: Docker, template: Blank, hardware: CPU basic, visibility: Public.
3. In the Space Files tab, keep the Space README front matter and make sure it includes:

```yaml
---
title: EquiReach API
sdk: docker
app_port: 7860
---
```

4. Upload only these files/folders from `apps/equireach` to the Space:
   - `Dockerfile`
   - `requirements.txt`
   - `equireach/`
   - `api/`

5. Do not upload `.venv`, `__pycache__`, `.pytest_cache`, private CSV files, `.sav` files, or anything from the repository root `data/` folder.

6. For testing, add Space variable `DEV=1`. This allows any site to call the API temporarily.

7. After the Space is running, test:
   - `https://YOUR_SPACE_URL/health`
   - `https://YOUR_SPACE_URL/api/demo?kind=selection&n=30&seed=1`

8. Before publishing the dashboard, remove `DEV=1` and set `ALLOWED_ORIGINS` to the exact Lovable site URL, with no trailing slash.
"""

(docs_dir / "DEPLOY_HF_SPACES.md").write_text(DEPLOY_MD, encoding="utf-8", newline="\n")

print("BACKEND_FILES_CREATED")