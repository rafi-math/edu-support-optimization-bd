from fastapi.testclient import TestClient

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
    assert body["objective"] is not None

    summary = body["summary"]
    assert summary["status"] == "ok"
    assert summary["proven_optimal"] is True
    assert summary["served_count"] > 0
    assert summary["total_priority_captured"] == body["objective"]


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

    r = client.post(
        "/api/sweep",
        json={"payload": payload, "kind": "price", "rule_id": "girls_30", "steps": 3},
    )
    assert r.status_code == 200
    assert isinstance(r.json(), dict)


def test_siting_demo_not_implemented():
    r = client.get("/api/demo", params={"kind": "siting"})
    assert r.status_code == 501

def test_root_redirects_to_docs():
    r = client.get("/", follow_redirects=False)
    assert r.status_code in (302, 307)
    assert r.headers["location"] == "/docs"