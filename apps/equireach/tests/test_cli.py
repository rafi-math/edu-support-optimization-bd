import json
from pathlib import Path

from equireach.allocate import allocate
from equireach.cli import main


def test_cli_matches_allocate(tmp_path, capsys):
    csv_path = tmp_path / "clients.csv"
    csv_path.write_text(
        "client_id,priority_score,weight,need_flag,age_band,gender\n"
        "c1,0.9,1,1,13-17,female\n"
        "c2,0.8,1,1,13-17,male\n"
        "c3,0.7,1,0,13-17,male\n"
        "c4,0.2,1,0,6-12,female\n"
        "c5,0.1,1,0,6-12,male\n",
        encoding="utf-8",
    )
    code = main(
        [
            "select",
            "--clients",
            str(csv_path),
            "--k",
            "3",
            "--rule",
            "age_band:6-12:33.3",
        ]
    )
    assert code == 0
    cli = json.loads(capsys.readouterr().out)
    payload = {
        "params": {
            "mode": "selection",
            "coverage_mode": "exactly_K",
            "coverage_target": {"type": "count", "value": 3},
            "objective_weighting": "unweighted",
            "exact": True,
            "time_limit_s": 20,
            "source": "imported",
        },
        "clients": [
            {
                "client_id": "c1",
                "priority_score": 0.9,
                "weight": 1,
                "need_flag": 1,
                "attributes": {"age_band": "13-17", "gender": "female"},
            },
            {
                "client_id": "c2",
                "priority_score": 0.8,
                "weight": 1,
                "need_flag": 1,
                "attributes": {"age_band": "13-17", "gender": "male"},
            },
            {
                "client_id": "c3",
                "priority_score": 0.7,
                "weight": 1,
                "need_flag": 0,
                "attributes": {"age_band": "13-17", "gender": "male"},
            },
            {
                "client_id": "c4",
                "priority_score": 0.2,
                "weight": 1,
                "need_flag": 0,
                "attributes": {"age_band": "6-12", "gender": "female"},
            },
            {
                "client_id": "c5",
                "priority_score": 0.1,
                "weight": 1,
                "need_flag": 0,
                "attributes": {"age_band": "6-12", "gender": "male"},
            },
        ],
        "rules": [
            {
                "rule_id": "r1",
                "attribute": "age_band",
                "level": "6-12",
                "min_share_pct": 33.3,
            }
        ],
    }
    api = allocate(payload)
    assert cli["served_count"] == api["summary"]["served_count"]
    assert cli["status"] == api["status"]
    assert cli["proven_optimal"] == api["proven_optimal"]
    assert abs(cli["objective"] - round(api["summary"]["total_priority_captured"], 6)) < 1e-12
    assert {row["client_id"] for row in cli["assignments"]} == {
        row["client_id"] for row in api["assignments"]
    }


def test_engine_does_not_write_user_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    before = set(Path(".").rglob("*"))
    allocate(
        {
            "params": {
                "mode": "selection",
                "coverage_mode": "exactly_K",
                "coverage_target": {"type": "count", "value": 1},
                "objective_weighting": "unweighted",
                "exact": True,
                "time_limit_s": 5,
                "source": "synthetic",
            },
            "clients": [
                {
                    "client_id": "c1",
                    "priority_score": 0.4,
                    "weight": 1,
                    "attributes": {"g": "a"},
                },
                {
                    "client_id": "c2",
                    "priority_score": 0.3,
                    "weight": 1,
                    "attributes": {"g": "b"},
                },
            ],
            "rules": [],
        }
    )
    after = set(Path(".").rglob("*"))
    assert after == before
