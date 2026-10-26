from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from equireach.allocate import allocate
from equireach.constants import ENGINE_VERSION, TAGLINE, TOOL_NAME
from equireach.errors import EquiReachError
from equireach.sweeps import run_sweep
from equireach.validation import load_clients_csv


def _parse_rule(text: str, index: int, count: bool) -> dict:
    parts = text.split(":")
    if len(parts) != 3:
        raise ValueError(
            f"rule must be attribute:level:value, got {text!r}"
        )
    attribute, level, raw = parts
    rule = {
        "rule_id": f"r{index + 1}",
        "attribute": attribute,
        "level": level,
        "max_share_pct": None,
    }
    if count:
        rule["min_count"] = int(raw)
        rule["min_share_pct"] = None
    else:
        rule["min_share_pct"] = float(raw)
        rule["min_count"] = None
    return rule


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="equireach",
        description=f"{TOOL_NAME}: {TAGLINE}",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="Mode A client selection")
    select.add_argument("--clients", required=True, help="CSV with client_id and priority_score")
    select.add_argument("--k", type=float, required=True, help="coverage count K")
    select.add_argument("--rule", action="append", default=[], help="attribute:level:percent")
    select.add_argument(
        "--rule-count",
        action="append",
        default=[],
        help="attribute:level:N exact count floor (exactly_K only)",
    )
    select.add_argument(
        "--coverage-mode",
        choices=["exactly_K", "at_most_K"],
        default="exactly_K",
    )
    select.add_argument(
        "--weighting",
        choices=["unweighted", "weighted"],
        default="unweighted",
    )
    select.add_argument("--time-limit", type=float, default=20.0)
    select.add_argument("--heuristic", action="store_true", help="force the greedy_swap fallback")
    select.add_argument("--compare-baseline", action="store_true")
    select.add_argument("--source", default="imported")

    sweep = sub.add_parser("sweep", help="price-of-fairness or interaction sweep")
    sweep.add_argument("--clients", required=True)
    sweep.add_argument("--k", type=float, required=True)
    sweep.add_argument("--rule", action="append", default=[])
    sweep.add_argument("--rule-count", action="append", default=[])
    sweep.add_argument("--kind", choices=["price", "interaction", "both"], default="price")
    sweep.add_argument("--rule-id", default="r1")
    sweep.add_argument("--steps", type=int, default=12)
    sweep.add_argument("--others", choices=["off", "current"], default="off")
    sweep.add_argument("--weighting", choices=["unweighted", "weighted"], default="unweighted")
    sweep.add_argument("--time-limit", type=float, default=20.0)
    return parser


def _scenario_from_args(args: argparse.Namespace) -> dict:
    loaded = load_clients_csv(args.clients)
    if loaded["clients"]["errors"]:
        raise EquiReachError(
            "INVALID_INPUT",
            "one or more client rows are invalid",
            details=loaded["clients"]["errors"],
        )
    clients = [c.to_record() for c in loaded["parsed_clients"]]
    rules = []
    for i, text in enumerate(args.rule):
        rules.append(_parse_rule(text, i, count=False))
    offset = len(rules)
    for i, text in enumerate(args.rule_count):
        rules.append(_parse_rule(text, offset + i, count=True))
        if getattr(args, "coverage_mode", "exactly_K") != "exactly_K":
            raise EquiReachError(
                "INVALID_INPUT",
                "--rule-count is valid with exactly_K only",
            )
    payload = {
        "params": {
            "mode": "selection",
            "coverage_mode": getattr(args, "coverage_mode", "exactly_K"),
            "coverage_target": {"type": "count", "value": args.k},
            "objective_weighting": args.weighting,
            "exact": not getattr(args, "heuristic", False),
            "time_limit_s": args.time_limit,
            "source": getattr(args, "source", "imported"),
            "seed": None,
        },
        "clients": clients,
        "centers": [],
        "rules": rules,
        "flags": {"compare_baseline": bool(getattr(args, "compare_baseline", False))},
    }
    return payload


def _print(data: dict) -> None:
    print(json.dumps(data, indent=2, sort_keys=False))


def _cli_select_view(result: dict) -> dict:
    summary = result["summary"]
    return {
        "tool": TOOL_NAME,
        "tagline": TAGLINE,
        "engine_version": ENGINE_VERSION,
        "objective": round(float(summary["total_priority_captured"]), 6),
        "served_count": summary["served_count"],
        "achieved_shares": summary["equity_compliance"],
        "status": result["status"],
        "proven_optimal": result["proven_optimal"],
        "solver": result["solver"],
        "mip_gap": result["mip_gap"],
        "summary": summary,
        "assignments": result["assignments"],
        "unserved": result["unserved"],
        "warnings": result["warnings"],
        "baseline_comparison": result.get("baseline_comparison"),
        "provenance": result["provenance"],
        "inputs_echo": result["inputs_echo"],
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        payload = _scenario_from_args(args)
        if args.command == "select":
            result = allocate(payload)
            _print(_cli_select_view(result))
            return 0
        result = run_sweep(
            payload,
            kind=args.kind,
            rule_id=args.rule_id,
            steps=args.steps,
            others=args.others,
        )
        _print(result)
        return 0
    except EquiReachError as exc:
        _print(exc.to_dict())
        return 2
    except ValueError as exc:
        _print({"error": {"code": "INVALID_INPUT", "message": str(exc), "details": []}})
        return 2


if __name__ == "__main__":
    sys.exit(main())
