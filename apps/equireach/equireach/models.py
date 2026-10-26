from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class Client:
    client_id: str
    priority_score: float
    weight: float = 1.0
    label: Optional[str] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    need_flag: Optional[int] = None
    source_row: int = 1

    def to_record(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "label": self.label,
            "lat": self.lat,
            "lng": self.lng,
            "priority_score": self.priority_score,
            "weight": self.weight,
            "attributes": dict(self.attributes),
            "need_flag": self.need_flag,
        }


@dataclass
class Rule:
    rule_id: str
    attribute: str
    level: str
    min_share_pct: Optional[float] = None
    max_share_pct: Optional[float] = None
    min_count: Optional[int] = None

    def to_record(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "attribute": self.attribute,
            "level": self.level,
            "min_share_pct": self.min_share_pct,
            "max_share_pct": self.max_share_pct,
            "min_count": self.min_count,
        }


@dataclass
class Params:
    mode: str = "selection"
    coverage_mode: str = "exactly_K"
    coverage_target: Any = None
    objective_weighting: str = "unweighted"
    exact: bool = True
    time_limit_s: float = 20.0
    source: str = "imported"
    seed: Optional[int] = None
    sweep_time_limit_s: float = 55.0
    exact_client_limit: int = 20_000

    def to_record(self, resolved_k: Optional[int] = None) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "coverage_mode": self.coverage_mode,
            "coverage_target": self.coverage_target,
            "resolved_k": resolved_k,
            "objective_weighting": self.objective_weighting,
            "exact": self.exact,
            "time_limit_s": self.time_limit_s,
            "source": self.source,
            "seed": self.seed,
            "max_travel_km": None,
            "center_mode": None,
            "n_centers": None,
            "max_centers": None,
            "budget": None,
        }


@dataclass
class Flags:
    compare_baseline: bool = False


@dataclass
class Scenario:
    params: Params
    clients: list[Client]
    rules: list[Rule] = field(default_factory=list)
    centers: list[dict] = field(default_factory=list)
    flags: Flags = field(default_factory=Flags)


@dataclass
class SolveResult:
    selected: list[bool]
    status: str
    solver: str
    proven_optimal: bool
    mip_gap: Optional[float]
    reasons: list[str]
    warnings: list[dict]
    unmet_rule_id: Optional[str] = None
