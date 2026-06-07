from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Candidate:
    id: str
    parents: list[str]
    algorithm: str
    code: str
    objective: float | None = None
    selection_score: float | None = None
    mechanism_summary: str | None = None
    mechanism_hypothesis: dict[str, Any] | None = None
    mechanism_profile: dict[str, Any] | None = None
    parent_delta: str | None = None
    probe_results: list[dict[str, Any]] | None = None
    evidence: dict[str, Any] | None = None
    hypothesis_verdict: str | None = None
    contract_passed: bool = False
    status: str = "candidate"
    generation: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: data.get(k) for k in allowed if k in data})
