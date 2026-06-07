from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentState:
    parent: Any
    population_summary: list[dict]
    compact_context: dict
    notebook: dict
    mechanism_archive: dict = field(default_factory=dict)
    step_count: int = 0
    current_candidate: Any = None
    contract_passed: bool = False
    contract_checked: bool = False
    probe_checked: bool = False
    probe_penalty: float | None = None
    probe_results: list[dict] = field(default_factory=list)
    repair_count: int = 0
    last_result: dict | None = None
    history: list[dict] = field(default_factory=list)

    def update(self, result: dict) -> None:
        self.step_count += 1
        serializable_result = dict(result)
        if "candidate" in serializable_result:
            serializable_result["candidate"] = serializable_result["candidate"].to_dict()
        self.last_result = serializable_result
        self.history.append(serializable_result)
        if "candidate" in result:
            self.current_candidate = result["candidate"]
            self.contract_checked = False
            self.contract_passed = False
            self.probe_checked = False
            self.probe_penalty = None
            self.probe_results = []
        if "contract_passed" in result:
            self.contract_checked = True
            self.contract_passed = bool(result["contract_passed"])
            if self.current_candidate is not None:
                self.current_candidate.contract_passed = self.contract_passed
        if result.get("event") == "probe_synthesis":
            self.probe_checked = True
            self.probe_results = result.get("probe_results") or result.get("probes") or []
            self.probe_penalty = result.get("probe_penalty")
            if self.current_candidate is not None:
                self.current_candidate.probe_results = self.probe_results
                self.current_candidate.metadata["probe_penalty"] = self.probe_penalty
                self.current_candidate.metadata["probe_evidence"] = result.get("probe_evidence", {})
        if result.get("event") in {"repair", "repair_failed"}:
            self.repair_count += 1

    def best_candidate(self):
        return self.current_candidate
