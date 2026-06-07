import json
from pathlib import Path

from .candidate import Candidate
from .config import resolve_path


class ResultStore:
    def __init__(self, config: dict):
        self.output_dir = resolve_path(config, config["project"]["output_dir"])
        self.pops_dir = self.output_dir / "results" / "pops"
        self.best_dir = self.output_dir / "results" / "pops_best"
        self.workflow_dir = self.output_dir / "results" / "workflow_audit"
        self.cost_dir = self.output_dir / "results" / "cost_audit"
        self.qd_archive_dir = self.output_dir / "results" / "qd_archive"
        self.routes_dir = self.output_dir / "results" / "routes"
        self.pops_dir.mkdir(parents=True, exist_ok=True)
        self.best_dir.mkdir(parents=True, exist_ok=True)
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        self.cost_dir.mkdir(parents=True, exist_ok=True)
        self.qd_archive_dir.mkdir(parents=True, exist_ok=True)
        self.routes_dir.mkdir(parents=True, exist_ok=True)

    def save_population(self, generation: int, population: list[Candidate]) -> None:
        path = self.pops_dir / f"population_generation_{generation}.json"
        path.write_text(json.dumps([c.to_dict() for c in population], indent=2), encoding="utf-8")
        if population:
            best_path = self.best_dir / f"population_generation_{generation}.json"
            best_path.write_text(json.dumps(population[0].to_dict(), indent=2), encoding="utf-8")

    def save_workflow_audit(self, generation: int, audit: dict) -> None:
        path = self.workflow_dir / f"workflow_generation_{generation}.json"
        path.write_text(json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")

    def save_cost_audit(self, generation: int, audit: dict) -> None:
        path = self.cost_dir / f"cost_generation_{generation}.json"
        path.write_text(json.dumps(audit, indent=2, ensure_ascii=True), encoding="utf-8")

    def save_qd_archive(self, generation: int, archive) -> None:
        cells = []
        for descriptor, candidate in archive.cells().items():
            cells.append(
                {
                    "descriptor": list(descriptor),
                    "candidate": candidate.to_dict(),
                }
            )
        cells.sort(key=lambda item: item["candidate"].get("objective"))
        path = self.qd_archive_dir / f"qd_archive_generation_{generation}.json"
        path.write_text(
            json.dumps({"coverage": archive.coverage, "cells": cells}, indent=2),
            encoding="utf-8",
        )

    def save_route_state(self, generation: int, route_manager) -> None:
        path = self.routes_dir / f"route_state_generation_{generation}.json"
        path.write_text(
            json.dumps(
                {
                    "coverage": route_manager.coverage,
                    "routes": route_manager.route_cards(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
