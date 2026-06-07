from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArchiveUpdate:
    descriptor: tuple[Any, ...]
    inserted: bool = False
    replaced: bool = False
    previous_id: str | None = None


class QDArchive:
    def __init__(self, objective_direction: str = "minimize", descriptor_mode: str = "full"):
        self.objective_direction = objective_direction
        self.descriptor_mode = descriptor_mode
        self._cells: dict[tuple[Any, ...], Any] = {}

    @property
    def coverage(self) -> int:
        return len(self._cells)

    def add(self, candidate) -> ArchiveUpdate:
        if candidate.objective is None:
            raise ValueError(f"candidate {candidate.id} has no objective")
        descriptor = descriptor_for_candidate(candidate, mode=self.descriptor_mode)
        current = self._cells.get(descriptor)
        if current is None:
            self._cells[descriptor] = candidate
            return ArchiveUpdate(descriptor=descriptor, inserted=True)
        if self._better(candidate, current):
            self._cells[descriptor] = candidate
            return ArchiveUpdate(descriptor=descriptor, replaced=True, previous_id=current.id)
        return ArchiveUpdate(descriptor=descriptor)

    def update(self, candidates: list) -> list[ArchiveUpdate]:
        return [self.add(candidate) for candidate in candidates if candidate.objective is not None]

    def elites(self) -> list:
        return sorted(self._cells.values(), key=self._sort_key, reverse=self.objective_direction == "maximize")

    def select_population(self, pop_size: int) -> list:
        ranked_cells = self.elites()
        selected = ranked_cells[: max(0, int(pop_size))]
        return selected

    def cells(self) -> dict[tuple[Any, ...], Any]:
        return dict(self._cells)

    def _better(self, candidate, existing) -> bool:
        if self.objective_direction == "maximize":
            return float(candidate.objective) > float(existing.objective)
        return float(candidate.objective) < float(existing.objective)

    @staticmethod
    def _sort_key(candidate) -> float:
        return float(candidate.objective)


def descriptor_for_candidate(candidate, mode: str = "full") -> tuple[Any, ...]:
    metadata = getattr(candidate, "metadata", None) or {}
    arch_spec = metadata.get("arch_spec") if isinstance(metadata, dict) else None
    solver_tree = metadata.get("solver_tree") if isinstance(metadata, dict) else None
    if isinstance(solver_tree, dict):
        from eohagent.problems.tsp_gls_solver_tree.tree_schema import solver_tree_descriptor

        return solver_tree_descriptor(solver_tree)
    if isinstance(arch_spec, dict):
        from eohagent.problems.tsp_gls_arch_spec.spec_schema import arch_spec_niche_key, canonical_arch_spec

        if mode == "mechanism":
            canonical = canonical_arch_spec(arch_spec)
            return (
                "arch_spec_mechanism",
                canonical["distance_update_policy"]["type"],
                canonical["penalty_policy"]["edge_selection"],
                str(canonical["penalty_policy"]["top_k"]),
                canonical["schedule_policy"]["strength_schedule"],
            )
        if mode != "full":
            raise ValueError(f"unsupported QD descriptor mode: {mode!r}")
        return ("arch_spec",) + arch_spec_niche_key(arch_spec)
    if mode not in {"full", "mechanism"}:
        raise ValueError(f"unsupported QD descriptor mode: {mode!r}")
    hypothesis = getattr(candidate, "mechanism_hypothesis", None) or {}
    if isinstance(hypothesis, dict) and hypothesis.get("mechanism_family"):
        return ("family", str(hypothesis["mechanism_family"]))
    profile = getattr(candidate, "mechanism_profile", None) or {}
    if isinstance(profile, dict) and profile.get("mechanism_family"):
        return ("family", str(profile["mechanism_family"]))
    return ("unspecified",)
