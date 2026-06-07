from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any


EMITTER_TYPES = (
    "exploit_best_route",
    "behavior_archive_route",
    "failure_repair",
    "underexplored_route",
    "transfer_or_revive",
)

COOLDOWN_AFTER_STALE_GENERATIONS = 5


@dataclass
class RouteCard:
    route_id: str
    route_signature: list[Any]
    best_candidate_id: str
    best_objective: float
    recent_candidates: list[dict[str, Any]] = field(default_factory=list)
    recent_improvement: float | None = None
    failure_signature: str | None = None
    worst_training_instances: list[int] = field(default_factory=list)
    status: str = "active"
    credit: float = 0.0
    trials: int = 0
    successes: int = 0
    max_gap: float | None = None
    last_emitter: str | None = None
    last_seen_generation: int | None = None
    last_improved_generation: int | None = None
    stale_generations: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RouteUpdate:
    candidate_id: str
    route_id: str
    inserted: bool = False
    route_best_improved: bool = False
    failure_repaired: bool = False
    emitter: str | None = None
    success: bool = False


class RouteManager:
    def __init__(self, objective_direction: str = "minimize"):
        self.objective_direction = objective_direction
        self._routes: dict[tuple[Any, ...], RouteCard] = {}
        self._best_candidates: dict[str, Any] = {}
        self._seen_candidate_ids: set[str] = set()
        self.cooldown_after_stale_generations = COOLDOWN_AFTER_STALE_GENERATIONS

    @property
    def coverage(self) -> int:
        return len(self._routes)

    def update(self, candidates: list, generation: int | None = None) -> list[RouteUpdate]:
        current_generation = generation if generation is not None else self._generation_from_candidates(candidates)
        updates = [
            update
            for candidate in candidates
            if (update := self.add(candidate, generation=current_generation)) is not None
        ]
        if current_generation is not None:
            self.refresh_statuses(current_generation)
        return updates

    def add(self, candidate, generation: int | None = None) -> RouteUpdate | None:
        if candidate.objective is None or candidate.id in self._seen_candidate_ids:
            return None
        self._seen_candidate_ids.add(candidate.id)
        descriptor = behavior_descriptor_for_candidate(candidate)
        route = self._routes.get(descriptor)
        inserted = route is None
        repaired = _repairs_parent_failure(candidate)
        emitter = _candidate_emitter(candidate)
        objective = float(candidate.objective)
        candidate_generation = generation if generation is not None else getattr(candidate, "generation", None)
        if route is None:
            route = RouteCard(
                route_id=_route_id(descriptor),
                route_signature=_jsonable_descriptor(descriptor),
                best_candidate_id=candidate.id,
                best_objective=objective,
                last_seen_generation=candidate_generation,
                last_improved_generation=candidate_generation,
            )
            self._routes[descriptor] = route
            route_best_improved = True
        else:
            route_best_improved = self._better_objective(objective, route.best_objective)
            if route_best_improved:
                route.recent_improvement = _objective_improvement(
                    previous=route.best_objective,
                    current=objective,
                    objective_direction=self.objective_direction,
                )
                route.best_candidate_id = candidate.id
                route.best_objective = objective
                route.last_improved_generation = candidate_generation
        if candidate_generation is not None:
            route.last_seen_generation = candidate_generation
        if inserted:
            route.recent_improvement = None
        if route_best_improved:
            self._best_candidates[route.route_id] = candidate
        success = inserted or route_best_improved or repaired
        route.trials += 1
        if success:
            route.successes += 1
        route.credit = route.successes / route.trials
        route.last_emitter = emitter
        self._refresh_behavior_fields(route, candidate)
        route.status = self._status_for(route, is_global_best=False, improved=route_best_improved, repaired=repaired)
        route.recent_candidates.append(
            {
                "candidate_id": candidate.id,
                "objective": objective,
                "generation": getattr(candidate, "generation", None),
                "emitter": emitter,
                "success": success,
            }
        )
        self._mark_global_best_route()
        return RouteUpdate(
            candidate_id=candidate.id,
            route_id=route.route_id,
            inserted=inserted,
            route_best_improved=route_best_improved,
            failure_repaired=repaired,
            emitter=emitter,
            success=success,
        )

    def routes(self) -> list[RouteCard]:
        return sorted(self._routes.values(), key=self._route_sort_key, reverse=self.objective_direction == "maximize")

    def best_candidates(self) -> list:
        return [self._best_candidates[route.route_id] for route in self.routes() if route.route_id in self._best_candidates]

    def route_cards(self) -> list[dict[str, Any]]:
        return [route.to_dict() for route in self.routes()]

    def parent_for_emitter(self, emitter: str, population: list, used_route_ids: set[str] | None = None):
        used_route_ids = used_route_ids or set()
        routes = self.routes()
        if not routes:
            raise RuntimeError("expert route manager has no evaluated routes")
        active_routes = self._non_cooldown_routes(routes) or routes
        population_ids = {candidate.id for candidate in population}
        if emitter == "exploit_best_route":
            route = self._first_route(active_routes, used_route_ids) or active_routes[0]
        elif emitter == "behavior_archive_route":
            route = self._first_route(
                [route for route in active_routes if route.best_candidate_id not in population_ids],
                used_route_ids,
            ) or self._first_route(active_routes, used_route_ids) or active_routes[0]
        elif emitter == "failure_repair":
            repair_routes = sorted(
                active_routes,
                key=lambda route: (
                    float(route.max_gap) if route.max_gap is not None else float("-inf"),
                    self._quality_for_descending(route),
                ),
                reverse=True,
            )
            route = self._first_route(repair_routes, used_route_ids) or repair_routes[0]
        elif emitter == "underexplored_route":
            underexplored = sorted(active_routes, key=lambda route: (route.trials, self._quality_for_ascending(route)))
            route = self._first_route(underexplored, used_route_ids) or underexplored[0]
        elif emitter == "transfer_or_revive":
            revive_routes = [route for route in routes if route.status == "cooldown"]
            transfer_routes = sorted(
                revive_routes or routes,
                key=lambda route: (route.credit, self._quality_for_descending(route)),
                reverse=True,
            )
            route = self._first_route(transfer_routes, used_route_ids) or transfer_routes[0]
        else:
            raise ValueError(f"unknown emitter: {emitter!r}")
        return self._best_candidates[route.route_id], route

    def refresh_statuses(self, generation: int) -> None:
        for route in self._routes.values():
            if route.last_improved_generation is None:
                route.stale_generations = 0
                continue
            route.stale_generations = max(0, int(generation) - int(route.last_improved_generation))
            if route.stale_generations >= self.cooldown_after_stale_generations:
                route.status = "cooldown"
            elif route.status == "cooldown":
                route.status = "active"
        self._mark_global_best_route()

    def _better_objective(self, candidate_objective: float, current_objective: float) -> bool:
        if self.objective_direction == "maximize":
            return candidate_objective > current_objective
        return candidate_objective < current_objective

    def _route_sort_key(self, route: RouteCard) -> float:
        return float(route.best_objective)

    def _quality_for_descending(self, route: RouteCard) -> float:
        objective = float(route.best_objective)
        return objective if self.objective_direction == "maximize" else -objective

    def _quality_for_ascending(self, route: RouteCard) -> float:
        objective = float(route.best_objective)
        return -objective if self.objective_direction == "maximize" else objective

    @staticmethod
    def _first_route(routes: list[RouteCard], used_route_ids: set[str]) -> RouteCard | None:
        for route in routes:
            if route.route_id not in used_route_ids:
                return route
        return None

    @staticmethod
    def _non_cooldown_routes(routes: list[RouteCard]) -> list[RouteCard]:
        return [route for route in routes if route.status != "cooldown"]

    def _refresh_behavior_fields(self, route: RouteCard, candidate) -> None:
        diagnosis = _training_diagnosis(candidate)
        route.failure_signature = diagnosis.get("failure_signature")
        route.worst_training_instances = [
            int(item["index"])
            for item in diagnosis.get("worst_training_instances", []) or []
            if isinstance(item, dict) and item.get("index") is not None
        ]
        max_gap = diagnosis.get("max_gap")
        route.max_gap = float(max_gap) if max_gap is not None else None

    def _status_for(self, route: RouteCard, is_global_best: bool, improved: bool, repaired: bool) -> str:
        if route.status == "cooldown":
            return "cooldown"
        if is_global_best:
            return "exploit"
        if repaired:
            return "repair"
        if improved:
            return "promising"
        return route.status or "active"

    def _mark_global_best_route(self) -> None:
        routes = self.routes()
        if not routes:
            return
        active_routes = self._non_cooldown_routes(routes)
        best_route_id = active_routes[0].route_id if active_routes else None
        for route in routes:
            if route.status == "cooldown":
                continue
            if route.route_id == best_route_id:
                route.status = "exploit"
            elif route.status == "exploit":
                route.status = "active"

    @staticmethod
    def _generation_from_candidates(candidates: list) -> int | None:
        generations = [
            int(candidate.generation)
            for candidate in candidates
            if getattr(candidate, "generation", None) is not None
        ]
        return max(generations) if generations else None


class EmitterScheduler:
    def __init__(self):
        self.stats = {emitter: {"trials": 0, "successes": 0} for emitter in EMITTER_TYPES}

    def record(self, emitter: str | None, success: bool) -> None:
        if emitter not in self.stats:
            return
        stat = self.stats[emitter]
        stat["trials"] += 1
        if success:
            stat["successes"] += 1

    def record_updates(self, updates: list[RouteUpdate]) -> None:
        for update in updates:
            self.record(update.emitter, update.success)

    def select_emitters(self, count: int) -> list[str]:
        if count < 0:
            raise ValueError("count must be non-negative")
        virtual = {key: dict(value) for key, value in self.stats.items()}
        selected = []
        for _ in range(count):
            emitter = self._select_one(virtual)
            selected.append(emitter)
            virtual[emitter]["trials"] += 1
        return selected

    @staticmethod
    def _select_one(stats: dict[str, dict[str, int]]) -> str:
        for emitter in EMITTER_TYPES:
            if stats[emitter]["trials"] == 0:
                return emitter
        total_trials = sum(stat["trials"] for stat in stats.values())
        scores = {}
        for emitter, stat in stats.items():
            average = stat["successes"] / stat["trials"]
            # UCB1 exploration term; no tunable exploration coefficient is introduced.
            scores[emitter] = average + math.sqrt(2.0 * math.log(total_trials) / stat["trials"])
        return sorted(scores, key=lambda emitter: (-scores[emitter], emitter))[0]

    def to_dict(self) -> dict[str, Any]:
        return {emitter: dict(values) for emitter, values in self.stats.items()}


def behavior_descriptor_for_candidate(candidate) -> tuple[Any, ...]:
    metadata = getattr(candidate, "metadata", None) or {}
    arch_spec = metadata.get("arch_spec") if isinstance(metadata, dict) else None
    diagnosis = _training_diagnosis(candidate)
    worst_indices = tuple(
        int(item["index"])
        for item in diagnosis.get("worst_training_instances", []) or []
        if isinstance(item, dict) and item.get("index") is not None
    )
    if isinstance(arch_spec, dict):
        distance = (arch_spec.get("distance_update_policy") or {}).get("type")
        penalty = (arch_spec.get("penalty_policy") or {}).get("edge_selection")
        top_k = (arch_spec.get("penalty_policy") or {}).get("top_k")
        escape = (arch_spec.get("escape_policy") or {}).get("trigger")
        restart = (arch_spec.get("restart_policy") or {}).get("action")
        schedule = (arch_spec.get("schedule_policy") or {}).get("strength_schedule")
        return (
            ("workflow", "arch_spec_behavior"),
            ("distance", str(distance)),
            ("penalty", str(penalty)),
            ("top_k", str(top_k)),
            ("escape", str(escape)),
            ("restart", str(restart)),
            ("schedule", str(schedule)),
            ("failure", str(diagnosis.get("failure_signature") or "unknown")),
            ("worst_instances", worst_indices),
        )
    hypothesis = getattr(candidate, "mechanism_hypothesis", None) or {}
    family = hypothesis.get("mechanism_family") if isinstance(hypothesis, dict) else None
    return (
        ("workflow", "mechanism_behavior"),
        ("family", str(family or "unspecified")),
        ("failure", str(diagnosis.get("failure_signature") or "unknown")),
        ("worst_instances", worst_indices),
    )


def _training_diagnosis(candidate) -> dict[str, Any]:
    evidence = getattr(candidate, "evidence", None) or {}
    diagnosis = evidence.get("training_diagnosis") if isinstance(evidence, dict) else None
    if isinstance(diagnosis, dict):
        return diagnosis
    metadata = getattr(candidate, "metadata", None) or {}
    diagnosis = metadata.get("training_diagnosis") if isinstance(metadata, dict) else None
    return diagnosis if isinstance(diagnosis, dict) else {}


def _repairs_parent_failure(candidate) -> bool:
    diagnosis = _training_diagnosis(candidate)
    if not diagnosis.get("improved_instances_vs_parent"):
        return False
    mean_delta = diagnosis.get("mean_delta_vs_parent")
    if mean_delta is None:
        return False
    return float(mean_delta) <= 0.0


def _candidate_emitter(candidate) -> str | None:
    metadata = getattr(candidate, "metadata", None) or {}
    plan = metadata.get("generation_plan") if isinstance(metadata, dict) else None
    if isinstance(plan, dict):
        emitter = plan.get("emitter")
        return str(emitter) if emitter else None
    return None


def _objective_improvement(previous: float, current: float, objective_direction: str) -> float:
    if objective_direction == "maximize":
        return current - previous
    return previous - current


def _route_id(descriptor: tuple[Any, ...]) -> str:
    payload = json.dumps(_jsonable_descriptor(descriptor), sort_keys=True)
    return "route_" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _jsonable_descriptor(descriptor: tuple[Any, ...]) -> list[Any]:
    return [
        [item[0], list(item[1]) if isinstance(item[1], tuple) else item[1]]
        if isinstance(item, tuple) and len(item) == 2
        else item
        for item in descriptor
    ]
