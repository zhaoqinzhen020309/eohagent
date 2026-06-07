from __future__ import annotations

import ast
import concurrent.futures
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from eohagent.bootstrap.expert_bootstrap import ExpertBootstrap
from eohagent.core.candidate import Candidate
from eohagent.core.config import load_config, resolve_path
from eohagent.core.evaluator_bridge import build_evaluator
from eohagent.core.experiment_guard import prepare_output_dir
from eohagent.core.population import PopulationManager
from eohagent.core.qd_archive import QDArchive, descriptor_for_candidate
from eohagent.core.result_store import ResultStore
from eohagent.core.route_manager import EmitterScheduler, RouteManager
from eohagent.core.selection import CellRoundRobinSelector, RankSelector
from eohagent.core.training_diagnostics import build_training_diagnosis
from eohagent.harness.agent_loop import AgentHarness
from eohagent.harness.context_runtime import ContextRuntime
from eohagent.harness.experiment_review import ExperimentReviewer
from eohagent.harness.generation_planner import GenerationPlanner
from eohagent.llm.client import build_llm_client
from eohagent.llm.json_call import JSONCall
from eohagent.memory.memory_manager import MemoryManager
from eohagent.problems.tsp_gls_arch_spec.spec_schema import (
    ARCH_SPEC_LAYERS,
    arch_spec_from_candidate,
    build_arch_spec_code,
    canonical_arch_spec,
    describe_arch_spec,
    mechanism_family_from_spec,
    validate_single_layer_change,
)
from eohagent.problems.tsp_gls_solver_tree.tree_schema import (
    build_solver_tree_code,
    canonical_solver_tree,
    changed_solver_tree_paths,
    classify_solver_tree_mutation,
    describe_solver_tree,
    solver_tree_descriptor,
    solver_tree_from_candidate,
)


class ExpertEoHEngine:
    def __init__(self, config: dict):
        config["_repo_root_path"] = Path(config["_repo_root"])
        self.config = config
        self._validate_workflow_contract()
        self.run_manifest = prepare_output_dir(config)
        self.evaluator = build_evaluator(config)
        self.memory = MemoryManager(config)
        self.disabled_patterns = []
        self.memory.disabled_patterns = self.disabled_patterns
        self.experiment_reviewer = ExperimentReviewer(config)
        self.result_store = ResultStore(config)
        self.progress_path = resolve_path(config, config["project"]["output_dir"]) / "progress.jsonl"
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self.objective_direction = config["evaluator"].get("objective_direction", "minimize")
        self.n_proc = max(1, int(config["evolution"].get("n_proc", config["evolution"].get("exp_n_proc", 1))))
        self.eval_timeout = int(config["evaluator"].get("timeout", 300))
        survival_selection = config["evolution"].get("survival_selection", "greedy")
        self.route_manager = RouteManager(self.objective_direction) if survival_selection == "expert_routes" else None
        self.emitter_scheduler = EmitterScheduler() if self.route_manager is not None else None
        qd_descriptor_mode = "mechanism" if survival_selection == "qd_cells" else "full"
        self.qd_archive = (
            QDArchive(self.objective_direction, descriptor_mode=qd_descriptor_mode)
            if survival_selection in {"qd_archive", "expert_routes", "qd_cells"}
            else None
        )
        self.selector = CellRoundRobinSelector(self.objective_direction) if self.qd_archive is not None or self.route_manager is not None else RankSelector(self.objective_direction)
        self.population_manager = PopulationManager(
            pop_size=int(config["evolution"]["pop_size"]),
            objective_direction=self.objective_direction,
            deduplicate_code=bool(config["evolution"].get("deduplicate_code", True)),
            deduplicate_objective=bool(config["evolution"].get("deduplicate_objective", True)),
            use_selection_score=bool(config.get("selection", {}).get("use_selection_score", True)),
            diversity_weight=float(config.get("selection", {}).get("diversity_weight", 0.0)),
            survival_selection=survival_selection,
        )
        llm_client = build_llm_client(config)
        self.context_runtime = ContextRuntime(config, self.memory, llm_client)
        self.json_call = JSONCall(
            llm_client,
            max_retries=int(config["llm"].get("max_retries", 3)),
            context_runtime=self.context_runtime,
        )
        self.harness = AgentHarness(config, self.memory, self.json_call, self.evaluator, progress=self._progress)
        self.generation_planner = GenerationPlanner(config, self.memory, self.json_call, self.selector)

    @classmethod
    def from_config_path(cls, config_path: str):
        return cls(load_config(config_path))

    def _validate_workflow_contract(self) -> None:
        if not self._requires_expert_workflow():
            return
        if not bool(self.config.get("agent", {}).get("enable_generation_planning", True)):
            raise RuntimeError("TSP-GLS workflow requires generation planning; dynamic offspring mode is not allowed")
        if not bool(self.config.get("agent", {}).get("enable_generation_reflection", True)):
            raise RuntimeError("TSP-GLS workflow requires generation reflection; no deterministic replacement is allowed")
        problem = self.config.get("project", {}).get("problem")
        evaluator_problem = self.config.get("evaluator", {}).get("problem")
        if problem == "tsp_gls" or evaluator_problem == "tsp_gls":
            if self.config.get("evaluator", {}).get("function_name") != "update_edge_distance":
                raise RuntimeError("TSP-GLS workflow requires update_edge_distance as the evaluated function")
        if problem == "tsp_gls_arch" or evaluator_problem == "tsp_gls_arch":
            class_name = self.config.get("evaluator", {}).get("architecture_class_name")
            if class_name != "CandidateGLSArchitecture":
                raise RuntimeError("TSP-GLS architecture workflow requires CandidateGLSArchitecture")
            layers = set(self.config.get("architecture", {}).get("open_layers", []))
            required_layers = {
                "distance_update",
                "penalty_edge_selection",
                "perturbation_schedule",
                "restart_policy",
            }
            if layers != required_layers:
                raise RuntimeError("TSP-GLS architecture workflow requires the four fixed GLS architecture layers")
        if problem == "tsp_gls_arch_spec" or evaluator_problem == "tsp_gls_arch_spec":
            class_name = self.config.get("evaluator", {}).get("architecture_class_name")
            if class_name != "CandidateGLSArchitecture":
                raise RuntimeError("TSP-GLS ARCH_SPEC workflow requires CandidateGLSArchitecture")
            layers = tuple(self.config.get("architecture", {}).get("open_layers", []))
            if layers != ARCH_SPEC_LAYERS:
                raise RuntimeError("TSP-GLS ARCH_SPEC workflow requires the fixed auditable ARCH_SPEC layers")
            if self.config["evolution"].get("survival_selection") not in {"quality_diversity", "qd_archive", "expert_routes", "qd_cells"}:
                raise RuntimeError("TSP-GLS ARCH_SPEC workflow requires quality_diversity, qd_archive, qd_cells, or expert_routes survival selection")
        if problem == "tsp_gls_solver_tree" or evaluator_problem == "tsp_gls_solver_tree":
            class_name = self.config.get("evaluator", {}).get("architecture_class_name")
            if class_name != "CandidateGLSArchitecture":
                raise RuntimeError("TSP-GLS solver-tree workflow requires CandidateGLSArchitecture")
            if self.config["evolution"].get("survival_selection") not in {"qd_archive", "qd_cells"}:
                raise RuntimeError("TSP-GLS solver-tree workflow requires qd_archive or qd_cells survival selection")

    def _requires_expert_workflow(self) -> bool:
        return (
            self.config.get("project", {}).get("problem") in {"tsp_gls", "tsp_gls_arch", "tsp_gls_arch_spec", "tsp_gls_solver_tree"}
            or self.config.get("evaluator", {}).get("problem") in {"tsp_gls", "tsp_gls_arch", "tsp_gls_arch_spec", "tsp_gls_solver_tree"}
        )

    def smoke(self) -> list:
        self._progress(
            "bootstrap_start",
            problem=self.config["project"]["problem"],
            evaluator=self.config["evaluator"].get("problem"),
            output_dir=self.config["project"]["output_dir"],
        )
        population = ExpertBootstrap(self.config, self.evaluator, self.memory, progress=self._progress).initialize_population()
        population = self._update_population([], population, generation=0)
        self.result_store.save_population(0, population)
        self._save_qd_archive(0)
        self._save_route_state(0)
        self.memory.compact_if_needed(population)
        self._progress(
            "bootstrap_done",
            population_size=len(population),
            best_objective=population[0].objective if population else None,
            seeds=[{"id": c.id, "objective": c.objective} for c in population],
        )
        return population

    def run(self) -> list:
        self._progress(
            "run_start",
            problem=self.config["project"]["problem"],
            generations=int(self.config["evolution"]["generations"]),
            pop_size=int(self.config["evolution"]["pop_size"]),
            offspring_per_generation=int(self.config["evolution"]["offspring_per_generation"]),
            n_proc=self.n_proc,
        )
        population = self.smoke()
        generations = int(self.config["evolution"]["generations"])
        offspring_per_generation = int(self.config["evolution"]["offspring_per_generation"])
        for generation in range(1, generations + 1):
            generation_start = time.time()
            self._progress(
                "generation_start",
                generation=generation,
                generations=generations,
                population_size=len(population),
                best_objective=population[0].objective if population else None,
            )
            slot_metadata = None
            if self._uses_qd_cells():
                parent_pool, slot_metadata = self._build_qd_cell_parent_plan(offspring_per_generation)
            elif self.route_manager is not None:
                parent_pool, slot_metadata = self._build_route_parent_plan(population, offspring_per_generation)
            else:
                parent_pool = self._build_parent_pool(population)
            if hasattr(self.selector, "reset"):
                self.selector.reset()
            workflow_audit = self._new_workflow_audit(
                generation,
                generations,
                offspring_per_generation,
                population,
            )
            try:
                generation_plan = self.generation_planner.plan(
                    generation,
                    population,
                    offspring_per_generation,
                    parent_pool,
                    slot_metadata=slot_metadata,
                )
                workflow_audit["planner"] = self._planner_audit(generation_plan, offspring_per_generation)
            except Exception as exc:
                workflow_audit["planner"] = {
                    "success": False,
                    "error": str(exc),
                    "plan_count": 0,
                    "complete": False,
                }
                self.memory.session_log.append(
                    {"event": "generation_failed", "generation": generation, "error": str(exc)}
                )
                self._progress(
                    "generation_failed",
                    generation=generation,
                    generations=generations,
                    reason=str(exc),
                    elapsed_min=round((time.time() - generation_start) / 60, 2),
                )
                self.result_store.save_population(generation, population)
                self._save_qd_archive(generation)
                self._save_route_state(generation)
                self.result_store.save_workflow_audit(generation, workflow_audit)
                self._save_cost_audit(generation)
                break
            self.memory.session_log.append(
                {
                    "event": "generation_plan",
                    "generation": generation,
                    "generation_goal": generation_plan.get("generation_goal"),
                    "avoid_repeating": generation_plan.get("avoid_repeating", []),
                }
            )
            self._progress(
                "generation_plan",
                generation=generation,
                goal=generation_plan.get("generation_goal"),
                plan_count=len(generation_plan.get("offspring_plans", [])),
            )
            tasks = []
            for offspring_index, offspring_plan in enumerate(generation_plan["offspring_plans"], start=1):
                if slot_metadata and offspring_index <= len(slot_metadata):
                    offspring_plan = {**offspring_plan, **slot_metadata[offspring_index - 1]}
                parent = self._parent_for_plan(parent_pool, offspring_plan)
                offspring_plan = {**offspring_plan, "parent_id": parent.id}
                self._progress(
                    "offspring_start",
                    generation=generation,
                    offspring_index=offspring_index,
                    offspring_total=offspring_per_generation,
                    parent_id=parent.id,
                    parent_objective=parent.objective,
                )
                tasks.append((generation, offspring_index, offspring_per_generation, parent, offspring_plan))
            results = self._generate_offspring_batch(population, tasks)
            self._record_emitter_failures(results)
            workflow_audit["offspring"] = [
                self._offspring_audit(result) for result in results
            ]
            offspring = self._merge_offspring_batch(population, results)
            population = self._update_population(population, offspring, generation=generation)
            try:
                if self.config.get("agent", {}).get("enable_generation_reflection", True):
                    reflection = self.generation_planner.reflect(generation, population, offspring)
                    workflow_audit["reflection"] = {
                        "success": True,
                        "validated_count": len(reflection.get("validated_principles") or []),
                        "failed_count": len(reflection.get("failed_hypotheses") or []),
                        "open_question_count": len(reflection.get("open_questions") or []),
                    }
                else:
                    workflow_audit["reflection"] = {
                        "success": False,
                        "error": "generation reflection disabled; no deterministic replacement is allowed",
                    }
                    raise RuntimeError("generation reflection disabled; no deterministic replacement is allowed")
            except Exception as exc:
                if not workflow_audit.get("reflection"):
                    workflow_audit["reflection"] = {"success": False, "error": str(exc)}
                self.memory.session_log.append(
                    {"event": "generation_failed", "generation": generation, "error": str(exc)}
                )
                self._progress(
                    "generation_failed",
                    generation=generation,
                    generations=generations,
                    reason=str(exc),
                    elapsed_min=round((time.time() - generation_start) / 60, 2),
                )
                self.result_store.save_population(generation, population)
                self._save_qd_archive(generation)
                self._save_route_state(generation)
                self.result_store.save_workflow_audit(generation, workflow_audit)
                self._save_cost_audit(generation)
                break
            self.result_store.save_population(generation, population)
            self._save_qd_archive(generation)
            self._save_route_state(generation)
            workflow_audit["population_after_update"] = [
                self._candidate_audit(candidate) for candidate in population
            ]
            self.result_store.save_workflow_audit(generation, workflow_audit)
            self._save_cost_audit(generation)
            self.memory.compact_if_needed(population)
            self._progress(
                "generation_done",
                generation=generation,
                generations=generations,
                population_size=len(population),
                best_objective=population[0].objective if population else None,
                elapsed_min=round((time.time() - generation_start) / 60, 2),
            )
        return population

    def _update_population(self, population: list, offspring: list, generation: int | None = None) -> list:
        if self.route_manager is not None:
            return self._update_route_population(population, offspring, generation)
        if self.qd_archive is None:
            return self.population_manager.update(population, offspring)
        candidates = [candidate for candidate in list(population) + list(offspring) if candidate.objective is not None]
        for candidate in candidates:
            candidate.objective = PopulationManager._objective_key(candidate.objective)
            if candidate.selection_score is None or not self.population_manager.use_selection_score:
                candidate.selection_score = candidate.objective
        self.qd_archive.update(candidates)
        selected = self.qd_archive.select_population(int(self.config["evolution"]["pop_size"]))
        self.memory.session_log.append(
            {
                "event": "qd_archive_update",
                "coverage": self.qd_archive.coverage,
                "selected": [candidate.id for candidate in selected],
            }
        )
        return selected

    def _uses_qd_cells(self) -> bool:
        return self.config.get("evolution", {}).get("survival_selection") == "qd_cells"

    def _update_route_population(self, population: list, offspring: list, generation: int | None = None) -> list:
        candidates = [candidate for candidate in list(population) + list(offspring) if candidate.objective is not None]
        for candidate in candidates:
            candidate.objective = PopulationManager._objective_key(candidate.objective)
            if candidate.selection_score is None or not self.population_manager.use_selection_score:
                candidate.selection_score = candidate.objective
        if self.qd_archive is not None:
            self.qd_archive.update(candidates)
        updates = self.route_manager.update(candidates, generation=generation)
        if self.emitter_scheduler is not None:
            self.emitter_scheduler.record_updates(updates)
        selected = self.route_manager.best_candidates()[: int(self.config["evolution"]["pop_size"])]
        self.memory.session_log.append(
            {
                "event": "route_manager_update",
                "coverage": self.route_manager.coverage,
                "selected": [candidate.id for candidate in selected],
                "updates": [update.__dict__ for update in updates],
            }
        )
        return selected

    def _generate_offspring_batch(self, population: list, tasks: list[tuple]) -> list[dict]:
        if not tasks:
            return []
        parallel_jobs = min(self.n_proc, len(tasks))
        print(
            f"    Starting parallel generation for {len(tasks)} offspring using {parallel_jobs} workers...",
            flush=True,
        )
        original_progress = self.harness.progress
        self.harness.progress = None
        try:
            if parallel_jobs == 1:
                results = [
                    self._generate_one_offspring(population, generation, offspring_index, offspring_total, parent, offspring_plan)
                    for generation, offspring_index, offspring_total, parent, offspring_plan in tasks
                ]
            else:
                results = Parallel(n_jobs=parallel_jobs)(
                    delayed(self._generate_one_offspring)(population, generation, offspring_index, offspring_total, parent, offspring_plan)
                    for generation, offspring_index, offspring_total, parent, offspring_plan in tasks
                )
        finally:
            self.harness.progress = original_progress
        print("    Parallel generation finished.", flush=True)
        return results

    def _generate_one_offspring(self, population: list, generation: int, offspring_index: int, offspring_total: int, parent, offspring_plan: dict) -> dict:
        if self._uses_solver_tree_workflow():
            return self._generate_one_solver_tree_offspring(
                population,
                generation,
                offspring_index,
                offspring_total,
                parent,
                offspring_plan,
            )
        if self._uses_arch_spec_workflow():
            return self._generate_one_arch_spec_offspring(
                population,
                generation,
                offspring_index,
                offspring_total,
                parent,
                offspring_plan,
            )

        max_llm_retries = int(self.config["evolution"].get("max_llm_retries", 3))
        max_eval_retries = int(self.config["evolution"].get("max_eval_retries", 1))
        last_failure_reason = "Unknown"
        all_events = []
        entered_evaluator_any = False
        known_codes = {candidate.code for candidate in population if candidate.code}
        if getattr(parent, "code", None):
            known_codes.add(parent.code)
        candidate_id = self._candidate_id(generation, offspring_index, offspring_total)

        for attempt in range(1, max_llm_retries + 1):
            candidate = None
            events = []
            try:
                print(
                    f"    [Worker] gen={generation} off={offspring_index}/{offspring_total} attempt={attempt}/{max_llm_retries}",
                    flush=True,
                )
                if not bool(self.config.get("agent", {}).get("enable_generation_planning", True)):
                    raise RuntimeError("planned offspring execution requires a generation plan")
                candidate = self.harness.run_with_plan(
                    parent,
                    population,
                    offspring_plan,
                    generation=generation,
                    offspring_index=offspring_index,
                    offspring_total=offspring_total,
                    record_memory=False,
                    emit_progress=False,
                )
                events = self.harness.consume_buffered_events()
                if candidate is None:
                    all_events.extend(events)
                    last_failure_reason = "Agent rejected candidate before evaluation"
                    all_events.append(
                        self._implementation_retry_event(
                            generation,
                            offspring_index,
                            attempt,
                            last_failure_reason,
                            offspring_plan,
                        )
                    )
                    print(f"    [Worker] gen={generation} off={offspring_index}: rejected before evaluation", flush=True)
                    time.sleep(0.5)
                    continue

                self._reassign_candidate_id(candidate, events, candidate_id)
                candidate.generation = generation
                candidate.metadata["generation_plan"] = offspring_plan
                all_events.extend(events)

                if self.config["evolution"].get("deduplicate_code", True) and candidate.code in known_codes:
                    last_failure_reason = "Duplicate code generated"
                    all_events.append(
                        self._implementation_retry_event(
                            generation,
                            offspring_index,
                            attempt,
                            last_failure_reason,
                            offspring_plan,
                            candidate,
                        )
                    )
                    print(f"    [Worker] gen={generation} off={offspring_index}: duplicate code, retrying", flush=True)
                    time.sleep(0.5)
                    continue

                eval_attempt = 0
                objective = None
                eval_elapsed = None
                while eval_attempt < max_eval_retries:
                    eval_attempt += 1
                    eval_start = time.time()
                    entered_evaluator_any = True
                    objective = self._evaluate_individual(candidate.code, candidate.id)
                    eval_elapsed = round(time.time() - eval_start, 2)
                    if objective is not None:
                        break
                    last_failure_reason = "Code evaluation failed (syntax/runtime error or timeout)"
                    if eval_attempt < max_eval_retries:
                        print(
                            f"    [Worker] gen={generation} off={offspring_index}: eval retry {eval_attempt}/{max_eval_retries}",
                            flush=True,
                        )
                        time.sleep(0.5)

                if objective is None:
                    candidate.status = "eval_failed"
                    candidate.metadata["eval_error"] = last_failure_reason
                    all_events.append(
                        self._implementation_retry_event(
                            generation,
                            offspring_index,
                            attempt,
                            last_failure_reason,
                            offspring_plan,
                            candidate,
                            entered_evaluator=True,
                        )
                    )
                    continue

                candidate.objective = objective
                self._attach_evaluation_evidence(candidate, parent)
                candidate.selection_score = self._selection_score(candidate)
                candidate.status = "evaluated"
                plan_alignment = self._plan_alignment(candidate, offspring_plan)
                return {
                    "status": "evaluated",
                    "candidate": candidate,
                    "events": all_events,
                    "generation": generation,
                    "offspring_index": offspring_index,
                    "offspring_total": offspring_total,
                    "parent_id": parent.id,
                    "eval_elapsed": eval_elapsed,
                    "attempt": attempt,
                    "implementation_retries": attempt - 1,
                    "entered_evaluator": True,
                    "plan_alignment": plan_alignment,
                }
            except Exception as exc:
                events = self.harness.consume_buffered_events()
                all_events.extend(events)
                last_failure_reason = f"Exception: {exc}"
                all_events.append(
                    self._implementation_retry_event(
                        generation,
                        offspring_index,
                        attempt,
                        last_failure_reason,
                        offspring_plan,
                    )
                )
                print(
                    f"    [Worker] gen={generation} off={offspring_index} attempt={attempt} failed: {exc}",
                    flush=True,
                )
                time.sleep(1)

        return {
            "status": "failed",
            "candidate": candidate,
            "events": all_events,
            "generation": generation,
            "offspring_index": offspring_index,
            "offspring_total": offspring_total,
            "parent_id": parent.id,
            "error": last_failure_reason,
            "attempt": max_llm_retries,
            "implementation_retries": max_llm_retries,
            "entered_evaluator": entered_evaluator_any,
            "plan_alignment": {"aligned": False, "reason": last_failure_reason},
        }

    def _generate_one_arch_spec_offspring(
        self,
        population: list,
        generation: int,
        offspring_index: int,
        offspring_total: int,
        parent,
        offspring_plan: dict,
    ) -> dict:
        candidate_id = self._candidate_id(generation, offspring_index, offspring_total)
        entered_evaluator = False
        events = []
        try:
            print(
                f"    [Worker] gen={generation} off={offspring_index}/{offspring_total} spec_plan",
                flush=True,
            )
            review = self._pre_eval_review(parent, offspring_plan)
            events.append(
                {
                    "event": "pre_eval_review",
                    "parent": parent.id,
                    "offspring_plan": offspring_plan,
                    **review,
                }
            )
            if not review.get("passed"):
                disabled_pattern = review.get("disabled_pattern")
                if review.get("rejection_reason") == "semantic_noop" and disabled_pattern:
                    self._record_disabled_pattern(disabled_pattern)
                reason = f"pre_eval_review_failed: {review.get('rejection_reason')}"
                result = self._failed_offspring_result(
                    generation,
                    offspring_index,
                    offspring_total,
                    parent.id,
                    None,
                    events,
                    reason,
                    entered_evaluator,
                )
                if disabled_pattern:
                    result["disabled_pattern"] = disabled_pattern
                return result

            candidate = self._candidate_from_arch_spec_plan(parent, offspring_plan, candidate_id, generation)
            events.append(
                {
                    "event": "candidate_generated",
                    "candidate_id": candidate.id,
                    "parent": parent.id,
                    "offspring_plan": offspring_plan,
                    "arch_spec_diff": candidate.metadata.get("spec_diff"),
                }
            )
            known_codes = {existing.code for existing in population if existing.code}
            if candidate.code in known_codes:
                reason = "Duplicate ARCH_SPEC code generated"
                events.append(self._implementation_retry_event(generation, offspring_index, 1, reason, offspring_plan, candidate))
                return self._failed_offspring_result(
                    generation,
                    offspring_index,
                    offspring_total,
                    parent.id,
                    candidate,
                    events,
                    reason,
                    entered_evaluator,
                )

            errors, warnings = self.harness.tool_router._check_code(candidate.code)
            candidate.contract_passed = not errors
            events.append(
                {
                    "event": "skill_called",
                    "skill": "contract_check",
                    "result_event": "contract_check",
                    "candidate_id": candidate.id,
                    "contract_passed": not errors,
                    "errors": errors,
                    "warnings": warnings,
                }
            )
            if errors:
                reason = "; ".join(errors)
                events.append(self._implementation_retry_event(generation, offspring_index, 1, reason, offspring_plan, candidate))
                return self._failed_offspring_result(
                    generation,
                    offspring_index,
                    offspring_total,
                    parent.id,
                    candidate,
                    events,
                    reason,
                    entered_evaluator,
                )

            eval_start = time.time()
            entered_evaluator = True
            objective = self._evaluate_individual(candidate.code, candidate.id)
            eval_elapsed = round(time.time() - eval_start, 2)
            if objective is None:
                reason = "Code evaluation failed (syntax/runtime error or timeout)"
                candidate.status = "eval_failed"
                candidate.metadata["eval_error"] = reason
                events.append(
                    self._implementation_retry_event(
                        generation,
                        offspring_index,
                        1,
                        reason,
                        offspring_plan,
                        candidate,
                        entered_evaluator=True,
                    )
                )
                return self._failed_offspring_result(
                    generation,
                    offspring_index,
                    offspring_total,
                    parent.id,
                    candidate,
                    events,
                    reason,
                    entered_evaluator,
                )

            candidate.objective = objective
            self._attach_evaluation_evidence(candidate, parent)
            candidate.selection_score = self._selection_score(candidate)
            candidate.status = "evaluated"
            plan_alignment = self._plan_alignment(candidate, offspring_plan)
            if not plan_alignment.get("aligned"):
                reason = f"ARCH_SPEC plan alignment failed: {plan_alignment}"
                candidate.status = "plan_misaligned"
                events.append(self._implementation_retry_event(generation, offspring_index, 1, reason, offspring_plan, candidate, True))
                return self._failed_offspring_result(
                    generation,
                    offspring_index,
                    offspring_total,
                    parent.id,
                    candidate,
                    events,
                    reason,
                    entered_evaluator,
                    plan_alignment,
                )
            return {
                "status": "evaluated",
                "candidate": candidate,
                "events": events,
                "generation": generation,
                "offspring_index": offspring_index,
                "offspring_total": offspring_total,
                "parent_id": parent.id,
                "eval_elapsed": eval_elapsed,
                "attempt": 1,
                "implementation_retries": 0,
                "entered_evaluator": True,
                "plan_alignment": plan_alignment,
            }
        except Exception as exc:
            reason = f"Exception: {exc}"
            events.append(self._implementation_retry_event(generation, offspring_index, 1, reason, offspring_plan))
            print(f"    [Worker] gen={generation} off={offspring_index} spec_plan failed: {exc}", flush=True)
            return self._failed_offspring_result(
                generation,
                offspring_index,
                offspring_total,
                parent.id,
                None,
                events,
                reason,
                entered_evaluator,
            )

    def _candidate_from_arch_spec_plan(self, parent, offspring_plan: dict, candidate_id: str, generation: int) -> Candidate:
        parent_spec = arch_spec_from_candidate(parent)
        child_spec = offspring_plan.get("child_spec")
        if not isinstance(child_spec, dict):
            raise ValueError("tsp_gls_arch_spec offspring plan must include child_spec")
        controlled_layer = str(offspring_plan.get("controlled_component") or "")
        changed_layers = validate_single_layer_change(parent_spec, child_spec, controlled_layer)
        code = build_arch_spec_code(child_spec)
        spec_diff = {
            "changed_layers": changed_layers,
            "controlled_component": controlled_layer,
            "parent_arch_spec": parent_spec,
            "child_spec": child_spec,
        }
        mechanism_family = mechanism_family_from_spec(child_spec)
        algorithm = (
            f"Hypothesis: {offspring_plan.get('group_hypothesis') or offspring_plan.get('delta')}\n"
            f"Mechanism family: {mechanism_family}\n"
            f"Parent delta: {offspring_plan.get('delta')}\n"
            f"Expected effect: {offspring_plan.get('expected_effect')}\n"
            f"Controlled component: {controlled_layer}\n"
            f"{describe_arch_spec(child_spec)}"
        )
        return Candidate(
            id=candidate_id,
            parents=[parent.id],
            algorithm=algorithm,
            code=code,
            mechanism_summary=describe_arch_spec(child_spec),
            mechanism_hypothesis={
                "id": f"mh_{candidate_id}",
                "parent_id": parent.id,
                "problem": self.config["project"]["problem"],
                "text": algorithm,
                "mechanism_family": mechanism_family,
                "parent_delta": str(offspring_plan.get("delta") or ""),
                "expected_effect": str(offspring_plan.get("expected_effect") or ""),
                "preserve_components": [str(offspring_plan.get("preserve") or "")],
                "change_components": [controlled_layer],
                "mechanism_components": {
                    "architecture_layer": controlled_layer,
                    "distance_update_policy": child_spec["distance_update_policy"]["type"],
                    "penalty_policy": child_spec["penalty_policy"]["edge_selection"],
                    "escape_policy": child_spec["escape_policy"]["trigger"],
                    "restart_policy": child_spec["restart_policy"]["action"],
                    "schedule_policy": child_spec["schedule_policy"]["strength_schedule"],
                },
                "risk_assumptions": [],
                "evidence_summary": {},
                "status": "proposed",
            },
            mechanism_profile={
                "mechanism_family": mechanism_family,
                "hypothesis_text": algorithm,
                "parent_delta": str(offspring_plan.get("delta") or ""),
                "expected_effect": str(offspring_plan.get("expected_effect") or ""),
                "mechanism_components": {
                    "architecture_layer": controlled_layer,
                    "arch_spec_niche": self._arch_spec_niche(child_spec),
                },
            },
            parent_delta=str(offspring_plan.get("delta") or ""),
            generation=generation,
            metadata={
                "planned_experiment": offspring_plan,
                "generation_plan": offspring_plan,
                "arch_spec": child_spec,
                "parent_arch_spec": parent_spec,
                "spec_diff": spec_diff,
                "code_generation": "deterministic_from_arch_spec",
            },
        )

    def _generate_one_solver_tree_offspring(
        self,
        population: list,
        generation: int,
        offspring_index: int,
        offspring_total: int,
        parent,
        offspring_plan: dict,
    ) -> dict:
        candidate_id = self._candidate_id(generation, offspring_index, offspring_total)
        entered_evaluator = False
        events = []
        try:
            print(
                f"    [Worker] gen={generation} off={offspring_index}/{offspring_total} solver_tree_plan",
                flush=True,
            )
            review = self._pre_eval_review(parent, offspring_plan)
            events.append(
                {
                    "event": "pre_eval_review",
                    "parent": parent.id,
                    "offspring_plan": offspring_plan,
                    **review,
                }
            )
            if not review.get("passed"):
                reason = f"pre_eval_review_failed: {review.get('rejection_reason')}"
                return self._failed_offspring_result(
                    generation,
                    offspring_index,
                    offspring_total,
                    parent.id,
                    None,
                    events,
                    reason,
                    entered_evaluator,
                )

            candidate = self._candidate_from_solver_tree_plan(parent, offspring_plan, candidate_id, generation)
            events.append(
                {
                    "event": "candidate_generated",
                    "candidate_id": candidate.id,
                    "parent": parent.id,
                    "offspring_plan": offspring_plan,
                    "solver_tree_diff": candidate.metadata.get("tree_diff"),
                }
            )
            known_codes = {existing.code for existing in population if existing.code}
            if candidate.code in known_codes:
                reason = "Duplicate SOLVER_TREE code generated"
                events.append(self._implementation_retry_event(generation, offspring_index, 1, reason, offspring_plan, candidate))
                return self._failed_offspring_result(
                    generation,
                    offspring_index,
                    offspring_total,
                    parent.id,
                    candidate,
                    events,
                    reason,
                    entered_evaluator,
                )

            errors, warnings = self.harness.tool_router._check_code(candidate.code)
            candidate.contract_passed = not errors
            events.append(
                {
                    "event": "skill_called",
                    "skill": "contract_check",
                    "result_event": "contract_check",
                    "candidate_id": candidate.id,
                    "contract_passed": not errors,
                    "errors": errors,
                    "warnings": warnings,
                }
            )
            if errors:
                reason = "; ".join(errors)
                events.append(self._implementation_retry_event(generation, offspring_index, 1, reason, offspring_plan, candidate))
                return self._failed_offspring_result(
                    generation,
                    offspring_index,
                    offspring_total,
                    parent.id,
                    candidate,
                    events,
                    reason,
                    entered_evaluator,
                )

            eval_start = time.time()
            entered_evaluator = True
            objective = self._evaluate_individual(candidate.code, candidate.id)
            eval_elapsed = round(time.time() - eval_start, 2)
            if objective is None:
                reason = "Code evaluation failed (syntax/runtime error or timeout)"
                candidate.status = "eval_failed"
                candidate.metadata["eval_error"] = reason
                events.append(self._implementation_retry_event(generation, offspring_index, 1, reason, offspring_plan, candidate, True))
                return self._failed_offspring_result(
                    generation,
                    offspring_index,
                    offspring_total,
                    parent.id,
                    candidate,
                    events,
                    reason,
                    entered_evaluator,
                )

            candidate.objective = objective
            self._attach_evaluation_evidence(candidate, parent)
            candidate.selection_score = self._selection_score(candidate)
            candidate.status = "evaluated"
            plan_alignment = self._plan_alignment(candidate, offspring_plan)
            if not plan_alignment.get("aligned"):
                reason = f"SOLVER_TREE plan alignment failed: {plan_alignment}"
                candidate.status = "plan_misaligned"
                events.append(self._implementation_retry_event(generation, offspring_index, 1, reason, offspring_plan, candidate, True))
                return self._failed_offspring_result(
                    generation,
                    offspring_index,
                    offspring_total,
                    parent.id,
                    candidate,
                    events,
                    reason,
                    entered_evaluator,
                    plan_alignment,
                )
            self.memory.record_experiment_memory(
                kind="solver_tree_candidate",
                candidate_id=candidate.id,
                generation=generation,
                tree_descriptor=solver_tree_descriptor(candidate.metadata["solver_tree"]),
                verdict=candidate.hypothesis_verdict or "uncertain",
                evidence_source={"objective": candidate.objective, "parent_id": parent.id},
                content=describe_solver_tree(candidate.metadata["solver_tree"]),
            )
            return {
                "status": "evaluated",
                "candidate": candidate,
                "events": events,
                "generation": generation,
                "offspring_index": offspring_index,
                "offspring_total": offspring_total,
                "parent_id": parent.id,
                "eval_elapsed": eval_elapsed,
                "attempt": 1,
                "implementation_retries": 0,
                "entered_evaluator": True,
                "plan_alignment": plan_alignment,
            }
        except Exception as exc:
            reason = f"Exception: {exc}"
            events.append(self._implementation_retry_event(generation, offspring_index, 1, reason, offspring_plan))
            print(f"    [Worker] gen={generation} off={offspring_index} solver_tree_plan failed: {exc}", flush=True)
            return self._failed_offspring_result(
                generation,
                offspring_index,
                offspring_total,
                parent.id,
                None,
                events,
                reason,
                entered_evaluator,
            )

    def _candidate_from_solver_tree_plan(self, parent, offspring_plan: dict, candidate_id: str, generation: int) -> Candidate:
        parent_tree = solver_tree_from_candidate(parent)
        child_tree = canonical_solver_tree(offspring_plan.get("child_solver_tree"))
        mutation_type = classify_solver_tree_mutation(parent_tree, child_tree)
        changed_paths = changed_solver_tree_paths(parent_tree, child_tree)
        code = build_solver_tree_code(child_tree)
        descriptor = solver_tree_descriptor(child_tree)
        tree_diff = {
            "changed_paths": changed_paths,
            "mutation_type": mutation_type,
            "controlled_node_path": str(offspring_plan.get("controlled_node_path") or ""),
            "parent_solver_tree": parent_tree,
            "child_solver_tree": child_tree,
        }
        mechanism_family = "_".join(descriptor[1:3])
        algorithm = (
            f"Hypothesis: {offspring_plan.get('group_hypothesis') or offspring_plan.get('delta')}\n"
            f"Mechanism family: {mechanism_family}\n"
            f"Mutation type: {mutation_type}\n"
            f"Controlled node path: {offspring_plan.get('controlled_node_path')}\n"
            f"Expected trace change: {offspring_plan.get('expected_trace_change')}\n"
            f"{describe_solver_tree(child_tree)}"
        )
        return Candidate(
            id=candidate_id,
            parents=[parent.id],
            algorithm=algorithm,
            code=code,
            mechanism_summary=describe_solver_tree(child_tree),
            mechanism_hypothesis={
                "id": f"mh_{candidate_id}",
                "parent_id": parent.id,
                "problem": self.config["project"]["problem"],
                "text": algorithm,
                "mechanism_family": mechanism_family,
                "parent_delta": str(offspring_plan.get("delta") or ""),
                "expected_effect": str(offspring_plan.get("expected_effect") or offspring_plan.get("expected_trace_change") or ""),
                "preserve_components": [str(offspring_plan.get("preserve") or "")],
                "change_components": [mutation_type],
                "mechanism_components": {
                    "tree_descriptor": list(descriptor),
                    "mutation_type": mutation_type,
                    "controlled_node_path": str(offspring_plan.get("controlled_node_path") or ""),
                },
                "risk_assumptions": [],
                "evidence_summary": {},
                "status": "proposed",
            },
            mechanism_profile={
                "mechanism_family": mechanism_family,
                "hypothesis_text": algorithm,
                "parent_delta": str(offspring_plan.get("delta") or ""),
                "expected_effect": str(offspring_plan.get("expected_effect") or offspring_plan.get("expected_trace_change") or ""),
                "mechanism_components": {
                    "tree_descriptor": list(descriptor),
                    "mutation_type": mutation_type,
                    "changed_paths": changed_paths,
                },
            },
            parent_delta=str(offspring_plan.get("delta") or ""),
            generation=generation,
            metadata={
                "planned_experiment": offspring_plan,
                "generation_plan": offspring_plan,
                "solver_tree": child_tree,
                "parent_solver_tree": parent_tree,
                "tree_diff": tree_diff,
                "code_generation": "deterministic_from_solver_tree",
            },
        )

    @staticmethod
    def _failed_offspring_result(
        generation: int,
        offspring_index: int,
        offspring_total: int,
        parent_id: str,
        candidate,
        events: list[dict],
        reason: str,
        entered_evaluator: bool,
        plan_alignment: dict | None = None,
    ) -> dict:
        return {
            "status": "failed",
            "candidate": candidate,
            "events": events,
            "generation": generation,
            "offspring_index": offspring_index,
            "offspring_total": offspring_total,
            "parent_id": parent_id,
            "error": reason,
            "attempt": 1,
            "implementation_retries": 0,
            "entered_evaluator": entered_evaluator,
            "plan_alignment": plan_alignment or {"aligned": False, "reason": reason},
        }

    def _uses_arch_spec_workflow(self) -> bool:
        return (
            self.config.get("project", {}).get("problem") == "tsp_gls_arch_spec"
            or self.config.get("evaluator", {}).get("problem") == "tsp_gls_arch_spec"
        )

    def _uses_solver_tree_workflow(self) -> bool:
        return (
            self.config.get("project", {}).get("problem") == "tsp_gls_solver_tree"
            or self.config.get("evaluator", {}).get("problem") == "tsp_gls_solver_tree"
        )

    def _pre_eval_review(self, parent, offspring_plan: dict) -> dict:
        reviewer = getattr(self, "experiment_reviewer", None)
        if reviewer is None:
            reviewer = ExperimentReviewer(self.config)
            self.experiment_reviewer = reviewer
        return reviewer.review_plan(
            parent_spec=solver_tree_from_candidate(parent) if self._uses_solver_tree_workflow() else arch_spec_from_candidate(parent),
            plan=offspring_plan,
            disabled_patterns=getattr(self, "disabled_patterns", []),
        )

    def _record_disabled_pattern(self, pattern: dict | None) -> None:
        if not isinstance(pattern, dict):
            return
        pattern_id = str(pattern.get("pattern_id") or "")
        if not pattern_id:
            return
        problem = pattern.get("problem")
        patterns = list(getattr(self, "disabled_patterns", []))
        for existing in patterns:
            if existing.get("pattern_id") == pattern_id and existing.get("problem") == problem:
                self.disabled_patterns = patterns
                if hasattr(self, "memory"):
                    self.memory.disabled_patterns = self.disabled_patterns
                return
        patterns.append(pattern)
        self.disabled_patterns = patterns
        if hasattr(self, "memory"):
            self.memory.disabled_patterns = self.disabled_patterns

    @staticmethod
    def _arch_spec_niche(spec: dict) -> list[str]:
        from eohagent.problems.tsp_gls_arch_spec.spec_schema import arch_spec_niche_key

        return list(arch_spec_niche_key(spec))

    def _evaluate_individual(self, code: str, candidate_id: str):
        if not isinstance(code, str) or "def" not in code:
            print(f"    [Evaluate] {candidate_id} invalid code received.", flush=True)
            return None
        try:
            ast.parse(code)
        except SyntaxError as exc:
            print(f"    [Evaluate] {candidate_id} syntax error: {exc}", flush=True)
            return None
        except Exception as exc:
            print(f"    [Evaluate] {candidate_id} AST parsing error: {exc}", flush=True)
            return None

        print(f"    [Evaluate] {candidate_id} Starting evaluation (timeout: {self.eval_timeout}s)...", flush=True)
        eval_start_time = time.time()
        executor = None
        try:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(self.evaluator.evaluate, code)
            try:
                fitness = future.result(timeout=self.eval_timeout)
                elapsed = time.time() - eval_start_time
                print(f"    [Evaluate] {candidate_id} Completed in {elapsed:.1f}s", flush=True)
            except concurrent.futures.TimeoutError:
                elapsed = time.time() - eval_start_time
                print(
                    f"    [TIMEOUT] {candidate_id} Evaluation exceeded {self.eval_timeout}s (elapsed: {elapsed:.1f}s)",
                    flush=True,
                )
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)
                executor = None
                return None
            except Exception as exc:
                elapsed = time.time() - eval_start_time
                print(f"    [Evaluate] {candidate_id} runtime error after {elapsed:.1f}s: {exc}", flush=True)
                return None
        except Exception as exc:
            print(f"    [Evaluate] {candidate_id} unexpected evaluation error: {exc}", flush=True)
            return None
        finally:
            if executor is not None:
                try:
                    executor.shutdown(wait=False, cancel_futures=True)
                except TypeError:
                    executor.shutdown(wait=False)

        if fitness is None:
            print(f"    [Evaluate] {candidate_id} returned None.", flush=True)
            return None
        if isinstance(fitness, np.ndarray):
            fitness = np.mean(fitness)
        try:
            if not np.isfinite(fitness):
                print(f"    [Evaluate] {candidate_id} non-finite fitness.", flush=True)
                return None
        except TypeError:
            print(f"    [Evaluate] {candidate_id} invalid fitness type: {type(fitness).__name__}", flush=True)
            return None
        return self._objective_value(fitness)

    def _merge_offspring_batch(self, population: list, results: list[dict]) -> list:
        offspring = []
        for result in results:
            for event in result.get("events", []):
                self.memory.session_log.append(event)
                self._merge_context_compaction_event(event)
            self._record_disabled_pattern(result.get("disabled_pattern"))
            generation = result.get("generation")
            offspring_index = result.get("offspring_index")
            candidate = result.get("candidate")
            if result.get("status") != "evaluated" or candidate is None:
                if candidate is not None:
                    self.memory.candidate_store.save(candidate)
                    self.memory.notebook.add_lesson_from_candidate(candidate, outcome="eval_failed")
                    self.memory.mechanism_archive.record_candidate(candidate, "eval_failed")
                    self.memory.session_log.append(
                        {
                            "event": "formal_eval_failed",
                            "candidate_id": candidate.id,
                            "error": result.get("error", "evaluation failed"),
                        }
                    )
                self._progress(
                    "offspring_rejected",
                    generation=generation,
                    offspring_index=offspring_index,
                    parent_id=result.get("parent_id"),
                    reason=result.get("error"),
                )
                continue

            if self.config["evolution"].get("deduplicate_code", True):
                known_codes = {c.code for c in population + offspring}
                if candidate.code in known_codes:
                    candidate.status = "duplicate"
                    self.memory.candidate_store.save(candidate)
                    self.memory.notebook.add_lesson_from_candidate(candidate, outcome="duplicate")
                    self.memory.mechanism_archive.record_candidate(candidate, "duplicate")
                    self.memory.session_log.append({"event": "candidate_duplicate", "candidate_id": candidate.id})
                    self._progress(
                        "candidate_duplicate",
                        generation=generation,
                        offspring_index=offspring_index,
                        candidate_id=candidate.id,
                    )
                    continue

            if self.config["evolution"].get("deduplicate_objective", True) and self._has_objective_duplicate(
                population + offspring,
                candidate,
            ):
                candidate.status = "objective_duplicate"
                self.memory.candidate_store.save(candidate)
                self.memory.notebook.add_lesson_from_candidate(candidate, outcome="objective_duplicate")
                self.memory.mechanism_archive.record_candidate(candidate, "objective_duplicate")
                self.memory.session_log.append(
                    {
                        "event": "objective_duplicate",
                        "candidate_id": candidate.id,
                        "objective": candidate.objective,
                        "reason": "EoH-style population management keeps one individual per rounded objective.",
                    }
                )
                self._progress(
                    "objective_duplicate",
                    generation=generation,
                    offspring_index=offspring_index,
                    candidate_id=candidate.id,
                    objective=candidate.objective,
                    elapsed_sec=result.get("eval_elapsed"),
                )
                continue

            candidate.status = "evaluated"
            self.memory.candidate_store.save(candidate)
            self.memory.notebook.add_lesson_from_candidate(candidate, outcome="evaluated")
            self.memory.mechanism_archive.record_candidate(candidate, "evaluated")
            self.memory.session_log.append(
                {
                    "event": "formal_eval",
                    "candidate_id": candidate.id,
                    "objective": candidate.objective,
                    "selection_score": candidate.selection_score,
                    "hypothesis_verdict": candidate.hypothesis_verdict,
                    "training_diagnosis": (candidate.evidence or {}).get("training_diagnosis"),
                    "generation_plan": candidate.metadata.get("generation_plan"),
                }
            )
            self._progress(
                "eval_done",
                generation=generation,
                offspring_index=offspring_index,
                candidate_id=candidate.id,
                objective=candidate.objective,
                elapsed_sec=result.get("eval_elapsed"),
            )
            offspring.append(candidate)
        return offspring

    def _merge_context_compaction_event(self, event: dict) -> None:
        if event.get("event") != "context_compaction" or not isinstance(event.get("working_memory"), dict):
            return
        context = self.memory.compact_context.read()
        context["working_memory"] = event["working_memory"]
        context["context_management"] = {
            "last_compaction_source": "parallel_worker",
            "purpose": event.get("purpose"),
            "estimated_tokens_before": event.get("estimated_tokens_before"),
            "pressure_before": event.get("pressure_before"),
            "threshold": self.config.get("context_management", {}).get("compact_threshold"),
            "context_window": self.config.get("context_management", {}).get("context_window"),
        }
        self.memory.compact_context.write(context)

    def _save_cost_audit(self, generation: int) -> None:
        calls = []
        for event in self.memory.session_log.recent(500):
            if event.get("generation") != generation or not isinstance(event.get("llm_usage"), dict):
                continue
            usage = event.get("llm_usage") or {}
            compression = event.get("prompt_compression") or {}
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            completion_tokens = int(usage.get("completion_tokens") or 0)
            cache_hit = self._cache_hit_tokens(usage)
            cache_miss = self._cache_miss_tokens(usage, prompt_tokens, cache_hit)
            calls.append(
                {
                    "event": event.get("event"),
                    "purpose": compression.get("purpose"),
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": int(usage.get("total_tokens") or prompt_tokens + completion_tokens),
                    "cache_hit_tokens": cache_hit,
                    "cache_miss_tokens": cache_miss,
                    "cache_hit_rate": 0.0 if prompt_tokens <= 0 else round(cache_hit / prompt_tokens, 6),
                    "compression": compression,
                }
            )
        totals = {
            "prompt_tokens": sum(call["prompt_tokens"] for call in calls),
            "completion_tokens": sum(call["completion_tokens"] for call in calls),
            "total_tokens": sum(call["total_tokens"] for call in calls),
            "cache_hit_tokens": sum(call["cache_hit_tokens"] for call in calls),
            "cache_miss_tokens": sum(call["cache_miss_tokens"] for call in calls),
            "compression_saved_chars": sum(int((call["compression"] or {}).get("saved_chars") or 0) for call in calls),
        }
        totals["cache_hit_rate"] = (
            0.0 if totals["prompt_tokens"] <= 0 else round(totals["cache_hit_tokens"] / totals["prompt_tokens"], 6)
        )
        self.result_store.save_cost_audit(
            generation,
            {
                "generation": generation,
                "calls": calls,
                "totals": totals,
            },
        )

    def _save_qd_archive(self, generation: int) -> None:
        if self.qd_archive is None:
            return
        self.result_store.save_qd_archive(generation, self.qd_archive)

    def _save_route_state(self, generation: int) -> None:
        if self.route_manager is None:
            return
        self.result_store.save_route_state(generation, self.route_manager)

    @staticmethod
    def _cache_hit_tokens(usage: dict) -> int:
        details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
        return int(usage.get("prompt_cache_hit_tokens") or details.get("cached_tokens") or 0)

    @staticmethod
    def _cache_miss_tokens(usage: dict, prompt_tokens: int, cache_hit: int) -> int:
        if usage.get("prompt_cache_miss_tokens") is not None:
            return int(usage.get("prompt_cache_miss_tokens") or 0)
        return max(0, int(prompt_tokens) - int(cache_hit))

    def _selection_score(self, candidate) -> float:
        base = float(candidate.objective)
        evidence = getattr(candidate, "evidence", None) or {}
        diagnosis = evidence.get("training_diagnosis") if isinstance(evidence, dict) else {}
        if not isinstance(diagnosis, dict):
            return round(base, 8)
        penalty = 0.0
        for key, weight in [("tail_gap", 1e-3), ("max_gap", 1e-4), ("std_gap", 1e-5)]:
            value = diagnosis.get(key)
            if value is not None:
                penalty += weight * float(value)
        if getattr(self, "objective_direction", "minimize") == "maximize":
            return round(base - penalty, 8)
        return round(base + penalty, 8)

    def _attach_evaluation_evidence(self, candidate, parent) -> None:
        evaluator_evidence = getattr(self.evaluator, "last_evidence", {}) or {}
        parent_evidence = {}
        if isinstance(getattr(parent, "evidence", None), dict):
            parent_evidence = parent.evidence.get("formal_evaluator") or {}
        training_diagnosis = build_training_diagnosis(evaluator_evidence, parent_evidence)
        evidence = {
            "train_objective": candidate.objective,
            "parent_objective": parent.objective,
            "delta_vs_parent": None
            if parent.objective is None or candidate.objective is None
            else round(float(candidate.objective) - float(parent.objective), 8),
            "formal_evaluator": evaluator_evidence,
            "training_diagnosis": training_diagnosis,
        }
        candidate.evidence = evidence
        candidate.metadata["evaluation_evidence"] = evaluator_evidence
        candidate.metadata["training_diagnosis"] = training_diagnosis
        candidate.hypothesis_verdict = self._hypothesis_verdict(candidate, parent)
        if isinstance(candidate.mechanism_hypothesis, dict):
            candidate.mechanism_hypothesis["evidence_summary"] = {
                "objective": candidate.objective,
                "delta_vs_parent": evidence["delta_vs_parent"],
                "std_gap": evaluator_evidence.get("std_gap") if isinstance(evaluator_evidence, dict) else None,
                "max_gap": evaluator_evidence.get("max_gap") if isinstance(evaluator_evidence, dict) else None,
                "tail_gap": training_diagnosis.get("tail_gap") if isinstance(training_diagnosis, dict) else None,
                "failure_signature": training_diagnosis.get("failure_signature") if isinstance(training_diagnosis, dict) else None,
            }
            candidate.mechanism_hypothesis["status"] = candidate.hypothesis_verdict

    @staticmethod
    def _hypothesis_verdict(candidate, parent) -> str:
        if candidate.objective is None or parent.objective is None:
            return "uncertain"
        delta = float(candidate.objective) - float(parent.objective)
        if delta < -1e-8:
            return "supported"
        if delta > 1e-8:
            return "refuted"
        return "uncertain"

    @staticmethod
    def _objective_value(raw_objective) -> float:
        return round(float(raw_objective), 5)

    @staticmethod
    def _has_objective_duplicate(existing: list, candidate) -> bool:
        if candidate.objective is None:
            return False
        objective = round(float(candidate.objective), 5)
        return any(c.objective is not None and round(float(c.objective), 5) == objective for c in existing)

    @staticmethod
    def _candidate_id(generation: int, offspring_index: int, offspring_total: int) -> str:
        return f"h_{((generation - 1) * offspring_total + offspring_index):04d}"

    @staticmethod
    def _reassign_candidate_id(candidate, events: list[dict], candidate_id: str) -> None:
        old_id = candidate.id
        candidate.id = candidate_id
        if isinstance(candidate.mechanism_hypothesis, dict):
            candidate.mechanism_hypothesis["id"] = f"mh_{candidate_id}"
        for metadata_key in ["variation", "repair_extraction"]:
            item = candidate.metadata.get(metadata_key) if isinstance(candidate.metadata, dict) else None
            if isinstance(item, dict) and isinstance(item.get("mechanism_hypothesis"), dict):
                item["mechanism_hypothesis"]["id"] = f"mh_{candidate_id}"
        for event in events:
            if event.get("candidate_id") == old_id:
                event["candidate_id"] = candidate_id

    def _new_workflow_audit(self, generation: int, generations: int, offspring_total: int, population: list) -> dict:
        return {
            "generation": generation,
            "generations": generations,
            "workflow_contract": {
                "requires_generation_planning": True,
                "requires_exact_offspring_plan_count": offspring_total,
                "requires_planned_offspring_execution": True,
                "requires_contract_check_before_eval": True,
                "repair_scope": "implementation_only",
                "replacement_plan_allowed": False,
            },
            "population_before_planning": [self._candidate_audit(candidate) for candidate in population],
            "planner": {"success": False, "complete": False, "plan_count": 0},
            "offspring": [],
            "reflection": {"success": None},
            "population_after_update": [],
        }

    @staticmethod
    def _planner_audit(generation_plan: dict, offspring_total: int) -> dict:
        plans = generation_plan.get("offspring_plans", [])
        return {
            "success": True,
            "complete": len(plans) == offspring_total,
            "plan_count": len(plans),
            "required_plan_count": offspring_total,
            "generation_goal": generation_plan.get("generation_goal"),
            "avoid_repeating": generation_plan.get("avoid_repeating", []),
            "plans": plans,
        }

    def _offspring_audit(self, result: dict) -> dict:
        candidate = result.get("candidate")
        plan = {}
        if isinstance(getattr(candidate, "metadata", None), dict):
            plan = candidate.metadata.get("generation_plan") or candidate.metadata.get("planned_experiment") or {}
        if not plan:
            events = result.get("events", [])
            for event in reversed(events):
                if isinstance(event.get("offspring_plan"), dict):
                    plan = event["offspring_plan"]
                    break
        return {
            "offspring_index": result.get("offspring_index"),
            "parent_id": result.get("parent_id"),
            "status": result.get("status"),
            "candidate_id": candidate.id if candidate else None,
            "objective": candidate.objective if candidate else None,
            "attempt": result.get("attempt"),
            "implementation_retries": result.get("implementation_retries", 0),
            "planned_experiment": plan,
            "contract_checked": self._event_seen(result, "skill_called", skill="contract_check"),
            "repair_used": self._event_seen(result, "skill_called", skill="repair"),
            "entered_evaluator": bool(result.get("entered_evaluator")),
            "plan_alignment": result.get("plan_alignment") or {"aligned": False, "reason": result.get("error")},
            "spec_diff": candidate.metadata.get("spec_diff") if candidate and isinstance(candidate.metadata, dict) else None,
            "error": result.get("error"),
        }

    @staticmethod
    def _event_seen(result: dict, event_name: str, **fields) -> bool:
        for event in result.get("events", []):
            if event.get("event") != event_name:
                continue
            if all(event.get(key) == value for key, value in fields.items()):
                return True
        return False

    @staticmethod
    def _candidate_audit(candidate) -> dict:
        hypothesis = candidate.mechanism_hypothesis if isinstance(candidate.mechanism_hypothesis, dict) else {}
        profile = candidate.mechanism_profile if isinstance(candidate.mechanism_profile, dict) else {}
        profile_components = profile.get("mechanism_components") if isinstance(profile.get("mechanism_components"), dict) else {}
        return {
            "id": candidate.id,
            "objective": candidate.objective,
            "selection_score": candidate.selection_score,
            "status": candidate.status,
            "mechanism_family": hypothesis.get("mechanism_family"),
            "mechanism_summary": str(candidate.mechanism_summary or "")[:240],
            "hypothesis_verdict": candidate.hypothesis_verdict,
            "arch_spec_niche": profile_components.get("arch_spec_niche"),
            "arch_spec": candidate.metadata.get("arch_spec") if isinstance(candidate.metadata, dict) else None,
        }

    @staticmethod
    def _implementation_retry_event(
        generation: int,
        offspring_index: int,
        attempt: int,
        reason: str,
        offspring_plan: dict,
        candidate=None,
        entered_evaluator: bool = False,
    ) -> dict:
        return {
            "event": "implementation_attempt_failed",
            "generation": generation,
            "offspring_index": offspring_index,
            "attempt": attempt,
            "reason": reason,
            "candidate_id": candidate.id if candidate else None,
            "entered_evaluator": entered_evaluator,
            "offspring_plan": offspring_plan,
        }

    @staticmethod
    def _plan_alignment(candidate, offspring_plan: dict) -> dict:
        planned = {}
        if isinstance(getattr(candidate, "metadata", None), dict):
            planned = candidate.metadata.get("planned_experiment") or {}
        mismatches = []
        for key in ["parent_id", "controlled_component", "delta"]:
            if str(planned.get(key) or "") != str(offspring_plan.get(key) or ""):
                mismatches.append(key)
        checked_fields = ["parent_id", "controlled_component", "delta"]
        spec_error = None
        changed_layers = []
        metadata = candidate.metadata if isinstance(getattr(candidate, "metadata", None), dict) else {}
        if isinstance(metadata.get("arch_spec"), dict):
            checked_fields.extend(["parent_arch_spec", "child_spec", "changed_arch_spec_layers"])
            try:
                parent_spec = canonical_arch_spec(metadata.get("parent_arch_spec"))
                child_spec = canonical_arch_spec(metadata.get("arch_spec"))
                plan_child_spec = canonical_arch_spec(offspring_plan.get("child_spec"))
                if plan_child_spec != child_spec:
                    mismatches.append("child_spec")
                controlled_layer = str(offspring_plan.get("controlled_component") or "")
                changed_layers = validate_single_layer_change(parent_spec, child_spec, controlled_layer)
                declared_layers = metadata.get("spec_diff", {}).get("changed_layers")
                if declared_layers != changed_layers:
                    mismatches.append("spec_diff.changed_layers")
            except Exception as exc:
                spec_error = str(exc)
                mismatches.append("arch_spec_single_layer_change")
        if isinstance(metadata.get("solver_tree"), dict):
            checked_fields.extend(["parent_solver_tree", "child_solver_tree", "changed_solver_tree_paths", "mutation_type"])
            try:
                parent_tree = canonical_solver_tree(metadata.get("parent_solver_tree"))
                child_tree = canonical_solver_tree(metadata.get("solver_tree"))
                plan_child_tree = canonical_solver_tree(offspring_plan.get("child_solver_tree"))
                if plan_child_tree != child_tree:
                    mismatches.append("child_solver_tree")
                mutation_type = classify_solver_tree_mutation(parent_tree, child_tree)
                if str(offspring_plan.get("mutation_type") or "") != mutation_type:
                    mismatches.append("mutation_type")
                declared_paths = metadata.get("tree_diff", {}).get("changed_paths")
                actual_paths = changed_solver_tree_paths(parent_tree, child_tree)
                if declared_paths != actual_paths:
                    mismatches.append("tree_diff.changed_paths")
            except Exception as exc:
                spec_error = str(exc)
                mismatches.append("solver_tree_diff")
        return {
            "aligned": not mismatches,
            "mismatches": mismatches,
            "checked_fields": checked_fields,
            "changed_arch_spec_layers": changed_layers,
            "changed_solver_tree_paths": metadata.get("tree_diff", {}).get("changed_paths") if isinstance(metadata.get("tree_diff"), dict) else [],
            "error": spec_error,
        }

    def _progress(self, event: str, **fields) -> None:
        record = {"t": time.time(), "event": event, **fields}
        with self.progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
        message = self._format_progress_message(record)
        print(message, flush=True)

    @staticmethod
    def _format_progress_message(record: dict) -> str:
        event = record["event"]
        if event == "run_start":
            return (
                f"[run] problem={record.get('problem')} pop={record.get('pop_size')} "
                f"generations={record.get('generations')} offspring/gen={record.get('offspring_per_generation')} "
                f"n_proc={record.get('n_proc')}"
            )
        if event == "bootstrap_start":
            return f"[bootstrap] start problem={record.get('problem')} evaluator={record.get('evaluator')}"
        if event == "bootstrap_parallel_start":
            return f"[bootstrap] evaluating {record.get('seed_total')} seeds using {record.get('n_jobs')} workers"
        if event == "bootstrap_seed_start":
            return (
                f"[bootstrap seed {record.get('seed_index')}/{record.get('seed_total')}] "
                f"{record.get('candidate_id')} start"
            )
        if event == "bootstrap_seed_done":
            return (
                f"[bootstrap seed {record.get('seed_index')}/{record.get('seed_total')}] "
                f"{record.get('candidate_id')} done objective={record.get('objective')} "
                f"elapsed={record.get('elapsed_sec')}s"
            )
        if event == "bootstrap_seed_failed":
            return (
                f"[bootstrap seed {record.get('seed_index')}/{record.get('seed_total')}] "
                f"{record.get('candidate_id')} failed error={record.get('error')}"
            )
        if event == "bootstrap_parallel_done":
            return f"[bootstrap] seed evaluation done workers={record.get('n_jobs')}"
        if event == "bootstrap_done":
            return f"[bootstrap] done size={record.get('population_size')} best={record.get('best_objective')}"
        if event == "generation_start":
            return (
                f"[gen {record.get('generation')}/{record.get('generations')}] start "
                f"best={record.get('best_objective')} pop_size={record.get('population_size')}"
            )
        if event == "generation_plan":
            return (
                f"[gen {record.get('generation')}] plan "
                f"count={record.get('plan_count')} goal={str(record.get('goal') or '')[:120]}"
            )
        if event == "offspring_start":
            return (
                f"[gen {record.get('generation')} off {record.get('offspring_index')}/{record.get('offspring_total')}] "
                f"parent={record.get('parent_id')} parent_obj={record.get('parent_objective')}"
            )
        if event == "eval_start":
            return f"[eval] {record.get('candidate_id')} start"
        if event == "eval_done":
            return (
                f"[eval] {record.get('candidate_id')} done objective={record.get('objective')} "
                f"elapsed={record.get('elapsed_sec')}s"
            )
        if event == "eval_failed":
            return f"[eval] {record.get('candidate_id')} failed error={record.get('error')}"
        if event == "generation_done":
            return (
                f"[gen {record.get('generation')}/{record.get('generations')}] done "
                f"best={record.get('best_objective')} elapsed={record.get('elapsed_min')}m"
            )
        if event.startswith("agent_"):
            parts = [f"[agent] {event}"]
            for key in ["generation", "offspring_index", "step", "skill", "result_event", "candidate_id", "contract_passed"]:
                if key in record:
                    parts.append(f"{key}={record[key]}")
            return " ".join(parts)
        return f"[{event}] " + " ".join(f"{k}={v}" for k, v in record.items() if k not in {"t", "event"})

    def _build_parent_pool(self, population: list) -> list:
        pool = list(population)
        if self.qd_archive is not None:
            by_id = {candidate.id: candidate for candidate in pool}
            for candidate in self.qd_archive.elites():
                if candidate.id not in by_id:
                    pool.append(candidate)
                    by_id[candidate.id] = candidate
            return pool
        if not bool(self.config.get("evolution", {}).get("allow_archive_parents", False)):
            return pool
        by_id = {candidate.id: candidate for candidate in pool}
        stored = self.memory.candidate_store.all()
        candidates = []
        for data in stored.values():
            try:
                candidate = Candidate.from_dict(data)
            except TypeError:
                continue
            if candidate.objective is None or not candidate.code or candidate.id in by_id:
                continue
            candidates.append(candidate)
        family_best = {}
        for candidate in candidates:
            family = PopulationManager._family(candidate)
            current = family_best.get(family)
            if current is None or float(candidate.objective) < float(current.objective):
                family_best[family] = candidate
        for candidate in sorted(family_best.values(), key=lambda c: float(c.objective))[: max(4, len(population))]:
            pool.append(candidate)
            by_id[candidate.id] = candidate
        return pool

    def _build_route_parent_plan(self, population: list, offspring_total: int) -> tuple[list, list[dict]]:
        if self.route_manager is None or self.emitter_scheduler is None:
            raise RuntimeError("expert route parent planning requires RouteManager and EmitterScheduler")
        emitters = self.emitter_scheduler.select_emitters(offspring_total)
        parent_pool = []
        slot_metadata = []
        used_route_ids: set[str] = set()
        for emitter in emitters:
            parent, route = self.route_manager.parent_for_emitter(emitter, population, used_route_ids)
            used_route_ids.add(route.route_id)
            parent_pool.append(parent)
            slot_metadata.append(
                {
                    "emitter": emitter,
                    "route_id": route.route_id,
                    "route_signature": route.route_signature,
                    "route_status": route.status,
                    "route_reason": self._route_slot_reason(emitter, route),
                }
            )
        return parent_pool, slot_metadata

    def _build_qd_cell_parent_plan(self, offspring_total: int) -> tuple[list, list[dict]]:
        if self.qd_archive is None:
            raise RuntimeError("qd_cells parent planning requires QDArchive")
        cells = self.qd_archive.cells()
        if not cells:
            raise RuntimeError("qd_cells parent planning requires at least one archive cell")
        elites = self.qd_archive.elites()
        parent_pool = []
        slot_metadata = []
        for index in range(max(0, int(offspring_total))):
            parent = elites[index % len(elites)]
            descriptor = descriptor_for_candidate(parent, mode="mechanism")
            parent_pool.append(parent)
            slot_metadata.append(
                {
                    "qd_cell_descriptor": list(descriptor),
                    "qd_cell_elite_id": parent.id,
                    "qd_cell_objective": parent.objective,
                    "intent_options": ["exploit", "expand", "repair", "validate"],
                    "evidence_packet": self._qd_cell_evidence_packet(parent, descriptor),
                }
            )
        return parent_pool, slot_metadata

    @staticmethod
    def _qd_cell_evidence_packet(parent, descriptor: tuple) -> dict:
        evidence = parent.evidence if isinstance(getattr(parent, "evidence", None), dict) else {}
        diagnosis = evidence.get("training_diagnosis") if isinstance(evidence, dict) else {}
        if not isinstance(diagnosis, dict):
            diagnosis = {}
        return {
            "parent_id": parent.id,
            "parent_objective": parent.objective,
            "qd_cell_descriptor": list(descriptor),
            "hypothesis_verdict": parent.hypothesis_verdict,
            "failure_signature": diagnosis.get("failure_signature"),
            "worst_training_instances": (diagnosis.get("worst_training_instances") or [])[:3],
            "worsened_instances_vs_parent": (diagnosis.get("worsened_instances_vs_parent") or [])[:3],
        }

    def _record_emitter_failures(self, results: list[dict]) -> None:
        if self.emitter_scheduler is None:
            return
        for result in results:
            if result.get("status") == "evaluated":
                continue
            plan = self._plan_from_result(result)
            emitter = plan.get("emitter") if isinstance(plan, dict) else None
            self.emitter_scheduler.record(emitter, False)

    @staticmethod
    def _plan_from_result(result: dict) -> dict:
        candidate = result.get("candidate")
        metadata = getattr(candidate, "metadata", None)
        if isinstance(metadata, dict):
            plan = metadata.get("generation_plan") or metadata.get("planned_experiment")
            if isinstance(plan, dict):
                return plan
        for event in reversed(result.get("events", [])):
            plan = event.get("offspring_plan") if isinstance(event, dict) else None
            if isinstance(plan, dict):
                return plan
        return {}

    @staticmethod
    def _route_slot_reason(emitter: str, route) -> str:
        if emitter == "exploit_best_route":
            return "continue improving the current best route"
        if emitter == "behavior_archive_route":
            return "breed a behaviorally distinct archive route"
        if emitter == "failure_repair":
            return "repair the route's recorded worst training behavior"
        if emitter == "underexplored_route":
            return "give a low-trial route another controlled experiment"
        if emitter == "transfer_or_revive":
            return "transfer or revive a route with positive historical credit"
        return f"assigned emitter {emitter}"

    @staticmethod
    def _parent_for_plan(population: list, offspring_plan: dict):
        parent_id = offspring_plan.get("parent_id") if isinstance(offspring_plan, dict) else None
        for candidate in population:
            if candidate.id == parent_id:
                return candidate
        raise ValueError(f"offspring plan references unknown parent_id: {parent_id!r}")
