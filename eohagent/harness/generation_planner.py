from __future__ import annotations

from eohagent.core.gls_architecture import ARCHITECTURE_LAYERS
from eohagent.harness.agent_state import AgentState
from eohagent.memory.budget import bounded_mechanism_archive, bounded_notebook, short_hypothesis
from eohagent.problems.tsp_gls_arch_spec.spec_schema import (
    ARCH_SPEC_LAYERS,
    arch_spec_from_candidate,
    canonical_arch_spec,
    validate_single_layer_change,
)
from eohagent.problems.tsp_gls_solver_tree.tree_schema import (
    MUTATION_TYPES,
    canonical_solver_tree,
    solver_tree_from_candidate,
)


class GenerationPlanner:
    def __init__(self, config: dict, memory, json_call, selector):
        self.config = config
        self.memory = memory
        self.json_call = json_call
        self.selector = selector

    def plan(
        self,
        generation: int,
        population: list,
        offspring_total: int,
        parent_pool: list | None = None,
        slot_metadata: list[dict] | None = None,
    ) -> dict:
        parent_pool = parent_pool or population
        if not bool(self.config.get("agent", {}).get("enable_generation_planning", True)):
            raise RuntimeError("generation planning is required for this workflow")
        selected_parents = [self.selector.select_parent(parent_pool) for _ in range(offspring_total)]
        planner_state = self._planner_state(population, selected_parents)
        attempts = max(1, int(self.config.get("evolution", {}).get("max_llm_retries", 3)))
        protocol_errors = []
        for attempt in range(1, attempts + 1):
            try:
                data = self.json_call.call_dynamic(
                    lambda: self._plan_payload(
                        generation,
                        population,
                        parent_pool,
                        selected_parents,
                        planner_state,
                        protocol_errors,
                        slot_metadata,
                    ),
                    required=["generation_goal", "offspring_plans"],
                    context_state=planner_state,
                )
                plans = self._normalize_plans(data, selected_parents)
                self.memory.session_log.append(
                    {
                        "event": "generation_plan_llm_usage",
                        "generation": generation,
                        "llm_usage": getattr(self.json_call, "last_usage", {}) or {},
                        "prompt_compression": getattr(self.json_call, "last_prompt_compression", {}) or {},
                    }
                )
                return {
                    "generation_goal": str(data.get("generation_goal") or "")[:800],
                    "avoid_repeating": self._string_list(data.get("avoid_repeating"), limit=8),
                    "offspring_plans": plans,
                }
            except Exception as exc:
                protocol_errors.append(str(exc)[:500])
                self.memory.session_log.append(
                    {
                        "event": "generation_plan_retry",
                        "generation": generation,
                        "attempt": attempt,
                        "attempts": attempts,
                        "error": str(exc)[:500],
                        "llm_usage": getattr(self.json_call, "last_usage", {}) or {},
                        "prompt_compression": getattr(self.json_call, "last_prompt_compression", {}) or {},
                    }
                )
        error = protocol_errors[-1] if protocol_errors else "unknown planner protocol error"
        self.memory.session_log.append(
            {
                "event": "generation_plan_failed",
                "generation": generation,
                "error": error,
                "llm_usage": getattr(self.json_call, "last_usage", {}) or {},
                "prompt_compression": getattr(self.json_call, "last_prompt_compression", {}) or {},
            }
        )
        raise RuntimeError(f"generation planner failed protocol after {attempts} retries: {error}")

    def reflect(self, generation: int, population: list, offspring: list) -> dict:
        if not bool(self.config.get("agent", {}).get("enable_generation_reflection", True)):
            return {}
        if not offspring:
            raise RuntimeError("generation reflection requires at least one evaluated offspring")
        reflection_state = self._planner_state(population, population[:1] or offspring[:1])
        attempts = max(1, int(self.config.get("evolution", {}).get("max_llm_retries", 3)))
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                data = self.json_call.call_dynamic(
                    lambda: self._reflection_payload(generation, population, offspring, reflection_state),
                    required=["validated_principles", "failed_hypotheses", "open_questions"],
                    context_state=reflection_state,
                )
                self.memory.notebook.merge_update(data)
                self.memory.record_reflection_memory(generation, data)
                self.memory.session_log.append(
                    {
                        "event": "generation_reflection",
                        "generation": generation,
                        "validated_count": len(data.get("validated_principles") or []),
                        "failed_count": len(data.get("failed_hypotheses") or []),
                        "llm_usage": getattr(self.json_call, "last_usage", {}) or {},
                        "prompt_compression": getattr(self.json_call, "last_prompt_compression", {}) or {},
                    }
                )
                return data
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    self.memory.session_log.append(
                        {
                            "event": "generation_reflection_retry",
                            "generation": generation,
                            "attempt": attempt,
                            "attempts": attempts,
                            "error": str(exc)[:500],
                            "llm_usage": getattr(self.json_call, "last_usage", {}) or {},
                            "prompt_compression": getattr(self.json_call, "last_prompt_compression", {}) or {},
                        }
                    )
                    continue
        if last_error is not None:
            self.memory.session_log.append(
                {
                    "event": "generation_reflection_failed",
                    "generation": generation,
                    "error": str(last_error),
                    "llm_usage": getattr(self.json_call, "last_usage", {}) or {},
                    "prompt_compression": getattr(self.json_call, "last_prompt_compression", {}) or {},
                }
            )
            raise last_error
        raise RuntimeError("generation reflection failed without an error")

    def _plan_payload(
        self,
        generation: int,
        population: list,
        parent_pool: list,
        selected_parents: list,
        planner_state,
        protocol_errors: list[str] | None = None,
        slot_metadata: list[dict] | None = None,
    ) -> dict:
        family_champions = self._family_champions(parent_pool, limit=8)
        problem = self.config.get("project", {}).get("problem")
        if problem == "tsp_gls_arch_spec":
            plan_instruction = (
                "Design one generation of controlled GLS ARCH_SPEC experiments. "
                "Each offspring must include a complete child_spec. child_spec must differ from the assigned "
                "parent_arch_spec in exactly one layer. controlled_component must be one of: "
                f"{', '.join(ARCH_SPEC_LAYERS)}. "
                "Do not propose any direction listed in disabled_patterns. "
                "Use controlled experiment groups: variants in the same group should test different values of the "
                "same layer from the same parent or same base specification. "
            )
        elif problem == "tsp_gls_solver_tree":
            plan_instruction = (
                "Design one generation of controlled typed SOLVER_TREE experiments. "
                "Each offspring must include a complete child_solver_tree. The tree is executable: node order controls "
                "the solver loop, so do not describe pipeline changes without changing the tree. "
                "Each offspring must include mutation_type, controlled_node_path, controlled_change, "
                "expected_trace_change, experiment_intent, and evidence_refs. mutation_type must be one of: "
                f"{', '.join(sorted(MUTATION_TYPES))}. "
                "Allowed action nodes are local_search, penalty_step, restart; allowed control nodes are sequence, "
                "if_stagnated, and root repeat. Do not invent node types or free-form Python."
            )
        elif problem == "tsp_gls_arch":
            plan_instruction = (
                "Design one generation of controlled GLS architecture experiments. "
                "Each offspring must change exactly one architecture layer. "
                "controlled_component must be one of: distance_update, penalty_edge_selection, "
                "perturbation_schedule, restart_policy. "
                "The delta must state the layer-level architecture change, and preserve must state which other "
                "architecture layers remain unchanged. "
            )
        else:
            plan_instruction = "Design one generation of controlled heuristic experiments. "
        population_parent_ids = [candidate.id for candidate in population]
        allowed_parent_ids = [candidate.id for candidate in parent_pool]
        archive_parent_pool = [
            self._candidate_summary(candidate)
            for candidate in parent_pool
            if candidate.id not in set(population_parent_ids)
        ][:6]
        return {
            "purpose": "generation_experiment_plan",
            "instruction": (
                plan_instruction
                +
                "Return compact JSON with exactly one offspring_plans item for every required_offspring_slots item. "
                "The list length must equal offspring_total exactly; no fewer and no more. "
                "Each offspring_plans[i] must copy required_offspring_slots[i].offspring_index and parent_id exactly. "
                "Every parent_id must be one of allowed_parent_ids. "
                "Use shared group_id values to express controlled experiment groups, but do not return experiment_groups "
                "as the only plan format. Use training diagnostics, mechanism memory, and archive champions. "
                "Treat disabled_patterns as exact pattern-level constraints when present. "
                "Never generalize one failed combination into a broad ban on an entire component such as "
                "distance_update_policy.type, penalty_policy.edge_selection, or schedule_policy.strength_schedule. "
                "If a route slot includes an emitter and route_signature, design the experiment for that assigned "
                "research action while still changing exactly one controlled component. "
                "If a slot includes qd_cell_descriptor and evidence_packet, use those fields as the evidence base "
                "for an adaptive expert intent: exploit, expand, repair, or validate. "
                "When harness.require_experiment_intent is true, every offspring plan must include "
                "experiment_intent and non-empty evidence_refs naming the evidence used. "
                "Do not propose risk scores, validation-set weights, or weak probe penalties."
            ),
            "problem": self.config.get("project", {}).get("problem"),
            "function_name": self.config.get("evaluator", {}).get("function_name"),
            "architecture_class_name": self.config.get("evaluator", {}).get("architecture_class_name"),
            "architecture_open_layers": self.config.get("architecture", {}).get("open_layers", []),
            "arch_spec_schema": self._arch_spec_schema() if problem == "tsp_gls_arch_spec" else None,
            "solver_tree_schema": self._solver_tree_schema() if problem == "tsp_gls_solver_tree" else None,
            "required_output": {
                "generation_goal": "short mechanism-level goal for this generation",
                "avoid_repeating": ["failed or duplicate directions to avoid this generation"],
                "offspring_plans": [
                    {
                        "offspring_index": "must match the corresponding required_offspring_slots item",
                        "parent_id": "must match the corresponding required_offspring_slots item",
                        "group_id": "shared id for controlled experiment group",
                        "group_hypothesis": "mechanism hypothesis tested by this group",
                        "controlled_component": "single mechanism component or architecture layer changed by this offspring",
                        "controlled_change": "single field or subcomponent changed inside controlled_component",
                        "delta": "one concrete mechanism change to implement",
                        "preserve": "what parent mechanism should stay fixed",
                        "avoid": "what should not be repeated",
                        "expected_effect": "what evaluator outcome would support this delta",
                        "experiment_intent": "one of exploit, expand, repair, validate when required by harness",
                        "evidence_refs": ["short references to parent, qd_cell, memory, or diagnostic evidence"],
                        "child_spec": "required only for tsp_gls_arch_spec: complete ARCH_SPEC after applying delta",
                        "mutation_type": "required only for tsp_gls_solver_tree: micro_param, replace_subtree, insert_node, remove_node, reorder_sequence, or wrap_if_stagnated",
                        "controlled_node_path": (
                            "required only for tsp_gls_solver_tree: exact canonical path changed, "
                            "for example body.steps[1].penalty.top_k or body.steps[1]"
                        ),
                        "expected_trace_change": "required only for tsp_gls_solver_tree: expected execution-trace difference",
                        "child_solver_tree": "required only for tsp_gls_solver_tree: complete SOLVER_TREE after applying delta",
                    }
                ],
            },
            "generation": generation,
            "offspring_total": len(selected_parents),
            "allowed_parent_ids": allowed_parent_ids,
            "current_population_parent_ids": population_parent_ids,
            "selected_parent_ids": [parent.id for parent in selected_parents],
            "required_offspring_slots": [
                self._required_slot(
                    index + 1,
                    parent,
                    problem,
                    (slot_metadata or [{}])[index] if index < len(slot_metadata or []) else {},
                )
                for index, parent in enumerate(selected_parents)
            ],
            "previous_protocol_errors": protocol_errors or [],
            "disabled_patterns": self._disabled_patterns(),
            "population_champions": [self._candidate_summary(candidate) for candidate in population[:5]],
            "family_champions": [self._candidate_summary(candidate) for candidate in family_champions],
            "archive_parent_pool": archive_parent_pool,
            "training_diagnostics": self._training_diagnostics_summary(population[:8]),
            "retrieved_memories": self._retrieved_memories(selected_parents, slot_metadata),
            "local_tuning_evidence": self._local_tuning_evidence(),
            "search_feedback": self.memory._search_feedback(
                planner_state.population_summary,
                self.memory.session_log.recent(12),
            ),
            "expert_notebook": bounded_notebook(planner_state.notebook, items=3, text_limit=220),
            "mechanism_archive": bounded_mechanism_archive(
                planner_state.mechanism_archive,
                per_bucket=2,
                family_limit=6,
                text_limit=220,
            ),
            "compact_context": self._compact_context_summary(planner_state.compact_context),
        }

    def _reflection_payload(self, generation: int, population: list, offspring: list, reflection_state) -> dict:
        return {
            "purpose": "generation_end_reflection",
            "instruction": (
                "Summarize this generation as reusable mechanism-level design memory. "
                "Use controlled experiment groups when available. Attribute which deltas helped or hurt mean, tail, "
                "and failure signatures. Do not list every candidate. Do not mark broad components or mechanism "
                "families as refuted; refuted observations must name the exact parent context, controlled component, "
                "and delta that failed."
            ),
            "required_output": {
                "validated_principles": ["mechanism lesson supported by objective evidence"],
                "failed_hypotheses": ["mechanism change that worsened, duplicated, or failed"],
                "supported_deltas": ["controlled component + delta with supporting training evidence"],
                "refuted_deltas": ["controlled component + delta with negative training evidence"],
                "ambiguous_deltas": ["delta with mixed or insufficient evidence"],
                "failure_patterns": ["recurring training failure signature and related candidate ids"],
                "family_status": ["family crowding, stagnation, or promising archive direction"],
                "next_generation_priorities": ["specific next mechanism experiments"],
                "open_questions": ["next experiment direction"],
            },
            "generation": generation,
            "surviving_population": [self._reflection_candidate_summary(candidate) for candidate in population[:8]],
            "evaluated_offspring": [self._reflection_candidate_summary(candidate) for candidate in offspring[:12]],
            "controlled_experiment_results": self._controlled_experiment_results(offspring),
            "search_feedback": self.memory._search_feedback(
                reflection_state.population_summary,
                self.memory.session_log.recent(24),
            ),
            "recent_events": [
                self._reflection_event_summary(event)
                for event in self.memory.session_log.recent(40)
                if event.get("event") in {"formal_eval", "formal_eval_failed", "objective_duplicate", "candidate_duplicate", "candidate_rejected"}
                or event.get("contract_passed") is False
            ][-16:],
        }

    def _normalize_plans(self, data: dict, selected_parents: list) -> list[dict]:
        raw_plans = data.get("offspring_plans")
        if not isinstance(raw_plans, list):
            raise ValueError("generation planner did not return offspring_plans")
        if len(raw_plans) != len(selected_parents):
            raise ValueError(
                f"generation planner returned {len(raw_plans)} plans for {len(selected_parents)} offspring"
            )
        plans = []
        for index, raw in enumerate(raw_plans):
            if not isinstance(raw, dict):
                raise ValueError(f"offspring plan {index + 1} is not a JSON object")
            expected_parent_id = str(selected_parents[index].id)
            parent_id = str(raw.get("parent_id") or "").strip()
            if parent_id != expected_parent_id:
                raise ValueError(
                    f"offspring plan {index + 1} parent_id {parent_id!r} does not match required parent_id {expected_parent_id!r}"
                )
            if "offspring_index" not in raw:
                raise ValueError(f"offspring plan {index + 1} is missing required key: offspring_index")
            try:
                raw_offspring_index = int(raw["offspring_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"offspring plan {index + 1} has non-integer offspring_index") from exc
            if raw_offspring_index != index + 1:
                raise ValueError(f"offspring plan {index + 1} has mismatched offspring_index")
            problem = self.config.get("project", {}).get("problem")
            required_text_keys = ["parent_id", "delta"]
            if problem != "tsp_gls_solver_tree":
                required_text_keys.append("controlled_component")
            for key in required_text_keys:
                if not str(raw.get(key) or "").strip():
                    raise ValueError(f"offspring plan {index + 1} is missing required key: {key}")
            controlled_component = str(raw.get("controlled_component") or raw.get("mutation_type") or "").strip()
            child_spec = None
            child_solver_tree = None
            if problem == "tsp_gls_arch":
                if controlled_component not in ARCHITECTURE_LAYERS:
                    raise ValueError(
                        f"offspring plan {index + 1} controlled_component {controlled_component!r} "
                        "is not an allowed GLS architecture layer"
                    )
            elif problem == "tsp_gls_arch_spec":
                if controlled_component not in ARCH_SPEC_LAYERS:
                    raise ValueError(
                        f"offspring plan {index + 1} controlled_component {controlled_component!r} "
                        "is not an allowed GLS ARCH_SPEC layer"
                    )
                raw_child_spec = raw.get("child_spec")
                if not isinstance(raw_child_spec, dict):
                    raise ValueError(f"offspring plan {index + 1} is missing required child_spec")
                parent_spec = arch_spec_from_candidate(selected_parents[index])
                child_spec = canonical_arch_spec(raw_child_spec)
                validate_single_layer_change(parent_spec, child_spec, controlled_component)
            elif problem == "tsp_gls_solver_tree":
                for key in ["mutation_type", "controlled_node_path", "controlled_change", "expected_trace_change"]:
                    if not str(raw.get(key) or "").strip():
                        raise ValueError(f"offspring plan {index + 1} is missing required key: {key}")
                mutation_type = str(raw["mutation_type"]).strip()
                if mutation_type not in MUTATION_TYPES:
                    raise ValueError(f"offspring plan {index + 1} has invalid mutation_type: {mutation_type!r}")
                raw_child_tree = raw.get("child_solver_tree")
                if not isinstance(raw_child_tree, dict):
                    raise ValueError(f"offspring plan {index + 1} is missing required child_solver_tree")
                child_solver_tree = canonical_solver_tree(raw_child_tree)
            experiment_intent = ""
            evidence_refs = []
            if self._requires_experiment_intent():
                experiment_intent = str(raw.get("experiment_intent") or "").strip()
                allowed_intents = {"exploit", "expand", "repair", "validate"}
                if experiment_intent not in allowed_intents:
                    raise ValueError(
                        f"offspring plan {index + 1} has invalid or missing experiment_intent"
                    )
                evidence_refs = self._string_list(raw.get("evidence_refs"), limit=8)
                if not evidence_refs:
                    raise ValueError(f"offspring plan {index + 1} has empty evidence_refs")
            plan = {
                "offspring_index": index + 1,
                "parent_id": expected_parent_id[:120],
                "group_id": str(raw.get("group_id") or f"g{index + 1}"),
                "variant_id": str(raw.get("variant_id") or f"v{index + 1}"),
                "group_hypothesis": str(raw.get("group_hypothesis") or raw.get("hypothesis") or "")[:700],
                "controlled_component": controlled_component[:120],
                "controlled_change": str(raw.get("controlled_change") or "")[:120],
                "delta": str(raw["delta"])[:900],
                "preserve": str(raw.get("preserve") or "")[:500],
                "avoid": str(raw.get("avoid") or "")[:500],
                "expected_effect": str(raw.get("expected_effect") or "")[:500],
            }
            if problem == "tsp_gls_solver_tree":
                plan["mutation_type"] = str(raw["mutation_type"])[:120]
                plan["controlled_node_path"] = str(raw["controlled_node_path"])[:220]
                plan["expected_trace_change"] = str(raw["expected_trace_change"])[:500]
            if self._requires_experiment_intent():
                plan["experiment_intent"] = experiment_intent
                plan["evidence_refs"] = evidence_refs
            if child_spec is not None:
                plan["child_spec"] = child_spec
                plan["parent_arch_spec"] = arch_spec_from_candidate(selected_parents[index])
            if child_solver_tree is not None:
                plan["child_solver_tree"] = child_solver_tree
                plan["parent_solver_tree"] = solver_tree_from_candidate(selected_parents[index])
            plans.append(plan)
        return plans

    def _planner_state(self, population: list, selected_parents: list) -> AgentState:
        parent = selected_parents[0] if selected_parents else population[0]
        return AgentState(
            parent=parent,
            population_summary=self.memory.summarize_population(population),
            compact_context=self.memory.compact_context.read(),
            notebook=self.memory.notebook.read(),
            mechanism_archive=self.memory.mechanism_archive.read(),
        )

    def _compact_context_summary(self, context: dict | None = None) -> dict:
        context = context if isinstance(context, dict) else self.memory.compact_context.read()
        if not isinstance(context, dict):
            return {}
        working = context.get("working_memory") if isinstance(context.get("working_memory"), dict) else {}
        return {
            "current_best": context.get("current_best"),
            "recent_successes": context.get("recent_successes", [])[-3:],
            "recent_failures": context.get("recent_failures", [])[-3:],
            "next_focus": context.get("next_focus", [])[-3:],
            "mechanism_cards": context.get("mechanism_cards", [])[-3:],
            "delta_cards": context.get("delta_cards", [])[-3:],
            "generation_attributions": context.get("generation_attributions", [])[-3:],
            "working_memory": {
                "resume_summary": str(working.get("resume_summary") or "")[:600],
                "keep_rules": working.get("keep_rules", [])[-3:],
                "avoid_rules": working.get("avoid_rules", [])[-3:],
                "next_search_hint": str(working.get("next_search_hint") or "")[:300],
            },
        }

    @staticmethod
    def _family_champions(candidates: list, limit: int) -> list:
        from eohagent.core.population import PopulationManager

        champions = {}
        for candidate in candidates:
            if candidate.objective is None:
                continue
            family = PopulationManager._family(candidate)
            current = champions.get(family)
            if current is None or float(candidate.objective) < float(current.objective):
                champions[family] = candidate
        return sorted(champions.values(), key=lambda candidate: float(candidate.objective))[:limit]

    @staticmethod
    def _candidate_summary(candidate) -> dict:
        evidence = candidate.evidence or {}
        return {
            "id": candidate.id,
            "objective": candidate.objective,
            "mechanism_summary": str(candidate.mechanism_summary or "")[:260],
            "mechanism_hypothesis": short_hypothesis(candidate.mechanism_hypothesis),
            "mechanism_components": (candidate.mechanism_hypothesis or {}).get("mechanism_components")
            if isinstance(candidate.mechanism_hypothesis, dict)
            else None,
            "training_diagnosis": (candidate.evidence or {}).get("training_diagnosis")
            if isinstance(candidate.evidence, dict)
            else None,
            "generation_plan": GenerationPlanner._compact_plan((candidate.metadata or {}).get("generation_plan")),
            "hypothesis_verdict": candidate.hypothesis_verdict,
            "delta_vs_parent": evidence.get("delta_vs_parent") if isinstance(evidence, dict) else None,
            "arch_spec": (candidate.metadata or {}).get("arch_spec") if isinstance(candidate.metadata, dict) else None,
        }

    @staticmethod
    def _reflection_candidate_summary(candidate) -> dict:
        evidence = candidate.evidence or {}
        diagnosis = evidence.get("training_diagnosis") if isinstance(evidence, dict) else {}
        plan = (candidate.metadata or {}).get("generation_plan") if isinstance(candidate.metadata, dict) else {}
        return {
            "id": candidate.id,
            "objective": candidate.objective,
            "parent_id": (plan or {}).get("parent_id"),
            "group_id": (plan or {}).get("group_id"),
            "variant_id": (plan or {}).get("variant_id"),
            "controlled_component": (plan or {}).get("controlled_component"),
            "delta": str((plan or {}).get("delta") or "")[:260],
            "hypothesis_verdict": candidate.hypothesis_verdict,
            "mechanism_family": ((candidate.mechanism_hypothesis or {}).get("mechanism_family"))
            if isinstance(candidate.mechanism_hypothesis, dict)
            else None,
            "mean_delta_vs_parent": diagnosis.get("mean_delta_vs_parent") if isinstance(diagnosis, dict) else None,
            "tail_delta_vs_parent": diagnosis.get("tail_delta_vs_parent") if isinstance(diagnosis, dict) else None,
            "std_gap": diagnosis.get("std_gap") if isinstance(diagnosis, dict) else None,
            "tail_gap": diagnosis.get("tail_gap") if isinstance(diagnosis, dict) else None,
            "max_gap": diagnosis.get("max_gap") if isinstance(diagnosis, dict) else None,
            "failure_signature": diagnosis.get("failure_signature") if isinstance(diagnosis, dict) else None,
            "changed_layers": ((candidate.metadata or {}).get("spec_diff") or {}).get("changed_layers")
            if isinstance(candidate.metadata, dict)
            else None,
        }

    @staticmethod
    def _controlled_experiment_results(offspring: list) -> list[dict]:
        groups = {}
        for candidate in offspring:
            summary = GenerationPlanner._reflection_candidate_summary(candidate)
            group_id = summary.get("group_id") or "ungrouped"
            groups.setdefault(group_id, []).append(summary)
        return [
            {
                "group_id": group_id,
                "controlled_component": next(
                    (item.get("controlled_component") for item in items if item.get("controlled_component")),
                    None,
                ),
                "results": sorted(items, key=lambda item: float(item["objective"]) if item.get("objective") is not None else float("inf")),
            }
            for group_id, items in groups.items()
        ]

    @staticmethod
    def _training_diagnostics_summary(population: list) -> list[dict]:
        summary = []
        for candidate in population[:12]:
            evidence = candidate.evidence or {}
            diagnosis = evidence.get("training_diagnosis") if isinstance(evidence, dict) else {}
            if not isinstance(diagnosis, dict):
                diagnosis = {}
            summary.append(
                {
                    "id": candidate.id,
                    "objective": candidate.objective,
                    "mechanism_family": ((candidate.mechanism_hypothesis or {}).get("mechanism_family"))
                    if isinstance(candidate.mechanism_hypothesis, dict)
                    else None,
                    "std_gap": diagnosis.get("std_gap"),
                    "tail_gap": diagnosis.get("tail_gap"),
                    "failure_signature": diagnosis.get("failure_signature"),
                    "worst_training_instances": (diagnosis.get("worst_training_instances") or [])[:3],
                    "worsened_instances_vs_parent": (diagnosis.get("worsened_instances_vs_parent") or [])[:3],
                }
            )
        return summary

    @staticmethod
    def _event_summary(event: dict) -> dict:
        return {
            key: event.get(key)
            for key in [
                "event",
                "candidate_id",
                "objective",
                "selection_score",
                "hypothesis_verdict",
                "training_diagnosis",
                "generation_plan",
                "error",
                "errors",
            ]
            if event.get(key) is not None
        }

    @staticmethod
    def _reflection_event_summary(event: dict) -> dict:
        out = {
            key: event.get(key)
            for key in [
                "event",
                "candidate_id",
                "objective",
                "selection_score",
                "hypothesis_verdict",
                "error",
                "errors",
            ]
            if event.get(key) is not None
        }
        diagnosis = event.get("training_diagnosis")
        if isinstance(diagnosis, dict):
            out["training_diagnosis"] = {
                key: diagnosis.get(key)
                for key in [
                    "std_gap",
                    "tail_gap",
                    "max_gap",
                    "failure_signature",
                    "mean_delta_vs_parent",
                    "tail_delta_vs_parent",
                ]
                if diagnosis.get(key) is not None
            }
        plan = event.get("generation_plan")
        if isinstance(plan, dict):
            out["generation_plan"] = GenerationPlanner._compact_plan(plan)
        return out

    @staticmethod
    def _compact_plan(plan) -> dict | None:
        if not isinstance(plan, dict):
            return None
        return {
            key: plan.get(key)
            for key in [
                "offspring_index",
                "parent_id",
                "group_id",
                "variant_id",
                "controlled_component",
                "controlled_change",
                "delta",
                "expected_effect",
                "experiment_intent",
                "evidence_refs",
            ]
            if plan.get(key) is not None
        }

    def _disabled_patterns(self) -> list[dict]:
        patterns = getattr(self.memory, "disabled_patterns", []) or []
        if not isinstance(patterns, list):
            return []
        compact = []
        for pattern in patterns[:12]:
            if not isinstance(pattern, dict):
                continue
            compact.append(
                {
                    key: pattern.get(key)
                    for key in ["problem", "pattern_id", "changed_fields", "reason"]
                    if pattern.get(key) is not None
                }
            )
        return compact

    def _retrieved_memories(self, selected_parents: list, slot_metadata: list[dict] | None = None) -> list[dict]:
        query_parts = []
        for parent in selected_parents[:4]:
            query_parts.append(str(parent.id))
            metadata = parent.metadata if isinstance(getattr(parent, "metadata", None), dict) else {}
            spec = metadata.get("arch_spec")
            if isinstance(spec, dict):
                query_parts.extend(
                    [
                        str(spec.get("distance_update_policy", {}).get("type", "")),
                        str(spec.get("penalty_policy", {}).get("edge_selection", "")),
                        str(spec.get("schedule_policy", {}).get("strength_schedule", "")),
                    ]
                )
        for slot in (slot_metadata or [])[:4]:
            descriptor = slot.get("qd_cell_descriptor") if isinstance(slot, dict) else None
            if isinstance(descriptor, list):
                query_parts.extend(str(item) for item in descriptor)
        query = " ".join(part for part in query_parts if part)
        limit = int(self.config.get("memory", {}).get("planner_memory_items", 5))
        return self.memory.retrieve_memories(query, limit=limit)

    def _local_tuning_evidence(self) -> dict:
        component_counts = {}
        for event in self.memory.session_log.recent(80):
            plan = event.get("generation_plan")
            if not isinstance(plan, dict):
                plan = event.get("planned_experiment")
            if not isinstance(plan, dict):
                continue
            component = str(plan.get("controlled_component") or "").strip()
            if component:
                component_counts[component] = component_counts.get(component, 0) + 1
        return {"recent_controlled_component_counts": component_counts}

    def _requires_experiment_intent(self) -> bool:
        return bool(self.config.get("harness", {}).get("require_experiment_intent", False))

    @staticmethod
    def _string_list(value, limit: int) -> list[str]:
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            return []
        return [str(item)[:500] for item in value[:limit]]

    @staticmethod
    def _required_slot(index: int, parent, problem: str, slot_metadata: dict | None = None) -> dict:
        slot = {"offspring_index": index, "parent_id": parent.id}
        if isinstance(slot_metadata, dict):
            for key in [
                "emitter",
                "route_id",
                "route_signature",
                "route_status",
                "route_reason",
                "qd_cell_descriptor",
                "qd_cell_elite_id",
                "qd_cell_objective",
                "intent_options",
                "evidence_packet",
            ]:
                if key in slot_metadata:
                    slot[key] = slot_metadata[key]
        if problem == "tsp_gls_arch_spec":
            slot["parent_arch_spec"] = arch_spec_from_candidate(parent)
        elif problem == "tsp_gls_solver_tree":
            slot["parent_solver_tree"] = solver_tree_from_candidate(parent)
        return slot

    @staticmethod
    def _arch_spec_schema() -> dict:
        return {
            "layers": list(ARCH_SPEC_LAYERS),
            "rule": "child_spec must be complete and change exactly controlled_component versus parent_arch_spec",
            "distance_update_policy": {
                "type": [
                    "linear_usage",
                    "sqrt_usage",
                    "log_usage",
                    "length_weighted_usage",
                    "utility_ratio",
                    "rank_usage",
                    "smooth_saturation",
                ],
                "strength": "float in [0.0, 0.6]",
                "usage_power": "float in [0.0, 2.0]",
                "length_weight": "float in [0.0, 2.0]",
                "cap_multiplier": "float in [1.0, 20.0]",
            },
            "penalty_policy": {
                "edge_selection": ["top_gap", "tour_utility", "rank_topk", "mixed_gap_utility"],
                "top_k": "integer in [1, 20]",
                "increment": "float in [0.1, 5.0]",
                "decay": "float in [0.0, 0.5]",
            },
            "escape_policy": {
                "trigger": ["constant", "periodic", "stagnation", "size_adaptive"],
                "moves": "integer in [1, 20]",
                "period": "integer in [1, 1000]",
                "stagnation_threshold": "integer in [1, 1000]",
                "size_threshold": "integer in [1, 10000]",
                "max_moves": "integer in [moves, 20]",
            },
            "restart_policy": {
                "action": ["none", "reset_to_best", "reset_to_initial", "decay_penalty", "rebuild_nearest"],
                "trigger": ["never", "periodic", "stagnation"],
                "period": "integer in [1, 1000]",
                "stagnation_threshold": "integer in [1, 1000]",
                "penalty_decay": "float in [0.0, 1.0]",
            },
            "schedule_policy": {
                "strength_schedule": ["constant", "decay_after_early", "stagnation_boost", "size_scaled"],
                "early_iterations": "integer in [1, 1000]",
                "late_multiplier": "float in [0.05, 5.0]",
                "stagnation_multiplier": "float in [0.05, 5.0]",
                "size_threshold": "integer in [1, 10000]",
                "size_multiplier": "float in [0.05, 5.0]",
            },
        }

    @staticmethod
    def _solver_tree_schema() -> dict:
        return {
            "root": {"type": "repeat", "budget": "main_loop", "body": "sequence"},
            "limits": {
                "max_depth": 4,
                "max_sequence_steps": 5,
                "max_total_nodes": 14,
                "root_repeat_only": True,
            },
            "control_nodes": {
                "sequence": {"required": ["type", "steps"]},
                "if_stagnated": {"required": ["type", "threshold", "then"], "threshold": "integer in [1, 1000]"},
            },
            "action_nodes": {
                "local_search": {
                    "required": ["type", "neighborhood", "first_improvement"],
                    "neighborhood": ["two_opt_relocate"],
                    "first_improvement": "bool",
                },
                "penalty_step": {
                    "required": ["type", "repeat", "distance_update", "penalty", "schedule"],
                    "repeat": "integer in [1, 20]",
                    "distance_update.type": [
                        "linear_usage",
                        "sqrt_usage",
                        "log_usage",
                        "length_weighted_usage",
                        "utility_ratio",
                        "rank_usage",
                        "smooth_saturation",
                    ],
                    "penalty.edge_selection": ["top_gap", "tour_utility", "rank_topk", "mixed_gap_utility"],
                    "penalty.top_k": "integer in [1, 20]",
                    "schedule.strength_schedule": ["constant", "decay_after_early", "stagnation_boost", "size_scaled"],
                },
                "restart": {
                    "required": ["type", "action"],
                    "action": ["reset_to_best", "reset_to_initial", "decay_penalty", "rebuild_nearest"],
                    "penalty_decay": "required only when action is decay_penalty; float in [0.0, 1.0]",
                },
            },
            "mutation_types": sorted(MUTATION_TYPES),
            "path_format": "Use canonical dot/list paths such as body.steps[1].penalty.edge_selection, not JSON Pointer paths.",
        }
