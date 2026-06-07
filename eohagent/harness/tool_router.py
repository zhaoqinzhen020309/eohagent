from __future__ import annotations

import ast
import re
import types
from itertools import count

import numpy as np

from eohagent.core.candidate import Candidate
from eohagent.core.gls_architecture import ARCHITECTURE_CLASS_NAME, validate_architecture_code
from eohagent.core.mechanism import MechanismHypothesis, canonicalize_family, infer_mechanism_components
from eohagent.harness.context_runtime import ContextOverflowError
from eohagent.memory.budget import bounded_mechanism_archive, bounded_notebook, short_hypothesis
from eohagent.problems.tsp_gls_arch_spec.spec_schema import build_arch_spec_code, canonical_arch_spec
from eohagent.problems.tsp_gls_solver_tree.tree_schema import build_solver_tree_code, canonical_solver_tree


class ToolRouter:
    def __init__(self, config: dict, json_call, evaluator=None):
        self.config = config
        self.json_call = json_call
        self.evaluator = evaluator
        self._ids = count(1)

    def execute(self, skill_name: str, skill_doc: str, decision: dict, state) -> dict:
        if skill_name == "mechanism_decomposition":
            return self._llm_skill(skill_name, skill_doc, decision, state, ["dominant_mechanism"])
        if skill_name == "controlled_variation":
            return self._controlled_variation(state, skill_doc, decision)
        if skill_name == "contract_check":
            return self._contract_check(state)
        if skill_name == "repair":
            return {
                "event": "repair_failed",
                "errors": ["repair skill is disabled; failed candidates must be rejected without repair"],
            }
        if skill_name == "probe_synthesis":
            return self._probe_synthesis(state, skill_doc, decision)
        if skill_name == "context_compaction":
            return {"event": "context_compaction", "summary": "compaction requested"}
        if skill_name == "notebook_update":
            return self._llm_skill(skill_name, skill_doc, decision, state, [])
        raise ValueError(f"Unsupported skill: {skill_name}")

    def _llm_skill(self, skill_name: str, skill_doc: str, decision: dict, state, required: list[str]) -> dict:
        try:
            data = self.json_call.call_dynamic(
                lambda: {
                    "purpose": skill_name,
                    "skill_instructions": skill_doc,
                    "agent_decision": self._decision_payload(decision),
                    "state": self._state_payload(state),
                    "required_json_keys": required,
                },
                required=required,
                context_state=state,
            )
        except ContextOverflowError:
            raise
        except Exception as exc:
            raise RuntimeError(f"{skill_name} LLM call failed: {exc}") from exc
        return {"event": skill_name, "output": data}

    def _controlled_variation(self, state, skill_doc: str, decision: dict) -> dict:
        output_instruction = self._candidate_output_instruction()
        response = self._call_code_llm(
            purpose="controlled_variation",
            skill_doc=skill_doc,
            state=state,
            decision=decision,
            extra_instruction=(
                "Create one controlled variation following agent_decision.arguments as the experiment plan. "
                "Respect design_mode, evidence_used, preserve, change, avoid, and expected_test when present. "
                "Keep supported mechanisms recognizable; explore only when the decision gives evidence. "
                "First describe the mechanism hypothesis inside braces using labels: "
                "Hypothesis, Mechanism family, Parent delta, Expected effect, Preserve, Change. "
                f"Then provide {output_instruction}. "
                "Do not use JSON."
            ),
        )
        data = self._extract_candidate_from_text(
            response,
            reason="EoH-style controlled variation extraction",
        )
        if data["candidate_code"] is None:
            return {
                "event": "candidate_generation_failed",
                "errors": ["LLM response did not contain extractable candidate code"],
                "raw_response": response[:1000],
            }
        data["mechanism_hypothesis"].parent_id = state.parent.id
        planned_experiment = self._planned_experiment(decision)
        candidate = Candidate(
            id=f"h_{next(self._ids):04d}",
            parents=[state.parent.id],
            algorithm=data["candidate_algorithm"],
            code=data["candidate_code"],
            mechanism_summary=data["mechanism_hypothesis"].short_summary(),
            mechanism_hypothesis=data["mechanism_hypothesis"].to_dict(),
            mechanism_profile=data["mechanism_profile"],
            parent_delta=data["mechanism_hypothesis"].parent_delta,
            generation=int(state.parent.generation) + 1,
            metadata={
                "variation": self._serializable_extraction(data),
                "planned_experiment": planned_experiment,
            },
        )
        candidate.mechanism_hypothesis["id"] = f"mh_{candidate.id}"
        candidate.metadata["variation"]["mechanism_hypothesis"]["id"] = candidate.mechanism_hypothesis["id"]
        return {"event": "candidate_generated", "candidate": candidate}

    def _contract_check(self, state) -> dict:
        candidate = state.current_candidate
        if candidate is None:
            return {"event": "contract_check", "contract_passed": False, "errors": ["no current candidate"]}
        errors, warnings = self._check_code(candidate.code)
        return {
            "event": "contract_check",
            "candidate_id": candidate.id,
            "contract_passed": not errors,
            "errors": errors,
            "warnings": warnings,
        }

    def _repair(self, state, skill_doc: str, decision: dict) -> dict:
        if state.current_candidate is None:
            return {"event": "repair_failed", "errors": ["no current candidate to repair"]}
        original_candidate = state.current_candidate
        output_instruction = self._candidate_output_instruction()
        response = self._call_code_llm(
            purpose="repair",
            skill_doc=skill_doc,
            state=state,
            decision=decision,
            extra_instruction=(
                "Repair the current candidate so it satisfies the exact evaluator contract. "
                "Preserve the candidate's intended mechanism when possible. "
                "If the only safe output is identical to the parent code, do not output parent code; explain failure instead. "
                "First describe the repaired mechanism hypothesis inside braces using labels: "
                "Hypothesis, Mechanism family, Parent delta, Expected effect, Preserve, Change. "
                f"Then provide {output_instruction}. Do not use JSON."
            ),
        )
        extracted = self._extract_candidate_from_text(
            response,
            reason="EoH-style repair extraction",
        )
        if extracted["candidate_code"] is None:
            return {
                "event": "repair_failed",
                "errors": ["LLM response did not contain extractable repaired code"],
                "raw_response": response[:1000],
            }
        if self._same_code(extracted["candidate_code"], state.parent.code):
            return {
                "event": "repair_failed",
                "errors": ["repair reverted to parent code; failed innovations must not enter the population"],
                "raw_response": response[:1000],
            }
        planned_experiment = self._planned_experiment(decision) or (
            original_candidate.metadata.get("planned_experiment")
            if isinstance(original_candidate.metadata, dict)
            else {}
        )
        deviation = self._repair_plan_deviation(original_candidate, extracted, planned_experiment)
        if deviation:
            return {
                "event": "repair_failed",
                "errors": [deviation],
                "raw_response": response[:1000],
            }
        extracted["mechanism_hypothesis"].parent_id = state.parent.id
        data = {
            "repair_reason": extracted["mechanism_hypothesis"].short_summary(),
            "preserved_mechanism": extracted["mechanism_hypothesis"].short_summary(),
            "changed_parts": ["extracted Python code from LLM response"],
            "repaired_algorithm": extracted["candidate_algorithm"],
            "repaired_code": extracted["candidate_code"],
            "expected_fix": "restore evaluator-compatible code",
        }
        candidate = Candidate(
            id=f"h_{next(self._ids):04d}",
            parents=[state.parent.id],
            algorithm=data["repaired_algorithm"],
            code=data["repaired_code"],
            mechanism_summary=extracted["mechanism_hypothesis"].short_summary(),
            mechanism_hypothesis=extracted["mechanism_hypothesis"].to_dict(),
            mechanism_profile=extracted["mechanism_profile"],
            parent_delta=extracted["mechanism_hypothesis"].parent_delta,
            generation=int(state.parent.generation) + 1,
            metadata={
                "repair": data,
                "repair_extraction": self._serializable_extraction(extracted),
                "planned_experiment": planned_experiment,
                "repair_preserved_plan": True,
            },
        )
        candidate.mechanism_hypothesis["id"] = f"mh_{candidate.id}"
        candidate.metadata["repair_extraction"]["mechanism_hypothesis"]["id"] = candidate.mechanism_hypothesis["id"]
        return {"event": "repair", "candidate": candidate}

    def _probe_synthesis(self, state, skill_doc: str, decision: dict | None = None) -> dict:
        problem = self.config["project"]["problem"]
        candidate = state.current_candidate
        if problem == "bp_online":
            probes = [
                {
                    "probe_name": "scaled_capacity_like_bins",
                    "transformation": "scale item and feasible bins by the same positive factor",
                    "expected_property": "relative bin ranking should remain stable for scale-free terms",
                    "cost": "low",
                    "reason": "detects brittle absolute thresholds in score functions",
                }
            ]
        elif problem == "tsp_construct":
            probes = [
                {
                    "probe_name": "distance_scale_invariance",
                    "transformation": "multiply all distances by a positive constant",
                    "expected_property": "nearest-node decisions should remain unchanged",
                    "cost": "low",
                    "reason": "detects non-normalized distance thresholds",
                }
            ]
        elif problem == "tsp_gls":
            return self._run_tsp_gls_probes(candidate, skill_doc, decision or {})
        else:
            probes = []
        return {
            "event": "probe_synthesis",
            "skill_instructions_used": bool(skill_doc),
            "probes": probes,
            "probe_results": probes,
            "probe_penalty": 0.0,
            "probe_evidence": {"executed": False, "reason": "descriptive probes only for this problem"},
        }

    def _run_tsp_gls_probes(self, candidate, skill_doc: str, decision: dict) -> dict:
        if candidate is None:
            return {
                "event": "probe_synthesis",
                "skill_instructions_used": bool(skill_doc),
                "probe_results": [],
                "probe_penalty": 1.0,
                "probe_evidence": {"executed": False, "error": "no current candidate"},
            }
        module = types.ModuleType("probe_candidate_module")
        try:
            exec(candidate.code, module.__dict__)
            fn = getattr(module, "update_edge_distance")
        except Exception as exc:
            return {
                "event": "probe_synthesis",
                "candidate_id": candidate.id,
                "skill_instructions_used": bool(skill_doc),
                "probe_results": [{"name": "load_candidate", "passed": False, "detail": str(exc)}],
                "probe_penalty": 1.0,
                "probe_evidence": {"executed": True, "error": str(exc)},
            }

        focus = self._probe_focuses(candidate, decision)
        edge_distance = np.array(
            [
                [0.0, 1.0, 2.0, 3.0],
                [1.0, 0.0, 4.0, 5.0],
                [2.0, 4.0, 0.0, 6.0],
                [3.0, 5.0, 6.0, 0.0],
            ],
            dtype=float,
        )
        local_opt_tour = np.array([0, 1, 2, 3], dtype=int)
        edge_n_used = np.zeros_like(edge_distance)
        edge_n_used[0, 1] = edge_n_used[1, 0] = 2.0
        edge_n_used[1, 2] = edge_n_used[2, 1] = 5.0
        results = []

        base_input = edge_distance.copy()
        try:
            base = np.asarray(fn(base_input, local_opt_tour, edge_n_used), dtype=float)
            results.append(self._probe_result("finite_nonnegative", bool(np.all(np.isfinite(base)) and np.all(base >= 0)), "updated matrix should be finite and non-negative"))
            results.append(self._probe_result("symmetry_preservation", bool(np.allclose(base, base.T, rtol=1e-8, atol=1e-8)), "GLS edge distance update should remain symmetric"))
            results.append(self._probe_result("no_input_mutation", bool(np.allclose(base_input, edge_distance, rtol=1e-12, atol=1e-12)), "candidate should return a copy rather than mutate edge_distance in place"))

            if "scale_invariance" in focus:
                scaled = np.asarray(fn(edge_distance * 7.0, local_opt_tour, edge_n_used), dtype=float)
                results.append(
                    self._probe_result(
                        "scale_invariance",
                        bool(np.allclose(scaled / 7.0, base, rtol=1e-6, atol=1e-6)),
                        "multiplying all edge distances should preserve relative GLS penalty behavior",
                    )
                )

            if "usage_monotonicity" in focus:
                low_usage = edge_n_used.copy()
                high_usage = edge_n_used.copy()
                low_usage[0, 1] = low_usage[1, 0] = 1.0
                high_usage[0, 1] = high_usage[1, 0] = 8.0
                low = np.asarray(fn(edge_distance.copy(), local_opt_tour, low_usage), dtype=float)
                high = np.asarray(fn(edge_distance.copy(), local_opt_tour, high_usage), dtype=float)
                low_ratio = low[0, 1] / edge_distance[0, 1]
                high_ratio = high[0, 1] / edge_distance[0, 1]
                results.append(
                    self._probe_result(
                        "usage_monotonicity",
                        bool(high_ratio + 1e-12 >= low_ratio),
                        "higher edge usage should not reduce the penalty on that edge",
                    )
                )
            if "zero_usage_bounded" in focus:
                zero = np.asarray(fn(edge_distance.copy(), local_opt_tour, np.zeros_like(edge_distance)), dtype=float)
                ratios = zero / np.maximum(edge_distance, 1e-12)
                tour_ratios = [ratios[int(local_opt_tour[i]), int(local_opt_tour[(i + 1) % len(local_opt_tour)])] for i in range(len(local_opt_tour))]
                results.append(
                    self._probe_result(
                        "zero_usage_bounded",
                        bool(np.all(np.isfinite(tour_ratios)) and min(tour_ratios) >= 1.0 - 1e-9 and max(tour_ratios) < 3.0),
                        "zero-usage tour edges should not be decreased or inflated excessively",
                    )
                )
            if "high_usage_saturation" in focus:
                moderate_usage = edge_n_used.copy()
                extreme_usage = edge_n_used.copy()
                moderate_usage[0, 1] = moderate_usage[1, 0] = 10.0
                extreme_usage[0, 1] = extreme_usage[1, 0] = 100.0
                moderate = np.asarray(fn(edge_distance.copy(), local_opt_tour, moderate_usage), dtype=float)
                extreme = np.asarray(fn(edge_distance.copy(), local_opt_tour, extreme_usage), dtype=float)
                moderate_ratio = moderate[0, 1] / edge_distance[0, 1]
                extreme_ratio = extreme[0, 1] / edge_distance[0, 1]
                results.append(
                    self._probe_result(
                        "high_usage_saturation",
                        bool(np.isfinite(extreme_ratio) and extreme_ratio >= moderate_ratio and extreme_ratio / max(moderate_ratio, 1e-12) < 25.0),
                        "extreme usage should increase penalties monotonically without explosive growth",
                    )
                )
            if "length_sensitivity" in focus:
                equal_usage = np.ones_like(edge_distance)
                length_base = np.asarray(fn(edge_distance.copy(), local_opt_tour, equal_usage), dtype=float)
                short_ratio = length_base[0, 1] / edge_distance[0, 1]
                long_ratio = length_base[2, 3] / edge_distance[2, 3]
                results.append(
                    self._probe_result(
                        "length_sensitivity",
                        bool(np.isfinite(short_ratio) and np.isfinite(long_ratio) and long_ratio + 1e-12 >= short_ratio),
                        "length-aware mechanisms should penalize longer equally-used tour edges at least as much as shorter ones",
                    )
                )
            max_ratio = float(np.max(base / np.maximum(edge_distance, 1e-12)))
            results.append(
                self._probe_result(
                    "bounded_penalty",
                    bool(np.isfinite(max_ratio) and max_ratio < 1e6),
                    f"maximum multiplicative penalty should stay bounded, observed {max_ratio:.4g}",
                )
            )
        except Exception as exc:
            results.append(self._probe_result("probe_runtime", False, f"probe execution failed: {exc}"))

        failed = [item for item in results if not item["passed"]]
        penalty = round(len(failed) / max(1, len(results)), 4)
        return {
            "event": "probe_synthesis",
            "candidate_id": candidate.id,
            "skill_instructions_used": bool(skill_doc),
            "probe_results": results,
            "probe_penalty": penalty,
            "probe_evidence": {
                "executed": True,
                "focus": focus,
                "failed_count": len(failed),
                "total_count": len(results),
                "failed_probe_names": [item["name"] for item in failed],
            },
        }

    @staticmethod
    def _probe_focuses(candidate, decision: dict) -> list[str]:
        requested = []
        args = decision.get("arguments") if isinstance(decision, dict) else {}
        if isinstance(args, dict):
            raw = args.get("probe_focus") or args.get("probe_focuses") or args.get("focus")
            if isinstance(raw, str):
                requested = [raw]
            elif isinstance(raw, list):
                requested = [str(item) for item in raw]
        hypothesis = getattr(candidate, "mechanism_hypothesis", None) or {}
        family = canonicalize_family(hypothesis.get("mechanism_family") if isinstance(hypothesis, dict) else None, getattr(candidate, "mechanism_summary", ""))
        focus = ["scale_invariance", "usage_monotonicity", "zero_usage_bounded", "high_usage_saturation"]
        if family in {"length_weighted_penalty", "utility_ratio_penalty"}:
            focus.append("length_sensitivity")
        for item in requested:
            normalized = str(item).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized and normalized not in focus:
                focus.append(normalized)
        return focus

    @staticmethod
    def _probe_result(name: str, passed: bool, detail: str) -> dict:
        return {"name": name, "passed": passed, "detail": detail}

    def _check_code(self, code: str) -> tuple[list[str], list[str]]:
        errors = []
        warnings = []
        if not isinstance(code, str):
            return [f"type_error: code must be a string, got {type(code).__name__}"], warnings
        if not code.strip():
            return ["empty_code_error: code is empty"], warnings
        problem = self.config["project"]["problem"]
        func_name = self.config.get("evaluator", {}).get("function_name")
        if not func_name:
            if problem == "bp_online":
                func_name = "score"
            elif problem == "tsp_gls":
                func_name = "update_edge_distance"
            else:
                func_name = "select_next_node"
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return [f"syntax_error: {exc}"], warnings
        if problem == "tsp_gls_solver_tree":
            return self._check_solver_tree_code(code)
        if problem == "tsp_gls_arch_spec":
            return self._check_arch_spec_code(code)
        if problem == "tsp_gls_arch":
            return validate_architecture_code(code)
        func_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == func_name]
        if not func_nodes:
            errors.append(f"signature_error: missing function {func_name}")
            return errors, warnings
        module = types.ModuleType("candidate_module")
        try:
            exec(code, module.__dict__)
        except Exception as exc:
            errors.append(f"runtime_error: exec failed: {exc}")
            return errors, warnings
        try:
            if problem == "bp_online":
                result = module.score(5.0, np.array([5.0, 6.0, 10.0]))
                arr = np.asarray(result, dtype=float)
                if arr.shape != (3,):
                    errors.append(f"output_shape_error: expected shape (3,), got {arr.shape}")
                if not np.all(np.isfinite(arr)):
                    errors.append("nan_inf_error: score output contains non-finite values")
            elif problem == "tsp_construct":
                dist = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]])
                unvisited = np.array([1, 2])
                result = module.select_next_node(0, 0, unvisited, dist)
                if int(result) not in {1, 2}:
                    errors.append(f"output_value_error: next node {result} not in unvisited nodes")
            elif problem == "tsp_gls":
                edge_distance = np.array(
                    [
                        [0.0, 1.0, 2.0, 3.0],
                        [1.0, 0.0, 4.0, 5.0],
                        [2.0, 4.0, 0.0, 6.0],
                        [3.0, 5.0, 6.0, 0.0],
                    ],
                    dtype=float,
                )
                original_edge_distance = edge_distance.copy()
                local_opt_tour = np.array([0, 1, 2, 3], dtype=int)
                edge_n_used = np.zeros_like(edge_distance)
                edge_n_used[0, 1] = edge_n_used[1, 0] = 2.0
                edge_n_used[1, 2] = edge_n_used[2, 1] = 3.0
                result = module.update_edge_distance(edge_distance, local_opt_tour, edge_n_used)
                arr = np.asarray(result, dtype=float)
                if arr.shape != edge_distance.shape:
                    errors.append(f"output_shape_error: expected shape {edge_distance.shape}, got {arr.shape}")
                if not np.all(np.isfinite(arr)):
                    errors.append("nan_inf_error: update_edge_distance output contains non-finite values")
                if np.any(arr < 0):
                    errors.append("negative_distance_error: updated edge distances contain negative values")
                if not np.allclose(arr, arr.T, rtol=1e-8, atol=1e-8):
                    errors.append("symmetry_error: updated edge distance matrix is not symmetric")
                if not np.allclose(edge_distance, original_edge_distance, rtol=1e-12, atol=1e-12):
                    errors.append("in_place_error: candidate modified edge_distance input in place")
            else:
                warnings.append("custom problem has syntax/signature checks only")
        except Exception as exc:
            errors.append(f"runtime_error: dummy call failed: {exc}")
        return errors, warnings

    def _call_code_llm(self, purpose: str, skill_doc: str, state, decision: dict, extra_instruction: str) -> str:
        problem_card = state.compact_context.get("problem_card", {})
        function_name = self._expected_symbol_name(problem_card)
        agent_decision = self._decision_payload(decision)
        if isinstance(agent_decision.get("arguments"), dict) and agent_decision["arguments"].get("offspring_plan"):
            system, build_user = self._build_planned_code_prompt(
                purpose,
                function_name,
                problem_card,
                state,
                agent_decision,
                extra_instruction,
            )
        else:
            system = (
                "You are improving a heuristic for an automated EoH-style experiment.\n"
                "Do not return JSON. Return a short algorithm description in braces, then Python code.\n"
                f"Purpose: {purpose}\n"
                f"Required code symbol: {function_name}\n"
                f"Problem-specific instruction:\n{self._problem_specific_instruction()}\n\n"
                f"Problem card: {problem_card}\n"
                f"Skill instructions:\n{skill_doc}\n\n"
                f"Instruction:\n{extra_instruction}\n"
                "The agent decision is binding guidance for search intent. "
                "Do not ignore requested avoid/try mechanisms or variation_mode arguments.\n"
                "The code must define the required symbol and must be executable by ast.parse."
            )
            build_user = lambda: (
                f"Agent decision reason:\n{agent_decision['reason']}\n\n"
                f"Agent decision arguments:\n{agent_decision['arguments']}\n\n"
                f"Working memory:\n{self._working_context(state)}\n\n"
                f"Population summary:\n{self._prompt_population_summary(state)}\n\n"
                f"Expert notebook summary:\n{self._prompt_notebook_summary(state)}\n\n"
                f"Mechanism archive summary:\n{self._prompt_mechanism_archive(state)}\n\n"
                f"Parent algorithm:\n{state.parent.algorithm}\n\n"
                f"Parent code:\n{state.parent.code}\n\n"
                f"Current candidate:\n{self._prompt_candidate_summary(state.current_candidate)}\n\n"
                f"Last result:\n{self._prompt_last_result(state)}"
            )
        return self.json_call.complete_text_dynamic(
            build_user,
            purpose=purpose,
            context_state=state,
            system=system,
        )

    def _build_planned_code_prompt(self, purpose: str, function_name: str, problem_card: dict, state, agent_decision: dict, extra_instruction: str):
        plan = agent_decision["arguments"].get("offspring_plan", {})
        memory = {
            "current_best": self._working_context(state).get("current_best"),
            "notebook": self._prompt_notebook_summary(state),
            "archive": self._prompt_mechanism_archive(state),
            "parent_training_diagnosis": (state.parent.evidence or {}).get("training_diagnosis")
            if isinstance(state.parent.evidence, dict)
            else {},
        }
        system = (
            "You are implementing one planned heuristic experiment.\n"
            "Do not return JSON. Return a short mechanism description in braces, then Python code.\n"
            f"Purpose: {purpose}\n"
            f"Required code symbol: {function_name}\n"
            f"Problem-specific instruction:\n{self._problem_specific_instruction()}\n\n"
            f"Problem card: {problem_card}\n"
            f"Instruction:\n{extra_instruction}\n"
            "Implement only the requested parent+delta. Preserve the parent mechanism where the plan says preserve. "
            "The controlled_component field identifies the one mechanism component or architecture layer this variant should change. "
            "Do not add unrelated terms, weak probe logic, risk scoring, or a constructive TSP selector."
        )
        def build_user():
            return (
                f"Generation offspring plan: {plan}\n"
                f"Compact mechanism memory: {memory}\n"
                f"Parent algorithm:\n{state.parent.algorithm}\n\n"
                f"Parent code:\n{state.parent.code}\n\n"
                f"Current candidate:\n{self._prompt_candidate_summary(state.current_candidate)}\n\n"
                f"Last result:\n{self._prompt_last_result(state)}"
            )
        return system, build_user

    def _expected_symbol_name(self, problem_card: dict) -> str:
        if self.config["project"]["problem"] in {"tsp_gls_arch", "tsp_gls_arch_spec"}:
            return self.config.get("evaluator", {}).get("architecture_class_name", ARCHITECTURE_CLASS_NAME)
        return self.config.get("evaluator", {}).get(
            "function_name",
            problem_card.get("expected_function_name", "heuristic"),
        )

    def _problem_specific_instruction(self) -> str:
        problem = self.config["project"]["problem"]
        if problem == "tsp_gls_arch_spec":
            return (
                "This is spec-driven architecture-level design for TSP guided local search. "
                "The generation planner must provide a complete ARCH_SPEC child_spec. "
                "Candidate code must be the deterministic wrapper generated from ARCH_SPEC; do not hand-write "
                "CandidateGLSArchitecture methods. The controlled_component field is the only ARCH_SPEC layer "
                "allowed to differ from the parent. Open layers are distance_update_policy, penalty_policy, "
                "escape_policy, restart_policy, and schedule_policy."
            )
        if problem == "tsp_gls_arch":
            return (
                "This is constrained architecture-level design for TSP guided local search. "
                "The code must define class CandidateGLSArchitecture with exactly these callable methods: "
                "update_edge_distance(edge_distance, local_opt_tour, edge_n_used), "
                "select_penalty_edges(edge_weight, edge_weight_guided, edge_weight_gap, local_opt_tour, best_tour, edge_penalty, iteration), "
                "perturbation_schedule(iteration, n_nodes, base_perturbation_moves), and "
                "restart_policy(iteration, current_cost, best_cost, stagnation_count). "
                "The generation plan's controlled_component is the only architecture layer that may change. "
                "All other layers must remain recognizable from the parent. Do not rewrite the TSP solver, "
                "local search operators, evaluator, data loading, or final testing protocol. "
                "select_penalty_edges must return 1 to 20 undirected non-duplicate node-index pairs. "
                "perturbation_schedule must return an integer in [1, 20], and restart_policy must return a boolean."
            )
        if problem == "tsp_gls":
            return (
                "This is the original EoH TSP-GLS task. The heuristic must update the edge distance matrix used by "
                "guided local search. It must define update_edge_distance(edge_distance, local_opt_tour, edge_n_used) "
                "and return updated_edge_distance. Do not write a constructive next-node selector. Use only the current "
                "edge_distance matrix, the local_opt_tour, and edge_n_used. Candidate ideas should be explicit GLS "
                "mechanism hypotheses, for example sublinear usage penalties, capped penalties, top-k edge targeting, "
                "length-aware penalties, GLS utility ratios, size-adaptive strength, mean-usage normalization, or smooth saturation."
            )
        if problem == "tsp_construct":
            return (
                "This is a constructive TSP task. The heuristic must choose the next unvisited node with "
                "select_next_node(current_node, destination_node, univisited_nodes, distance_matrix)."
            )
        if problem == "bp_online":
            return "This is online bin packing. The heuristic must score feasible bins for the current item only."
        return "Follow the problem card exactly."

    def _candidate_output_instruction(self) -> str:
        if self.config["project"]["problem"] == "tsp_gls_arch_spec":
            return (
                "Python code containing ARCH_SPEC and the deterministic CandidateGLSArchitecture wrapper"
            )
        if self.config["project"]["problem"] in {"tsp_gls_arch", "tsp_gls_arch_spec"}:
            return (
                "Python code defining the complete CandidateGLSArchitecture class with all four required methods"
            )
        return "Python code only for the required heuristic function"

    def _extract_candidate_from_text(
        self,
        response: str,
        reason: str,
    ) -> dict:
        code = self._extract_code_like_eoh(response)
        if not isinstance(code, str):
            code = None
        algorithm = self._extract_algorithm_like_eoh(response) or "Candidate extracted from LLM response."
        hypothesis = MechanismHypothesis.from_description(
            algorithm,
            parent_id=None,
            problem=self.config["project"]["problem"],
        )
        hypothesis.mechanism_family = canonicalize_family(hypothesis.mechanism_family, algorithm)
        hypothesis.mechanism_components = infer_mechanism_components(hypothesis.mechanism_family, algorithm, code)
        return {
            "extraction_reason": f"{reason}; used EoH-style code extraction.",
            "mechanism_hypothesis": hypothesis,
            "mechanism_profile": self._mechanism_profile(hypothesis, code),
            "preserve": ["function signature", "evaluator compatibility"],
            "modify": ["LLM proposed code when extractable"],
            "remove": [],
            "expected_effect": "continue experiment without requiring strict JSON",
            "possible_side_effect": "non-json extraction may be less structured",
            "candidate_algorithm": algorithm,
            "candidate_code": code,
        }

    @staticmethod
    def _mechanism_profile(hypothesis: MechanismHypothesis, code: str | None) -> dict:
        code_text = code or ""
        return {
            "mechanism_family": hypothesis.mechanism_family,
            "hypothesis_text": hypothesis.text,
            "parent_delta": hypothesis.parent_delta,
            "expected_effect": hypothesis.expected_effect,
            "mechanism_components": hypothesis.mechanism_components
            or infer_mechanism_components(hypothesis.mechanism_family, hypothesis.text, code_text),
            "uses_numpy": "np." in code_text or "numpy" in code_text,
            "numeric_constant_count": len(re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?", code_text, re.IGNORECASE)),
            "branch_count": code_text.count("if ") + code_text.count("for ") + code_text.count("while "),
        }

    @staticmethod
    def _serializable_extraction(data: dict) -> dict:
        serializable = dict(data)
        hypothesis = serializable.get("mechanism_hypothesis")
        if isinstance(hypothesis, MechanismHypothesis):
            serializable["mechanism_hypothesis"] = hypothesis.to_dict()
        return serializable

    @staticmethod
    def _planned_experiment(decision: dict) -> dict:
        args = decision.get("arguments") if isinstance(decision, dict) else {}
        if not isinstance(args, dict):
            return {}
        plan = args.get("offspring_plan")
        if not isinstance(plan, dict):
            return {}
        keys = [
            "offspring_index",
            "parent_id",
            "group_id",
            "variant_id",
            "group_hypothesis",
            "controlled_component",
            "delta",
            "preserve",
            "avoid",
            "expected_effect",
        ]
        return {key: plan.get(key) for key in keys if plan.get(key) is not None}

    @staticmethod
    def _repair_plan_deviation(original_candidate, extracted: dict, planned_experiment: dict) -> str | None:
        original_hypothesis = (
            original_candidate.mechanism_hypothesis
            if isinstance(getattr(original_candidate, "mechanism_hypothesis", None), dict)
            else {}
        )
        original_family = canonicalize_family(
            original_hypothesis.get("mechanism_family"),
            original_hypothesis.get("text") or getattr(original_candidate, "mechanism_summary", ""),
        )
        repaired_hypothesis = extracted.get("mechanism_hypothesis")
        repaired_family = canonicalize_family(
            repaired_hypothesis.mechanism_family if isinstance(repaired_hypothesis, MechanismHypothesis) else None,
            repaired_hypothesis.text if isinstance(repaired_hypothesis, MechanismHypothesis) else "",
        )
        if original_family != "unspecified" and repaired_family != original_family:
            return (
                "repair changed mechanism family from "
                f"{original_family!r} to {repaired_family!r}; repair may only fix implementation errors"
            )
        original_plan = (
            original_candidate.metadata.get("planned_experiment")
            if isinstance(getattr(original_candidate, "metadata", None), dict)
            else {}
        )
        if isinstance(original_plan, dict) and original_plan and planned_experiment:
            for key in ["parent_id", "controlled_component", "delta"]:
                if str(original_plan.get(key) or "") != str(planned_experiment.get(key) or ""):
                    return f"repair changed planned experiment field {key!r}; repair must preserve the assigned plan"
        return None

    def _extract_algorithm_like_eoh(self, response: str) -> str | None:
        text = response or ""
        brace = re.findall(r"\{(.*?)\}", text, re.DOTALL)
        if brace:
            return brace[0].strip()
        code_start = self._first_code_start(text)
        if code_start > 0:
            algorithm = text[:code_start].replace("```python", "").replace("```", "").strip()
            return algorithm[:1000] if algorithm else None
        return None

    def _extract_code_like_eoh(self, response: str) -> str | None:
        text = response or ""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
        fence_blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        for block in fence_blocks:
            code = self._valid_code_prefix(block.strip())
            if code:
                return code
        start = self._first_code_start(text)
        if start < 0:
            return None
        return self._valid_code_prefix(text[start:].strip())

    def _first_code_start(self, text: str) -> int:
        func_name = self.config.get("evaluator", {}).get("function_name")
        starts = []
        if self.config["project"]["problem"] == "tsp_gls_arch":
            class_name = self.config.get("evaluator", {}).get("architecture_class_name", ARCHITECTURE_CLASS_NAME)
            idx = text.find(f"class {class_name}")
            if idx >= 0:
                starts.append(idx)
        if func_name:
            idx = text.find(f"def {func_name}")
            if idx >= 0:
                starts.append(idx)
        for token in ["import ", "from ", "class ", "def "]:
            idx = text.find(token)
            if idx >= 0:
                starts.append(idx)
        return min(starts) if starts else -1

    def _valid_code_prefix(self, text: str) -> str | None:
        lines = text.splitlines()
        best = None
        for end in range(len(lines), 0, -1):
            candidate = "\n".join(lines[:end]).strip()
            if not candidate:
                continue
            try:
                tree = ast.parse(candidate)
            except SyntaxError:
                continue
            if self._contains_expected_function(tree):
                best = candidate
                break
        return best

    def _contains_expected_function(self, tree) -> bool:
        if self.config["project"]["problem"] in {"tsp_gls_arch", "tsp_gls_arch_spec"}:
            class_name = self.config.get("evaluator", {}).get("architecture_class_name", ARCHITECTURE_CLASS_NAME)
            return any(isinstance(node, ast.ClassDef) and node.name == class_name for node in ast.walk(tree))
        func_name = self.config.get("evaluator", {}).get("function_name")
        if not func_name:
            return True
        return any(isinstance(node, ast.FunctionDef) and node.name == func_name for node in ast.walk(tree))

    @staticmethod
    def _decision_payload(decision: dict) -> dict:
        return {
            "next_skill": decision.get("next_skill"),
            "reason": decision.get("reason", ""),
            "arguments": decision.get("arguments", {}),
        }

    @staticmethod
    def _working_context(state) -> dict:
        compact_context = state.compact_context or {}
        if isinstance(compact_context.get("working_memory"), dict):
            return {
                "current_best": compact_context.get("current_best"),
                "working_memory": compact_context.get("working_memory"),
                "search_feedback": compact_context.get("search_feedback"),
                "mechanism_archive": compact_context.get("mechanism_archive"),
                "mechanism_cards": compact_context.get("mechanism_cards"),
                "delta_cards": compact_context.get("delta_cards"),
                "generation_attributions": compact_context.get("generation_attributions"),
                "next_focus": compact_context.get("next_focus"),
            }
        return compact_context

    @staticmethod
    def _prompt_population_summary(state) -> list[dict]:
        return [
            {
                "id": item.get("id"),
                "objective": item.get("objective"),
                "mechanism_summary": str(item.get("mechanism_summary", ""))[:220],
                "mechanism_hypothesis": short_hypothesis(item.get("mechanism_hypothesis")),
                "training_diagnosis": item.get("training_diagnosis"),
                "hypothesis_verdict": item.get("hypothesis_verdict"),
            }
            for item in (state.population_summary or [])[:6]
        ]

    @staticmethod
    def _prompt_notebook_summary(state) -> dict:
        return bounded_notebook(state.notebook or {}, items=3)

    @staticmethod
    def _prompt_mechanism_archive(state) -> dict:
        archive = state.mechanism_archive or {}
        if not isinstance(archive, dict):
            return {}
        return bounded_mechanism_archive(archive, per_bucket=3, family_limit=8)

    @staticmethod
    def _prompt_candidate_summary(candidate) -> dict | None:
        if candidate is None:
            return None
        return {
            "id": candidate.id,
            "objective": candidate.objective,
            "contract_passed": candidate.contract_passed,
            "mechanism_summary": str(candidate.mechanism_summary or "")[:320],
            "mechanism_hypothesis": short_hypothesis(candidate.mechanism_hypothesis),
            "training_diagnosis": (candidate.evidence or {}).get("training_diagnosis")
            if isinstance(candidate.evidence, dict)
            else None,
            "hypothesis_verdict": candidate.hypothesis_verdict,
            "algorithm": str(candidate.algorithm or "")[:450],
            "code": str(candidate.code or "")[:1800],
        }

    @staticmethod
    def _prompt_last_result(state):
        result = state.last_result
        if isinstance(result, str):
            return result[:1000]
        if isinstance(result, dict):
            return {
                key: (str(value)[:1000] if isinstance(value, str) else value)
                for key, value in result.items()
                if key != "raw_response"
            }
        return result

    @staticmethod
    def _same_code(left: str, right: str) -> bool:
        return " ".join(left.split()) == " ".join(right.split())

    @staticmethod
    def _state_payload(state) -> dict:
        return {
            "parent": {
                "id": state.parent.id,
                "objective": state.parent.objective,
                "mechanism_summary": state.parent.mechanism_summary,
                "mechanism_hypothesis": short_hypothesis(state.parent.mechanism_hypothesis),
                "training_diagnosis": (state.parent.evidence or {}).get("training_diagnosis")
                if isinstance(state.parent.evidence, dict)
                else None,
                "hypothesis_verdict": state.parent.hypothesis_verdict,
                "algorithm": str(state.parent.algorithm or "")[:500],
                "code": state.parent.code,
            },
            "current_candidate": ToolRouter._prompt_candidate_summary(state.current_candidate),
            "population_summary": ToolRouter._prompt_population_summary(state),
            "working_memory": ToolRouter._working_context(state),
            "notebook": ToolRouter._prompt_notebook_summary(state),
            "mechanism_archive": ToolRouter._prompt_mechanism_archive(state),
            "step_count": state.step_count,
            "last_result": ToolRouter._prompt_last_result(state),
        }

    @staticmethod
    def _check_arch_spec_code(code: str) -> tuple[list[str], list[str]]:
        errors, warnings = validate_architecture_code(code)
        if errors:
            return errors, warnings
        module = types.ModuleType("candidate_arch_spec_contract")
        try:
            exec(code, module.__dict__)
        except Exception as exc:
            return [f"runtime_error: exec failed: {exc}"], warnings
        raw_spec = getattr(module, "ARCH_SPEC", None)
        if not isinstance(raw_spec, dict):
            return ["arch_spec_error: missing ARCH_SPEC dict"], warnings
        try:
            spec = canonical_arch_spec(raw_spec)
            expected_code = build_arch_spec_code(spec)
        except Exception as exc:
            return [f"arch_spec_error: {exc}"], warnings
        if " ".join(code.split()) != " ".join(expected_code.split()):
            return [
                "arch_spec_code_error: tsp_gls_arch_spec candidates must use the deterministic code generated from ARCH_SPEC"
            ], warnings
        return [], warnings

    @staticmethod
    def _check_solver_tree_code(code: str) -> tuple[list[str], list[str]]:
        warnings: list[str] = []
        module = types.ModuleType("candidate_solver_tree_contract")
        try:
            exec(code, module.__dict__)
        except Exception as exc:
            return [f"runtime_error: exec failed: {exc}"], warnings
        raw_tree = getattr(module, "SOLVER_TREE", None)
        if not isinstance(raw_tree, dict):
            return ["solver_tree_error: missing SOLVER_TREE dict"], warnings
        try:
            solver_tree = canonical_solver_tree(raw_tree)
            expected_code = build_solver_tree_code(solver_tree)
        except Exception as exc:
            return [f"solver_tree_error: {exc}"], warnings
        if " ".join(code.split()) != " ".join(expected_code.split()):
            return [
                "solver_tree_code_error: tsp_gls_solver_tree candidates must use the deterministic code generated from SOLVER_TREE"
            ], warnings
        return [], warnings
