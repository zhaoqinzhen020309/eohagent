from __future__ import annotations

import re
from typing import Any


class ExperimentReviewer:
    def __init__(self, config: dict):
        self.config = config
        self.problem = config.get("project", {}).get("problem")

    def review_plan(
        self,
        parent_spec: dict[str, Any],
        plan: dict[str, Any],
        disabled_patterns: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self.problem == "tsp_gls_solver_tree":
            return self._review_solver_tree(parent_spec, plan, disabled_patterns or [])
        if self.problem != "tsp_gls_arch_spec":
            return self._result(False, f"unsupported_problem: {self.problem}", [], [], "unknown", None)
        return self._review_arch_spec(parent_spec, plan, disabled_patterns or [])

    def _review_solver_tree(
        self,
        parent_tree: dict[str, Any],
        plan: dict[str, Any],
        disabled_patterns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        del disabled_patterns
        from eohagent.problems.tsp_gls_solver_tree.tree_schema import (
            MUTATION_TYPES,
            canonical_solver_tree,
            changed_solver_tree_paths,
            classify_solver_tree_mutation,
            solver_tree_control_trace,
        )

        try:
            parent = canonical_solver_tree(parent_tree)
            child = canonical_solver_tree(plan.get("child_solver_tree"))
            changed_fields = changed_solver_tree_paths(parent, child)
            actual_mutation = classify_solver_tree_mutation(parent, child)
        except Exception as exc:
            return self._result(False, f"schema_or_diff_error: {exc}", [], [], "unknown", None)

        if not changed_fields:
            return self._result(False, "semantic_noop", [], [], "no_op", None)

        intent = str(plan.get("experiment_intent") or "").strip()
        if intent not in {"exploit", "expand", "repair", "validate"}:
            return self._result(False, "invalid_experiment_intent", [], changed_fields, "unknown", None)
        evidence_refs = plan.get("evidence_refs")
        if not isinstance(evidence_refs, list) or not all(isinstance(item, str) and item.strip() for item in evidence_refs):
            return self._result(False, "invalid_evidence_refs", [], changed_fields, "unknown", None)

        declared_mutation = str(plan.get("mutation_type") or "").strip()
        if declared_mutation not in MUTATION_TYPES:
            return self._result(False, "invalid_mutation_type", [], changed_fields, "unknown", None)
        if declared_mutation != actual_mutation:
            return self._result(False, "mutation_type_mismatch", [], changed_fields, actual_mutation, None)

        node_path = str(plan.get("controlled_node_path") or "").strip()
        if not node_path:
            return self._result(False, "missing_controlled_node_path", [], changed_fields, actual_mutation, None)
        normalized_node_path = self._normalize_tree_path(node_path)
        if declared_mutation == "micro_param" and not changed_fields:
            return self._result(False, "semantic_noop", [], changed_fields, "no_op", None)
        if declared_mutation == "micro_param" and not all(
            self._tree_path_is_under_controlled(changed_path, normalized_node_path) for changed_path in changed_fields
        ):
            return self._result(False, "controlled_node_path_mismatch", [], changed_fields, actual_mutation, None)

        expected_trace = str(plan.get("expected_trace_change") or "").strip()
        if not expected_trace:
            return self._result(False, "missing_expected_trace_change", [], changed_fields, actual_mutation, None)
        parent_trace = solver_tree_control_trace(parent)
        child_trace = solver_tree_control_trace(child)
        if declared_mutation == "micro_param" and parent_trace != child_trace:
            return self._result(False, "trace_change_mismatch", [], changed_fields, actual_mutation, None)
        if declared_mutation != "micro_param" and parent_trace == child_trace:
            return self._result(False, "trace_unchanged_for_structural_mutation", [], changed_fields, actual_mutation, None)

        return self._result(True, None, [], changed_fields, actual_mutation, None)

    @staticmethod
    def _normalize_tree_path(path: str) -> str:
        raw = str(path or "").strip()
        if not raw.startswith("/"):
            return raw
        parts = [part.replace("~1", "/").replace("~0", "~") for part in raw.strip("/").split("/") if part != ""]
        normalized: list[str] = []
        for part in parts:
            if part.isdigit() and normalized:
                normalized[-1] = f"{normalized[-1]}[{part}]"
            else:
                normalized.append(part)
        return ".".join(normalized)

    @staticmethod
    def _tree_path_is_under_controlled(changed_path: str, controlled_path: str) -> bool:
        if not controlled_path:
            return False
        return (
            changed_path == controlled_path
            or changed_path.startswith(f"{controlled_path}.")
            or changed_path.startswith(f"{controlled_path}[")
        )

    def _review_arch_spec(
        self,
        parent_spec: dict[str, Any],
        plan: dict[str, Any],
        disabled_patterns: list[dict[str, Any]],
    ) -> dict[str, Any]:
        from eohagent.problems.tsp_gls_arch_spec.spec_schema import (
            changed_arch_spec_fields,
            canonical_arch_spec,
            semantic_noop_patterns,
            validate_single_layer_change,
        )

        try:
            controlled_layer = str(plan.get("controlled_component") or "")
            child_spec = canonical_arch_spec(plan.get("child_spec"))
            parent = canonical_arch_spec(parent_spec)
            changed_layers = validate_single_layer_change(parent, child_spec, controlled_layer)
            changed_fields = changed_arch_spec_fields(parent, child_spec)
        except Exception as exc:
            return self._result(False, f"schema_or_diff_error: {exc}", [], [], "unknown", None)

        declared_change = str(plan.get("controlled_change") or "").strip()
        if not declared_change:
            return self._result(False, "missing_controlled_change", changed_layers, changed_fields, "unknown", None)
        declared_context = " ".join([declared_change, str(plan.get("delta") or "")])
        if declared_change and not self._declared_change_matches(controlled_layer, declared_context, changed_fields):
            return self._result(False, "controlled_change_mismatch", changed_layers, changed_fields, "unknown", None)

        disabled_hit = self._disabled_pattern_hit(parent, child_spec, disabled_patterns)
        if disabled_hit is not None:
            return self._result(False, "disabled_pattern", changed_layers, changed_fields, "unknown", disabled_hit)

        noops = semantic_noop_patterns(parent, child_spec)
        if noops:
            return self._result(
                False,
                "semantic_noop",
                changed_layers,
                changed_fields,
                "no_op",
                noops[0].get("pattern_id"),
                disabled_pattern=noops[0],
            )

        return self._result(True, None, changed_layers, changed_fields, "unknown", None)

    @staticmethod
    def _declared_change_matches(controlled_layer: str, declared_change: str, changed_fields: list[str]) -> bool:
        declared = declared_change.strip()
        if not declared:
            return False
        if declared in changed_fields:
            return True
        if f"{controlled_layer}.{declared}" in changed_fields:
            return True
        return all(
            field.startswith(f"{controlled_layer}.")
            and ExperimentReviewer._field_name_mentioned(field.split(".", 1)[1], declared)
            for field in changed_fields
        )

    @staticmethod
    def _field_name_mentioned(field_name: str, text: str) -> bool:
        normalized = " " + re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip() + " "
        phrase = field_name.replace("_", " ").lower()
        if f" {phrase} " in normalized:
            return True
        if "_" in field_name:
            return field_name.replace("_", "").lower() in normalized.replace(" ", "")
        return False

    def _disabled_pattern_hit(
        self,
        parent_spec: dict[str, Any],
        child_spec: dict[str, Any],
        disabled_patterns: list[dict[str, Any]],
    ) -> str | None:
        if not disabled_patterns:
            return None
        from eohagent.problems.tsp_gls_arch_spec.spec_schema import semantic_noop_patterns

        current = {
            str(pattern.get("pattern_id"))
            for pattern in semantic_noop_patterns(parent_spec, child_spec)
            if pattern.get("problem") == self.problem and pattern.get("pattern_id")
        }
        for pattern in disabled_patterns:
            if pattern.get("problem") not in {None, self.problem}:
                continue
            pattern_id = str(pattern.get("pattern_id") or "")
            if pattern_id in current:
                return pattern_id
        return None

    @staticmethod
    def _result(
        passed: bool,
        rejection_reason: str | None,
        changed_layers: list[str],
        changed_fields: list[str],
        semantic_change: str,
        disabled_pattern_hit: str | None,
        disabled_pattern: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        result = {
            "passed": passed,
            "rejection_reason": rejection_reason,
            "changed_layers": changed_layers,
            "changed_fields": changed_fields,
            "semantic_change": semantic_change,
            "disabled_pattern_hit": disabled_pattern_hit,
        }
        if disabled_pattern is not None:
            result["disabled_pattern"] = disabled_pattern
        return result
