from __future__ import annotations

import json
from typing import Any


def shorten(value: Any, limit: int = 500):
    if isinstance(value, str):
        return value if len(value) <= limit else value[:limit] + "...[truncated]"
    if isinstance(value, list):
        return [shorten(item, limit) for item in value]
    if isinstance(value, dict):
        return {key: shorten(val, limit) for key, val in value.items()}
    return value


def short_text(value: Any, limit: int = 500) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=True)
    value = str(value)
    return value if len(value) <= limit else value[:limit] + "...[truncated]"


def short_hypothesis(hypothesis, text_limit: int = 220) -> dict | str | None:
    if not isinstance(hypothesis, dict):
        return hypothesis
    return {
        "id": hypothesis.get("id"),
        "text": short_text(hypothesis.get("text"), text_limit),
        "mechanism_family": hypothesis.get("mechanism_family"),
        "parent_delta": short_text(hypothesis.get("parent_delta"), 180),
        "expected_effect": short_text(hypothesis.get("expected_effect"), 180),
        "mechanism_components": shorten(hypothesis.get("mechanism_components"), 120),
        "evidence_summary": shorten(hypothesis.get("evidence_summary"), 120),
        "status": hypothesis.get("status"),
    }


def compact_candidate(candidate, include_code: bool = False, code_limit: int = 2200) -> dict:
    if candidate is None:
        return {}
    ref = {
        "id": getattr(candidate, "id", None),
        "objective": getattr(candidate, "objective", None),
        "hypothesis_verdict": getattr(candidate, "hypothesis_verdict", None),
        "mechanism_summary": short_text(getattr(candidate, "mechanism_summary", None), 260),
        "mechanism_hypothesis": short_hypothesis(getattr(candidate, "mechanism_hypothesis", None)),
        "algorithm": short_text(getattr(candidate, "algorithm", None), 360),
        "training_diagnosis": shorten((getattr(candidate, "evidence", None) or {}).get("training_diagnosis"), 140),
    }
    if include_code:
        ref["code"] = short_text(getattr(candidate, "code", None), code_limit)
    return ref


def bounded_notebook(notebook: dict, items: int = 3, text_limit: int = 300) -> dict:
    if not isinstance(notebook, dict):
        return {}
    return {
        "validated_principles": [short_text(item, text_limit) for item in notebook.get("validated_principles", [])[-items:]],
        "mechanism_cards": [shorten(item, text_limit) for item in notebook.get("mechanism_cards", [])[-items:]],
        "delta_cards": [shorten(item, text_limit) for item in notebook.get("delta_cards", [])[-items:]],
        "generation_attributions": [shorten(item, text_limit) for item in notebook.get("generation_attributions", [])[-max(1, items - 1):]],
        "anti_patterns": [short_text(item, text_limit) for item in notebook.get("anti_patterns", [])[-items:]],
        "failed_hypotheses": [short_text(item, text_limit) for item in notebook.get("failed_hypotheses", [])[-items:]],
        "open_questions": [short_text(item, text_limit) for item in notebook.get("open_questions", [])[-max(1, items - 1):]],
    }


def bounded_mechanism_archive(archive: dict, per_bucket: int = 3, family_limit: int = 8, text_limit: int = 300) -> dict:
    if not isinstance(archive, dict):
        return {}
    return {
        "supported_hypotheses": _compact_entries(archive.get("supported_hypotheses", []), per_bucket, text_limit),
        "refuted_hypotheses": _compact_entries(archive.get("refuted_hypotheses", []), per_bucket, text_limit),
        "uncertain_hypotheses": _compact_entries(archive.get("uncertain_hypotheses", []), max(1, per_bucket - 1), text_limit),
        "mechanism_families": _compact_families(archive.get("mechanism_families", {}), family_limit),
        "architecture_tree": _compact_architecture_tree(archive.get("architecture_tree", []), per_bucket * 2, text_limit),
    }


def _compact_entries(entries: list, count: int, text_limit: int) -> list[dict]:
    compacted = []
    for entry in entries[-count:]:
        if not isinstance(entry, dict):
            compacted.append({"text": short_text(entry, text_limit)})
            continue
        compacted.append(
            {
                "candidate_id": entry.get("candidate_id"),
                "mechanism_family": entry.get("mechanism_family"),
                "objective": entry.get("objective"),
                "verdict": entry.get("verdict"),
                "outcome": entry.get("outcome"),
                "text": short_text(entry.get("text"), text_limit),
                "parent_delta": short_text(entry.get("parent_delta"), 180),
                "training_diagnosis": shorten(entry.get("training_diagnosis"), 160),
                "mechanism_components": shorten(entry.get("mechanism_components"), 160),
            }
        )
    return compacted


def _compact_families(families: dict, family_limit: int) -> dict:
    if not isinstance(families, dict):
        return {}
    ranked = sorted(
        families.items(),
        key=lambda item: (
            item[1].get("best_objective") is None if isinstance(item[1], dict) else True,
            item[1].get("best_objective", float("inf")) if isinstance(item[1], dict) else float("inf"),
            -item[1].get("count", 0) if isinstance(item[1], dict) else 0,
        ),
    )[:family_limit]
    out = {}
    for name, data in ranked:
        if not isinstance(data, dict):
            continue
        out[str(name)[:120]] = {
            "count": data.get("count"),
            "best_objective": data.get("best_objective"),
            "best_candidate_id": data.get("best_candidate_id"),
            "best_selection_score": data.get("best_selection_score"),
            "supported": data.get("supported"),
            "refuted": data.get("refuted"),
            "uncertain": data.get("uncertain"),
            "recent_candidates": data.get("recent_candidates", [])[-3:],
        }
    return out


def _compact_architecture_tree(edges: list, count: int, text_limit: int) -> list[dict]:
    compacted = []
    for edge in edges[-count:]:
        if not isinstance(edge, dict):
            continue
        compacted.append(
            {
                "parent_id": edge.get("parent_id"),
                "child_id": edge.get("child_id"),
                "changed_layer": edge.get("changed_layer"),
                "delta": short_text(edge.get("delta"), text_limit),
                "objective": edge.get("objective"),
                "verdict": edge.get("verdict"),
                "outcome": edge.get("outcome"),
                "training_diagnosis": shorten(edge.get("training_diagnosis"), 160),
            }
        )
    return compacted
