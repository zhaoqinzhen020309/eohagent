from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


CANONICAL_MECHANISM_FAMILIES = {
    "linear_usage_penalty",
    "sqrt_usage_penalty",
    "log_usage_penalty",
    "capped_usage_penalty",
    "rank_topk_penalty",
    "length_weighted_penalty",
    "utility_ratio_penalty",
    "adaptive_strength_by_n",
    "mean_usage_normalized",
    "smooth_saturation_penalty",
    "decay_penalty",
    "unspecified",
}


@dataclass
class MechanismHypothesis:
    id: str | None = None
    parent_id: str | None = None
    problem: str | None = None
    text: str = ""
    mechanism_family: str = "unspecified"
    parent_delta: str = ""
    expected_effect: str = ""
    preserve_components: list[str] = field(default_factory=list)
    change_components: list[str] = field(default_factory=list)
    mechanism_components: dict[str, str] = field(default_factory=dict)
    risk_assumptions: list[str] = field(default_factory=list)
    evidence_summary: dict[str, Any] = field(default_factory=dict)
    status: str = "proposed"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_description(cls, description: str | None, parent_id: str | None, problem: str | None) -> "MechanismHypothesis":
        text = (description or "").strip()
        fields = _extract_labeled_fields(text)
        hypothesis = fields.get("hypothesis") or fields.get("mechanism hypothesis") or text
        family = fields.get("mechanism family") or fields.get("family") or _infer_family(text)
        parent_delta = fields.get("parent delta") or fields.get("delta") or fields.get("implementation delta") or ""
        expected_effect = fields.get("expected effect") or fields.get("effect") or ""
        return cls(
            parent_id=parent_id,
            problem=problem,
            text=hypothesis.strip(),
            mechanism_family=canonicalize_family(family, text),
            parent_delta=parent_delta.strip(),
            expected_effect=expected_effect.strip(),
            preserve_components=_split_items(fields.get("preserve") or fields.get("preserve components")),
            change_components=_split_items(
                fields.get("change") or fields.get("change components") or fields.get("controlled component")
            ),
            mechanism_components=infer_mechanism_components(family, text),
            risk_assumptions=_split_items(fields.get("risk") or fields.get("risk assumptions")),
        )

    def short_summary(self) -> str:
        parts = [self.text]
        if self.parent_delta:
            parts.append(f"delta: {self.parent_delta}")
        if self.expected_effect:
            parts.append(f"expected: {self.expected_effect}")
        return "; ".join(part for part in parts if part)


def _extract_labeled_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    if not text:
        return fields
    pattern = re.compile(
        r"(hypothesis|mechanism hypothesis|mechanism family|family|controlled component|parent delta|implementation delta|delta|expected effect|effect|preserve components|preserve|change components|change|risk assumptions|risk)\s*:\s*",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    for index, match in enumerate(matches):
        key = match.group(1).lower()
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        value = text[start:end].strip(" ;,\n")
        if value:
            fields[key] = value
    return fields


def _split_items(value: str | None) -> list[str]:
    if not value:
        return []
    items = re.split(r"[,;]\s*", value)
    return [item.strip(" .") for item in items if item.strip(" .")]


def _infer_family(text: str) -> str:
    lower = text.lower()
    if "top-k" in lower or "top k" in lower or "rank" in lower or "quantile" in lower:
        return "rank_topk_penalty"
    if "utility" in lower or ("ratio" in lower and ("cost" in lower or "length" in lower)):
        return "utility_ratio_penalty"
    if "sigmoid" in lower or "tanh" in lower or "saturation" in lower or "cap" in lower or "bounded" in lower:
        return "smooth_saturation_penalty"
    if "sqrt" in lower or "root" in lower or "sublinear" in lower:
        return "sqrt_usage_penalty"
    if "log" in lower:
        return "log_usage_penalty"
    if "length" in lower or "edge cost" in lower or "long edge" in lower:
        return "length_weighted_penalty"
    if "harmonic" in lower:
        return "mean_usage_normalized"
    if "decay" in lower:
        return "decay_penalty"
    if "scale by n" in lower or "problem size" in lower or "tour length" in lower:
        return "adaptive_strength_by_n"
    if "normal" in lower or "mean" in lower or "relative" in lower:
        return "mean_usage_normalized"
    if "linear" in lower or "usage" in lower:
        return "linear_usage_penalty"
    return "unspecified"


def canonicalize_family(family: str | None, text: str | None = None) -> str:
    raw = (family or "").strip().lower()
    raw = raw.strip(" '\"`.,;:")
    raw = raw.replace("-", "_").replace(" ", "_")
    aliases = {
        "usage_penalty": "linear_usage_penalty",
        "usage_penalty_strength_variation": "linear_usage_penalty",
        "adaptive_usage_penalty": "adaptive_strength_by_n",
        "usage_penalty_adaptive": "adaptive_strength_by_n",
        "usage_penalty_with_cap": "capped_usage_penalty",
        "usage_penalty_saturation": "smooth_saturation_penalty",
        "usage_penalty_saturated": "smooth_saturation_penalty",
        "edge_cost_weighted_usage_penalty": "length_weighted_penalty",
        "usage_penalty_with_length_correction": "length_weighted_penalty",
        "normalized_usage_penalty": "mean_usage_normalized",
        "harmonic_scaling": "mean_usage_normalized",
    }
    if raw in CANONICAL_MECHANISM_FAMILIES:
        return raw
    if raw in aliases:
        return aliases[raw]
    combined = " ".join(part for part in [raw.replace("_", " "), text or ""] if part).lower()
    inferred = _infer_family(combined)
    return inferred if inferred in CANONICAL_MECHANISM_FAMILIES else "unspecified"


def infer_mechanism_components(family: str | None, text: str | None = None, code: str | None = None) -> dict[str, str]:
    family = canonicalize_family(family, text)
    combined = " ".join(part for part in [text or "", code or ""] if part).lower()
    components = {
        "edge_selection": "all_local_opt_tour_edges",
        "usage_signal": "raw_edge_usage",
        "distance_normalization": "none",
        "penalty_transform": family,
        "update_strength": "fixed_or_code_defined",
        "stability_control": "none",
    }
    if family == "rank_topk_penalty" or "top" in combined or "rank" in combined:
        components["edge_selection"] = "rank_or_topk_tour_edges"
    if family in {"sqrt_usage_penalty", "log_usage_penalty", "smooth_saturation_penalty"}:
        components["usage_signal"] = family
    if "entropy" in combined:
        components["usage_signal"] = "entropy_adjusted_usage"
    if family in {"length_weighted_penalty", "utility_ratio_penalty"}:
        components["distance_normalization"] = "edge_length_or_utility_scaled"
    if family in {"mean_usage_normalized", "adaptive_strength_by_n"} or "normal" in combined:
        components["distance_normalization"] = "relative_or_size_normalized"
    if family in {"capped_usage_penalty", "smooth_saturation_penalty"} or "cap" in combined or "clip" in combined:
        components["stability_control"] = "bounded_penalty_growth"
    if "decay" in combined:
        components["stability_control"] = "decay_or_memory_reset"
    if "adaptive" in combined or "schedule" in combined:
        components["update_strength"] = "adaptive_schedule"
    return components
