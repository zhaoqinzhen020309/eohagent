from __future__ import annotations

import json
from copy import deepcopy
from typing import Any


ARCH_SPEC_LAYERS = (
    "distance_update_policy",
    "penalty_policy",
    "escape_policy",
    "restart_policy",
    "schedule_policy",
)

DISTANCE_TYPES = {
    "linear_usage",
    "sqrt_usage",
    "log_usage",
    "length_weighted_usage",
    "utility_ratio",
    "rank_usage",
    "smooth_saturation",
}
PENALTY_EDGE_SELECTION = {"top_gap", "tour_utility", "rank_topk", "mixed_gap_utility"}
ESCAPE_TRIGGERS = {"constant", "periodic", "stagnation", "size_adaptive"}
RESTART_ACTIONS = {"none", "reset_to_best", "reset_to_initial", "decay_penalty", "rebuild_nearest"}
RESTART_TRIGGERS = {"never", "periodic", "stagnation"}
STRENGTH_SCHEDULES = {"constant", "decay_after_early", "stagnation_boost", "size_scaled"}


def canonical_arch_spec(spec: dict[str, Any]) -> dict[str, Any]:
    validate_arch_spec(spec)
    return {
        "distance_update_policy": {
            "type": str(spec["distance_update_policy"]["type"]),
            "strength": float(spec["distance_update_policy"]["strength"]),
            "usage_power": float(spec["distance_update_policy"]["usage_power"]),
            "length_weight": float(spec["distance_update_policy"]["length_weight"]),
            "cap_multiplier": float(spec["distance_update_policy"]["cap_multiplier"]),
        },
        "penalty_policy": {
            "edge_selection": str(spec["penalty_policy"]["edge_selection"]),
            "top_k": int(spec["penalty_policy"]["top_k"]),
            "increment": float(spec["penalty_policy"]["increment"]),
            "decay": float(spec["penalty_policy"]["decay"]),
        },
        "escape_policy": {
            "trigger": str(spec["escape_policy"]["trigger"]),
            "moves": int(spec["escape_policy"]["moves"]),
            "period": int(spec["escape_policy"]["period"]),
            "stagnation_threshold": int(spec["escape_policy"]["stagnation_threshold"]),
            "size_threshold": int(spec["escape_policy"]["size_threshold"]),
            "max_moves": int(spec["escape_policy"]["max_moves"]),
        },
        "restart_policy": {
            "action": str(spec["restart_policy"]["action"]),
            "trigger": str(spec["restart_policy"]["trigger"]),
            "period": int(spec["restart_policy"]["period"]),
            "stagnation_threshold": int(spec["restart_policy"]["stagnation_threshold"]),
            "penalty_decay": float(spec["restart_policy"]["penalty_decay"]),
        },
        "schedule_policy": {
            "strength_schedule": str(spec["schedule_policy"]["strength_schedule"]),
            "early_iterations": int(spec["schedule_policy"]["early_iterations"]),
            "late_multiplier": float(spec["schedule_policy"]["late_multiplier"]),
            "stagnation_multiplier": float(spec["schedule_policy"]["stagnation_multiplier"]),
            "size_threshold": int(spec["schedule_policy"]["size_threshold"]),
            "size_multiplier": float(spec["schedule_policy"]["size_multiplier"]),
        },
    }


def validate_arch_spec(spec: dict[str, Any]) -> None:
    if not isinstance(spec, dict):
        raise ValueError("ARCH_SPEC must be a JSON object")
    _require_exact_keys(spec, set(ARCH_SPEC_LAYERS), "ARCH_SPEC")

    distance = _object(spec["distance_update_policy"], "distance_update_policy")
    _require_exact_keys(distance, {"type", "strength", "usage_power", "length_weight", "cap_multiplier"}, "distance_update_policy")
    _choice(distance["type"], DISTANCE_TYPES, "distance_update_policy.type")
    _range_float(distance["strength"], 0.0, 0.6, "distance_update_policy.strength")
    _range_float(distance["usage_power"], 0.0, 2.0, "distance_update_policy.usage_power")
    _range_float(distance["length_weight"], 0.0, 2.0, "distance_update_policy.length_weight")
    _range_float(distance["cap_multiplier"], 1.0, 20.0, "distance_update_policy.cap_multiplier")

    penalty = _object(spec["penalty_policy"], "penalty_policy")
    _require_exact_keys(penalty, {"edge_selection", "top_k", "increment", "decay"}, "penalty_policy")
    _choice(penalty["edge_selection"], PENALTY_EDGE_SELECTION, "penalty_policy.edge_selection")
    _range_int(penalty["top_k"], 1, 20, "penalty_policy.top_k")
    _range_float(penalty["increment"], 0.1, 5.0, "penalty_policy.increment")
    _range_float(penalty["decay"], 0.0, 0.5, "penalty_policy.decay")

    escape = _object(spec["escape_policy"], "escape_policy")
    _require_exact_keys(
        escape,
        {"trigger", "moves", "period", "stagnation_threshold", "size_threshold", "max_moves"},
        "escape_policy",
    )
    _choice(escape["trigger"], ESCAPE_TRIGGERS, "escape_policy.trigger")
    _range_int(escape["moves"], 1, 20, "escape_policy.moves")
    _range_int(escape["period"], 1, 1000, "escape_policy.period")
    _range_int(escape["stagnation_threshold"], 1, 1000, "escape_policy.stagnation_threshold")
    _range_int(escape["size_threshold"], 1, 10000, "escape_policy.size_threshold")
    _range_int(escape["max_moves"], 1, 20, "escape_policy.max_moves")
    if int(escape["max_moves"]) < int(escape["moves"]):
        raise ValueError("escape_policy.max_moves must be >= escape_policy.moves")

    restart = _object(spec["restart_policy"], "restart_policy")
    _require_exact_keys(restart, {"action", "trigger", "period", "stagnation_threshold", "penalty_decay"}, "restart_policy")
    _choice(restart["action"], RESTART_ACTIONS, "restart_policy.action")
    _choice(restart["trigger"], RESTART_TRIGGERS, "restart_policy.trigger")
    _range_int(restart["period"], 1, 1000, "restart_policy.period")
    _range_int(restart["stagnation_threshold"], 1, 1000, "restart_policy.stagnation_threshold")
    _range_float(restart["penalty_decay"], 0.0, 1.0, "restart_policy.penalty_decay")
    if restart["action"] == "none" and restart["trigger"] != "never":
        raise ValueError("restart_policy.action 'none' requires trigger 'never'")
    if restart["action"] != "none" and restart["trigger"] == "never":
        raise ValueError("restart_policy.trigger 'never' requires action 'none'")

    schedule = _object(spec["schedule_policy"], "schedule_policy")
    _require_exact_keys(
        schedule,
        {
            "strength_schedule",
            "early_iterations",
            "late_multiplier",
            "stagnation_multiplier",
            "size_threshold",
            "size_multiplier",
        },
        "schedule_policy",
    )
    _choice(schedule["strength_schedule"], STRENGTH_SCHEDULES, "schedule_policy.strength_schedule")
    _range_int(schedule["early_iterations"], 1, 1000, "schedule_policy.early_iterations")
    _range_float(schedule["late_multiplier"], 0.05, 5.0, "schedule_policy.late_multiplier")
    _range_float(schedule["stagnation_multiplier"], 0.05, 5.0, "schedule_policy.stagnation_multiplier")
    _range_int(schedule["size_threshold"], 1, 10000, "schedule_policy.size_threshold")
    _range_float(schedule["size_multiplier"], 0.05, 5.0, "schedule_policy.size_multiplier")


def changed_arch_spec_layers(parent_spec: dict[str, Any], child_spec: dict[str, Any]) -> list[str]:
    parent = canonical_arch_spec(parent_spec)
    child = canonical_arch_spec(child_spec)
    return [layer for layer in ARCH_SPEC_LAYERS if parent[layer] != child[layer]]


def changed_arch_spec_fields(parent_spec: dict[str, Any], child_spec: dict[str, Any]) -> list[str]:
    parent = canonical_arch_spec(parent_spec)
    child = canonical_arch_spec(child_spec)
    changed = []
    for layer in ARCH_SPEC_LAYERS:
        for field in parent[layer]:
            if parent[layer][field] != child[layer][field]:
                changed.append(f"{layer}.{field}")
    return changed


def validate_single_layer_change(parent_spec: dict[str, Any], child_spec: dict[str, Any], controlled_layer: str) -> list[str]:
    if controlled_layer not in ARCH_SPEC_LAYERS:
        raise ValueError(f"controlled_component {controlled_layer!r} is not an allowed ARCH_SPEC layer")
    changed = changed_arch_spec_layers(parent_spec, child_spec)
    if changed != [controlled_layer]:
        raise ValueError(
            f"child_spec changed layers {changed}, expected exactly [{controlled_layer!r}]"
        )
    return changed


def semantic_noop_patterns(parent_spec: dict[str, Any], child_spec: dict[str, Any]) -> list[dict[str, Any]]:
    parent = canonical_arch_spec(parent_spec)
    child = canonical_arch_spec(child_spec)
    changed_fields = set(changed_arch_spec_fields(parent, child))
    patterns = []
    if changed_fields == {"escape_policy.trigger"}:
        escape = parent["escape_policy"]
        if int(escape["moves"]) >= int(escape["max_moves"]):
            patterns.append(
                {
                    "problem": "tsp_gls_arch_spec",
                    "pattern_id": "escape_policy.trigger_at_max_moves",
                    "changed_fields": ["escape_policy.trigger"],
                    "reason": "escape_policy.moves already equals max_moves, so trigger changes are clipped away",
                }
            )
    return patterns


def build_arch_spec_code(spec: dict[str, Any]) -> str:
    canonical = canonical_arch_spec(spec)
    spec_json = json.dumps(canonical, indent=2, sort_keys=True)
    return (
        "from eohagent.problems.tsp_gls_arch_spec.spec_runtime import SpecCandidateGLSArchitecture\n\n"
        f"ARCH_SPEC = {spec_json}\n\n"
        "class CandidateGLSArchitecture(SpecCandidateGLSArchitecture):\n"
        "    def __init__(self):\n"
        "        super().__init__(ARCH_SPEC)\n"
    )


def arch_spec_from_candidate(candidate) -> dict[str, Any]:
    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, dict) and isinstance(metadata.get("arch_spec"), dict):
        return canonical_arch_spec(metadata["arch_spec"])
    raise ValueError(f"candidate {getattr(candidate, 'id', '<unknown>')} has no auditable arch_spec")


def arch_spec_niche_key(spec: dict[str, Any]) -> tuple[str, ...]:
    canonical = canonical_arch_spec(spec)
    return (
        canonical["distance_update_policy"]["type"],
        canonical["penalty_policy"]["edge_selection"],
        str(canonical["penalty_policy"]["top_k"]),
        canonical["escape_policy"]["trigger"],
        canonical["restart_policy"]["action"],
        canonical["schedule_policy"]["strength_schedule"],
    )


def mechanism_family_from_spec(spec: dict[str, Any]) -> str:
    distance_type = canonical_arch_spec(spec)["distance_update_policy"]["type"]
    mapping = {
        "linear_usage": "linear_usage_penalty",
        "sqrt_usage": "sqrt_usage_penalty",
        "log_usage": "log_usage_penalty",
        "length_weighted_usage": "length_weighted_penalty",
        "utility_ratio": "utility_ratio_penalty",
        "rank_usage": "rank_topk_penalty",
        "smooth_saturation": "smooth_saturation_penalty",
    }
    return mapping.get(distance_type, "unspecified")


def describe_arch_spec(spec: dict[str, Any]) -> str:
    canonical = canonical_arch_spec(spec)
    return (
        "GLS ARCH_SPEC: "
        f"distance={canonical['distance_update_policy']['type']} "
        f"strength={canonical['distance_update_policy']['strength']}; "
        f"penalty={canonical['penalty_policy']['edge_selection']} "
        f"top_k={canonical['penalty_policy']['top_k']} "
        f"increment={canonical['penalty_policy']['increment']} "
        f"decay={canonical['penalty_policy']['decay']}; "
        f"escape={canonical['escape_policy']['trigger']} moves={canonical['escape_policy']['moves']}; "
        f"restart={canonical['restart_policy']['trigger']}:{canonical['restart_policy']['action']}; "
        f"schedule={canonical['schedule_policy']['strength_schedule']}"
    )


def apply_layer_change(parent_spec: dict[str, Any], layer: str, layer_value: dict[str, Any]) -> dict[str, Any]:
    parent = canonical_arch_spec(parent_spec)
    if layer not in ARCH_SPEC_LAYERS:
        raise ValueError(f"unknown arch spec layer: {layer}")
    child = deepcopy(parent)
    child[layer] = deepcopy(layer_value)
    return canonical_arch_spec(child)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return value


def _require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    keys = set(value)
    missing = sorted(expected - keys)
    extra = sorted(keys - expected)
    if missing or extra:
        raise ValueError(f"{label} keys mismatch: missing={missing}, extra={extra}")


def _choice(value: Any, allowed: set[str], label: str) -> None:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{label} must be one of {sorted(allowed)}, got {value!r}")


def _range_float(value: Any, low: float, high: float, label: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if number < low or number > high:
        raise ValueError(f"{label} must be in [{low}, {high}], got {number}")


def _range_int(value: Any, low: int, high: int, label: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer, got bool")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    if float(value) != float(number):
        raise ValueError(f"{label} must be an integer, got {value!r}")
    if number < low or number > high:
        raise ValueError(f"{label} must be in [{low}, {high}], got {number}")
