from __future__ import annotations

import hashlib
import json
import pprint
from copy import deepcopy
from typing import Any


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
STRENGTH_SCHEDULES = {"constant", "decay_after_early", "stagnation_boost", "size_scaled"}
LOCAL_SEARCH_NEIGHBORHOODS = {"two_opt_relocate"}
RESTART_ACTIONS = {"reset_to_best", "reset_to_initial", "decay_penalty", "rebuild_nearest"}
MUTATION_TYPES = {"micro_param", "replace_subtree", "insert_node", "remove_node", "reorder_sequence", "wrap_if_stagnated"}

MAX_TREE_DEPTH = 4
MAX_SEQUENCE_STEPS = 5
MAX_TOTAL_NODES = 14


def canonical_solver_tree(tree: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(tree, dict):
        raise ValueError("SOLVER_TREE must be a JSON object")
    canonical = _canonical_node(tree, path="root", depth=1, root=True)
    if canonical["type"] != "repeat":
        raise ValueError("SOLVER_TREE root must be a repeat node")
    node_count = len(_flatten_nodes(canonical))
    if node_count > MAX_TOTAL_NODES:
        raise ValueError(f"SOLVER_TREE has {node_count} nodes, limit is {MAX_TOTAL_NODES}")
    return canonical


def validate_solver_tree(tree: dict[str, Any]) -> None:
    canonical_solver_tree(tree)


def build_solver_tree_code(tree: dict[str, Any]) -> str:
    canonical = canonical_solver_tree(tree)
    tree_literal = pprint.pformat(canonical, sort_dicts=True, width=100)
    return (
        "from eohagent.problems.tsp_gls_solver_tree.tree_runtime import SolverTreeCandidateGLSArchitecture\n\n"
        f"SOLVER_TREE = {tree_literal}\n\n"
        "class CandidateGLSArchitecture(SolverTreeCandidateGLSArchitecture):\n"
        "    def __init__(self):\n"
        "        super().__init__(SOLVER_TREE)\n"
    )


def solver_tree_from_candidate(candidate) -> dict[str, Any]:
    metadata = getattr(candidate, "metadata", None)
    if isinstance(metadata, dict) and isinstance(metadata.get("solver_tree"), dict):
        return canonical_solver_tree(metadata["solver_tree"])
    raise ValueError(f"candidate {getattr(candidate, 'id', '<unknown>')} has no auditable solver_tree")


def solver_tree_hash(tree: dict[str, Any]) -> str:
    canonical = canonical_solver_tree(tree)
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def solver_tree_descriptor(tree: dict[str, Any]) -> tuple[str, ...]:
    from eohagent.problems.tsp_gls_solver_tree.tree_runtime import trace_solver_tree

    canonical = canonical_solver_tree(tree)
    trace = trace_solver_tree(canonical)
    actions = [item.split(":", 1)[0] for item in trace]
    if "penalty_step" in actions and "local_search" in actions:
        penalty_index = actions.index("penalty_step")
        local_index = actions.index("local_search")
        penalty_timing = "penalty_after_local_search" if local_index < penalty_index else "penalty_before_local_search"
    else:
        penalty_timing = "no_penalty_step"
    shape = "_".join(actions[:6]) or "empty"
    if_stagnated = "has_if_stagnated" if "if_stagnated" in actions else "no_if_stagnated"
    restart = "has_restart" if "restart" in actions else "no_restart"
    penalty_repeat = _max_penalty_repeat(canonical)
    return (
        "solver_tree",
        f"shape={shape}",
        penalty_timing,
        if_stagnated,
        restart,
        f"penalty_repeat={penalty_repeat}",
    )


def describe_solver_tree(tree: dict[str, Any]) -> str:
    from eohagent.problems.tsp_gls_solver_tree.tree_runtime import trace_solver_tree

    canonical = canonical_solver_tree(tree)
    return "SOLVER_TREE: " + " -> ".join(trace_solver_tree(canonical))


def changed_solver_tree_paths(parent_tree: dict[str, Any], child_tree: dict[str, Any]) -> list[str]:
    parent = _flatten_scalars(canonical_solver_tree(parent_tree))
    child = _flatten_scalars(canonical_solver_tree(child_tree))
    paths = sorted(set(parent) | set(child))
    return [path for path in paths if parent.get(path) != child.get(path)]


def classify_solver_tree_mutation(parent_tree: dict[str, Any], child_tree: dict[str, Any]) -> str:
    parent = canonical_solver_tree(parent_tree)
    child = canonical_solver_tree(child_tree)
    parent_trace = solver_tree_control_trace(parent)
    child_trace = solver_tree_control_trace(child)
    if parent_trace == child_trace:
        return "micro_param"
    if len(parent_trace) < len(child_trace):
        return "insert_node"
    if len(parent_trace) > len(child_trace):
        return "remove_node"
    if sorted(parent_trace) == sorted(child_trace):
        return "reorder_sequence"
    return "replace_subtree"


def solver_tree_control_trace(tree: dict[str, Any]) -> list[str]:
    trace: list[str] = []
    _control_trace_node(canonical_solver_tree(tree), trace)
    return trace


def _control_trace_node(node: dict[str, Any], trace: list[str]) -> None:
    node_type = node["type"]
    if node_type == "repeat":
        _control_trace_node(node["body"], trace)
        return
    if node_type == "sequence":
        for child in node["steps"]:
            _control_trace_node(child, trace)
        return
    if node_type == "if_stagnated":
        trace.append("if_stagnated")
        _control_trace_node(node["then"], trace)
        return
    trace.append(node_type)


def _canonical_node(node: Any, path: str, depth: int, root: bool = False) -> dict[str, Any]:
    if depth > MAX_TREE_DEPTH:
        raise ValueError(f"SOLVER_TREE exceeds max depth {MAX_TREE_DEPTH} at {path}")
    obj = _object(node, path)
    node_type = str(obj.get("type") or "")
    if node_type == "repeat":
        if not root:
            raise ValueError("repeat node is only allowed at the root")
        _require_exact_keys(obj, {"type", "budget", "body"}, path)
        if obj["budget"] != "main_loop":
            raise ValueError("repeat.budget must be 'main_loop'")
        return {
            "type": "repeat",
            "budget": "main_loop",
            "body": _canonical_node(obj["body"], f"{path}.body", depth + 1),
        }
    if node_type == "sequence":
        _require_exact_keys(obj, {"type", "steps"}, path)
        steps = obj["steps"]
        if not isinstance(steps, list) or not steps:
            raise ValueError(f"{path}.steps must be a non-empty list")
        if len(steps) > MAX_SEQUENCE_STEPS:
            raise ValueError(f"{path}.steps has {len(steps)} steps, limit is {MAX_SEQUENCE_STEPS}")
        return {
            "type": "sequence",
            "steps": [_canonical_node(step, f"{path}.steps[{idx}]", depth + 1) for idx, step in enumerate(steps)],
        }
    if node_type == "if_stagnated":
        _require_exact_keys(obj, {"type", "threshold", "then"}, path)
        return {
            "type": "if_stagnated",
            "threshold": _int_range(obj["threshold"], 1, 1000, f"{path}.threshold"),
            "then": _canonical_node(obj["then"], f"{path}.then", depth + 1),
        }
    if node_type == "local_search":
        _require_exact_keys(obj, {"type", "neighborhood", "first_improvement"}, path)
        _choice(obj["neighborhood"], LOCAL_SEARCH_NEIGHBORHOODS, f"{path}.neighborhood")
        if not isinstance(obj["first_improvement"], bool):
            raise ValueError(f"{path}.first_improvement must be bool")
        return {
            "type": "local_search",
            "neighborhood": str(obj["neighborhood"]),
            "first_improvement": bool(obj["first_improvement"]),
        }
    if node_type == "penalty_step":
        _require_exact_keys(obj, {"type", "repeat", "distance_update", "penalty", "schedule"}, path)
        return {
            "type": "penalty_step",
            "repeat": _int_range(obj["repeat"], 1, 20, f"{path}.repeat"),
            "distance_update": _canonical_distance(obj["distance_update"], f"{path}.distance_update"),
            "penalty": _canonical_penalty(obj["penalty"], f"{path}.penalty"),
            "schedule": _canonical_schedule(obj["schedule"], f"{path}.schedule"),
        }
    if node_type == "restart":
        allowed_keys = {"type", "action"} if obj.get("action") != "decay_penalty" else {"type", "action", "penalty_decay"}
        _require_exact_keys(obj, allowed_keys, path)
        _choice(obj["action"], RESTART_ACTIONS, f"{path}.action")
        result = {"type": "restart", "action": str(obj["action"])}
        if obj["action"] == "decay_penalty":
            result["penalty_decay"] = _float_range(obj["penalty_decay"], 0.0, 1.0, f"{path}.penalty_decay")
        return result
    raise ValueError(f"unsupported solver tree node at {path}: {node_type!r}")


def _canonical_distance(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    _require_exact_keys(obj, {"type", "strength", "usage_power", "length_weight", "cap_multiplier"}, path)
    _choice(obj["type"], DISTANCE_TYPES, f"{path}.type")
    return {
        "type": str(obj["type"]),
        "strength": _float_range(obj["strength"], 0.0, 0.6, f"{path}.strength"),
        "usage_power": _float_range(obj["usage_power"], 0.0, 2.0, f"{path}.usage_power"),
        "length_weight": _float_range(obj["length_weight"], 0.0, 2.0, f"{path}.length_weight"),
        "cap_multiplier": _float_range(obj["cap_multiplier"], 1.0, 20.0, f"{path}.cap_multiplier"),
    }


def _canonical_penalty(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    _require_exact_keys(obj, {"edge_selection", "top_k", "increment", "decay"}, path)
    _choice(obj["edge_selection"], PENALTY_EDGE_SELECTION, f"{path}.edge_selection")
    return {
        "edge_selection": str(obj["edge_selection"]),
        "top_k": _int_range(obj["top_k"], 1, 20, f"{path}.top_k"),
        "increment": _float_range(obj["increment"], 0.1, 5.0, f"{path}.increment"),
        "decay": _float_range(obj["decay"], 0.0, 0.5, f"{path}.decay"),
    }


def _canonical_schedule(value: Any, path: str) -> dict[str, Any]:
    obj = _object(value, path)
    _require_exact_keys(
        obj,
        {
            "strength_schedule",
            "early_iterations",
            "late_multiplier",
            "stagnation_multiplier",
            "size_threshold",
            "size_multiplier",
        },
        path,
    )
    _choice(obj["strength_schedule"], STRENGTH_SCHEDULES, f"{path}.strength_schedule")
    return {
        "strength_schedule": str(obj["strength_schedule"]),
        "early_iterations": _int_range(obj["early_iterations"], 1, 1000, f"{path}.early_iterations"),
        "late_multiplier": _float_range(obj["late_multiplier"], 0.05, 5.0, f"{path}.late_multiplier"),
        "stagnation_multiplier": _float_range(obj["stagnation_multiplier"], 0.05, 5.0, f"{path}.stagnation_multiplier"),
        "size_threshold": _int_range(obj["size_threshold"], 1, 10000, f"{path}.size_threshold"),
        "size_multiplier": _float_range(obj["size_multiplier"], 0.05, 5.0, f"{path}.size_multiplier"),
    }


def _flatten_nodes(node: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = [node]
    if node["type"] == "repeat":
        nodes.extend(_flatten_nodes(node["body"]))
    elif node["type"] == "sequence":
        for child in node["steps"]:
            nodes.extend(_flatten_nodes(child))
    elif node["type"] == "if_stagnated":
        nodes.extend(_flatten_nodes(node["then"]))
    return nodes


def _flatten_scalars(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        flattened = {}
        for key in sorted(value):
            flattened.update(_flatten_scalars(value[key], f"{prefix}.{key}" if prefix else key))
        return flattened
    if isinstance(value, list):
        flattened = {}
        for index, item in enumerate(value):
            flattened.update(_flatten_scalars(item, f"{prefix}[{index}]"))
        return flattened
    return {prefix: value}


def _max_penalty_repeat(tree: dict[str, Any]) -> int:
    repeats = [int(node["repeat"]) for node in _flatten_nodes(tree) if node["type"] == "penalty_step"]
    return max(repeats) if repeats else 0


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


def _float_range(value: Any, low: float, high: float, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if number < low or number > high:
        raise ValueError(f"{label} must be in [{low}, {high}], got {number}")
    return number


def _int_range(value: Any, low: int, high: int, label: str) -> int:
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
    return number
