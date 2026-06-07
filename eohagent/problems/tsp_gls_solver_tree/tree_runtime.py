from __future__ import annotations

from typing import Any

from eohagent.problems.tsp_gls_solver_tree.tree_schema import canonical_solver_tree


class SolverTreeCandidateGLSArchitecture:
    def __init__(self, solver_tree):
        self.solver_tree = canonical_solver_tree(solver_tree)


def trace_solver_tree(tree: dict[str, Any]) -> list[str]:
    canonical = canonical_solver_tree(tree)
    trace: list[str] = []
    _trace_node(canonical, trace)
    return trace


def _trace_node(node: dict[str, Any], trace: list[str]) -> None:
    node_type = node["type"]
    if node_type == "repeat":
        _trace_node(node["body"], trace)
        return
    if node_type == "sequence":
        for child in node["steps"]:
            _trace_node(child, trace)
        return
    if node_type == "if_stagnated":
        trace.append("if_stagnated")
        _trace_node(node["then"], trace)
        return
    if node_type == "local_search":
        trace.append(f"local_search:{node['neighborhood']}")
        return
    if node_type == "penalty_step":
        trace.append(f"penalty_step:{node['penalty']['edge_selection']}")
        return
    if node_type == "restart":
        trace.append(f"restart:{node['action']}")
        return
    raise ValueError(f"unsupported solver tree node in trace: {node_type}")

