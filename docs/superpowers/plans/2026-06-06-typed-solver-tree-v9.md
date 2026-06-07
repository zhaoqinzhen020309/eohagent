# Typed Solver Tree V9 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real typed solver-tree workflow for TSP-GLS so the executable search pipeline, not only fixed ARCH_SPEC fields, can be evolved under QD, LangGraph, LangMem, and harness checks.

**Architecture:** Add a new `tsp_gls_solver_tree` problem path beside the existing `tsp_gls_arch_spec` baseline. Candidates are deterministic wrappers around `SOLVER_TREE`; a tree interpreter executes node order in the solver loop, QD stores cell elites by tree behavior descriptor, LangGraph exposes real workflow nodes, LangMem stores typed experiment memories, and harness gates fail closed on schema/evidence/intent/diff violations.

**Tech Stack:** Python, existing DeepSeek/OpenAI-compatible client, LangGraph StateGraph, LangMem memory tools, deterministic JSON/YAML specs, existing TSP-GLS operators.

---

## Semantic Contract

- `SOLVER_TREE` must change actual execution order in `gls_tree_evol.py`; metadata-only trees are invalid.
- Candidate code must be exactly the deterministic wrapper generated from `SOLVER_TREE`; handwritten Python extensions are rejected.
- Tree mutations must declare a mutation type and node path, and the reviewer must verify the declared mutation matches the actual tree diff.
- QD archive must update by cell and parent selection must sample cells, not just sort all elites globally and keep the first ten.
- LangGraph must contain separate workflow nodes for bootstrap/generation planning/review+evaluation/archive update/reflection, not one `run_engine` wrapper node.
- LangMem must record typed memories with candidate id, tree descriptor, evidence source, verdict, and generation.
- No fallback: invalid plans, invalid trees, memory failures, review failures, and reflection failures must fail closed.

## Implementation Tasks

- [ ] Add tests for solver-tree schema canonicalization, duplicate hash, node limits, and illegal nodes.
- [ ] Add tests proving two legal trees with different node order produce different execution traces.
- [ ] Add tests for deterministic candidate wrapper validation in `ToolRouter`.
- [ ] Add tests for planner protocol requiring `child_solver_tree`, `mutation_type`, `controlled_node_path`, `expected_trace_change`, `experiment_intent`, and real `evidence_refs`.
- [ ] Add tests for `ExperimentReviewer` rejecting declared micro mutations that actually reorder or insert nodes.
- [ ] Add tests for QD archive parent selection preserving multiple tree descriptors instead of selecting only global top objectives.
- [ ] Add tests for LangGraph workflow node names and LangMem typed record persistence.
- [ ] Implement `eohagent/problems/tsp_gls_solver_tree/tree_schema.py`.
- [ ] Implement `tree_runtime.py`, `gls_tree_evol.py`, and `gls_tree_run.py`.
- [ ] Integrate `tsp_gls_solver_tree` into `EvaluatorBridge`, `ToolRouter`, `BaselineLoader`, `GenerationPlanner`, `ExperimentReviewer`, `QDArchive`, and `ExpertEoHEngine`.
- [ ] Add v9 seed file and config.
- [ ] Run focused tests, smoke tests, and semantic audit commands before reporting completion.
