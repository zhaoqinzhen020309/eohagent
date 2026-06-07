# Generalization-Aware Expert Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the approved generalization-aware expert harness changes for typed ARCH_SPEC search without probe sets, fallback children, or repair-driven experiment mutation.

**Architecture:** Add a deterministic pre-eval review gate for typed specs, store run-local disabled semantic no-op patterns, feed those constraints back to the generation planner, and make ARCH_SPEC contract failure reject instead of repair. Keep problem-specific semantic knowledge in the TSP ARCH_SPEC adapter and keep core orchestration problem-agnostic where possible.

**Tech Stack:** Python 3, existing `unittest` suite in `eohagent/tests/test_eohagent.py`, CodeGraph CLI for structural lookup.

---

### Task 1: Add Deterministic ARCH_SPEC Experiment Review

**Files:**
- Create: `eohagent/harness/experiment_review.py`
- Modify: `eohagent/problems/tsp_gls_arch_spec/spec_schema.py`
- Test: `eohagent/tests/test_eohagent.py`

- [ ] **Step 1: Write failing tests**

Add tests that call a new review API directly:

```python
from eohagent.harness.experiment_review import ExperimentReviewer
from eohagent.problems.tsp_gls_arch_spec.spec_schema import apply_layer_change

def test_arch_spec_review_rejects_disabled_pattern(self):
    parent = self._base_arch_spec()
    child = apply_layer_change(parent, "escape_policy", {**parent["escape_policy"], "trigger": "periodic"})
    reviewer = ExperimentReviewer({"project": {"problem": "tsp_gls_arch_spec"}})
    result = reviewer.review_plan(
        parent_spec=parent,
        plan={
            "parent_id": "p",
            "controlled_component": "escape_policy",
            "controlled_change": "trigger",
            "child_spec": child,
        },
        disabled_patterns=[{"problem": "tsp_gls_arch_spec", "pattern_id": "escape_policy.trigger_at_max_moves"}],
    )
    self.assertFalse(result["passed"])
    self.assertEqual(result["rejection_reason"], "disabled_pattern")

def test_arch_spec_review_records_unknown_semantic_when_no_rule_matches(self):
    parent = self._base_arch_spec()
    child = apply_layer_change(parent, "penalty_policy", {**parent["penalty_policy"], "top_k": 8})
    reviewer = ExperimentReviewer({"project": {"problem": "tsp_gls_arch_spec"}})
    result = reviewer.review_plan(
        parent_spec=parent,
        plan={
            "parent_id": "p",
            "controlled_component": "penalty_policy",
            "controlled_change": "top_k",
            "child_spec": child,
        },
        disabled_patterns=[],
    )
    self.assertTrue(result["passed"])
    self.assertEqual(result["semantic_change"], "unknown")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -B -m unittest eohagent.tests.test_eohagent.TestEoHAgent.test_arch_spec_review_rejects_disabled_pattern eohagent.tests.test_eohagent.TestEoHAgent.test_arch_spec_review_records_unknown_semantic_when_no_rule_matches`

Expected: fail because `eohagent.harness.experiment_review` does not exist.

- [ ] **Step 3: Implement minimal review gate**

Create `ExperimentReviewer.review_plan(...)` that validates schema, checks declared component/change against actual diff, checks disabled patterns, and returns a structured dict with `passed`, `rejection_reason`, `changed_layers`, `changed_fields`, `semantic_change`, and `disabled_pattern_hit`.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same command. Expected: both tests pass.

### Task 2: Wire Review Before Evaluation and Disable Confirmed No-Op Patterns

**Files:**
- Modify: `eohagent/core/engine.py`
- Modify: `eohagent/harness/generation_planner.py`
- Test: `eohagent/tests/test_eohagent.py`

- [ ] **Step 1: Write failing tests**

Add tests that verify run-local disabled patterns are passed into planner payload and that review failure produces rejected offspring before evaluator entry.

- [ ] **Step 2: Run tests and verify RED**

Run the new tests individually. Expected: fail because engine has no reviewer state and planner payload has no disabled pattern list.

- [ ] **Step 3: Implement minimal wiring**

Instantiate `ExperimentReviewer` in `EvolutionEngine`, run review before `harness.run_with_plan` for ARCH_SPEC offspring, append a `pre_eval_review` event, reject failed review without evaluator entry, and add confirmed semantic no-op patterns to an engine-local disabled list.

- [ ] **Step 4: Pass disabled patterns to planner**

Add a bounded `disabled_patterns` field to `GenerationPlanner._plan_payload(...)` and include it in the instruction as hard constraints.

- [ ] **Step 5: Run tests and verify GREEN**

Run the new tests and the relevant existing planner/engine tests.

### Task 3: Fail Closed Instead of Repair for ARCH_SPEC Contract Failure

**Files:**
- Modify: `eohagent/harness/commit_gate.py`
- Test: `eohagent/tests/test_eohagent.py`

- [ ] **Step 1: Write failing test**

Add a test that builds `CommitGate` with `project.problem=tsp_gls_arch_spec`, sets `contract_checked=True` and `contract_passed=False`, and expects `reject`.

- [ ] **Step 2: Run test and verify RED**

Run: `python -B -m unittest eohagent.tests.test_eohagent.TestEoHAgent.test_arch_spec_commit_gate_rejects_failed_contract_without_repair`

Expected: fail because current gate returns `repair`.

- [ ] **Step 3: Implement minimal gate change**

If problem is `tsp_gls_arch_spec` and contract failed, return `reject`. Leave existing repair behavior for non-ARCH_SPEC workflows unchanged.

- [ ] **Step 4: Run test and verify GREEN**

Run the same command. Expected: pass.

### Task 4: Tighten Training-Robust Selection Evidence

**Files:**
- Modify: `eohagent/core/engine.py`
- Modify: `eohagent/core/population.py`
- Test: `eohagent/tests/test_eohagent.py`

- [ ] **Step 1: Write failing tests**

Add tests showing `_selection_score` and Pareto metrics prefer candidates with better training tail/max/std when mean objective is similar, using only `training_diagnosis`.

- [ ] **Step 2: Run tests and verify RED**

Run the new selection tests. Expected: fail because `_selection_score` currently returns objective only and `_pareto_metrics` partly reads stale formal fields.

- [ ] **Step 3: Implement minimal robust scoring**

Compute selection score from objective plus training-only robust diagnostics, using small deterministic penalties for tail/max/std and no test evidence. Update `_pareto_metrics` to prefer `training_diagnosis` for tail, max, and std.

- [ ] **Step 4: Run tests and verify GREEN**

Run selection tests and existing population tests.

---

## Self-Review

- Spec coverage: pre-eval review, disabled patterns, no fallback/repair for ARCH_SPEC, planner constraints, and training robust selection are covered.
- Scope: this is one implementation slice. It does not add probe sets, runtime heuristic portfolios, planner-created typed components, or operator-probability adaptation.
- Repository note: this checkout is not a git repository, so plan/task commits cannot be made here.
