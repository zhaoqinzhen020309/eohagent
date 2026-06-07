# V8 Archive Population LangMem Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the v7 fixed-emitter route experiment with a simpler v8 path where QD archive cells drive population, LLM plans choose intent adaptively, and memory is retrieved through LangMem when configured.

**Architecture:** Keep existing fail-closed ARCH_SPEC evaluation and LangGraph workflow. Add mechanism-level QD cell descriptors, QD slot metadata, plan intent/evidence gates, and a memory provider abstraction that fails closed when `provider: langmem` is configured without LangMem.

**Tech Stack:** Python unittest, existing `eohagent` modules, LangGraph, LangMem.

---

### Task 1: QD Cell Semantics

**Files:**
- Modify: `eohagent/core/qd_archive.py`
- Modify: `eohagent/tests/test_eohagent.py`

- [ ] Add tests showing ARCH_SPEC QD cells ignore `escape_policy` and `restart_policy`, but separate `distance_update_policy.type`, `penalty_policy.edge_selection`, `penalty_policy.top_k` bucket, and `schedule_policy.strength_schedule`.
- [ ] Implement a mechanism-level ARCH_SPEC descriptor.
- [ ] Verify QD archive keeps one elite per mechanism cell.

### Task 2: Archive-As-Population Engine Path

**Files:**
- Modify: `eohagent/core/engine.py`
- Modify: `eohagent/tests/test_eohagent.py`

- [ ] Add tests showing survival selection `qd_cells` uses archive elites as the full population source.
- [ ] Add QD slot metadata for planner slots: cell descriptor, intent options, evidence packet.
- [ ] Allow `qd_cells` in the ARCH_SPEC workflow contract.
- [ ] Keep route-manager code untouched for old v7 configs.

### Task 3: Harness Planner Gates

**Files:**
- Modify: `eohagent/harness/generation_planner.py`
- Modify: `eohagent/tests/test_eohagent.py`

- [ ] Add tests requiring each ARCH_SPEC plan to declare `experiment_intent` and `evidence_refs`.
- [ ] Reject unsupported intents or empty evidence refs for `exploit`, `expand`, `repair`, and `validate`.
- [ ] Pass QD slot metadata and local-tuning exhaustion evidence into the prompt.

### Task 4: LangMem Memory Provider

**Files:**
- Create: `eohagent/memory/provider.py`
- Modify: `eohagent/memory/memory_manager.py`
- Modify: `eohagent/harness/generation_planner.py`
- Modify: `eohagent/tests/test_eohagent.py`

- [ ] Add tests that `provider: langmem` imports LangMem and fails closed if unavailable.
- [ ] Store short reflection memories through the provider.
- [ ] Retrieve slot-relevant memories for planner payloads.
- [ ] Keep local provider as the explicit default for old configs.

### Task 5: V8 Config

**Files:**
- Create: `eohagent/config/tsp_gls_arch_spec_deepseek_10x20_v8.yaml`

- [ ] Copy v7 settings.
- [ ] Set output dir/name to v8.
- [ ] Set `survival_selection: qd_cells`.
- [ ] Set `parent_selection: qd_cells`.
- [ ] Set `memory.provider: langmem`.
- [ ] Keep temperature `1.0`.

### Verification

- [ ] Run targeted tests for QD archive, planner gates, memory provider, and engine parent planning.
- [ ] Run full `python -B -m unittest eohagent.tests.test_eohagent`.
- [ ] Run `python -B -m py_compile` on modified modules.
- [ ] Run CodeGraph sync/status.
