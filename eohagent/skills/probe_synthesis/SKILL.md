---
name: probe_synthesis
description: Use to test whether a mechanism hypothesis satisfies cheap problem invariances and stress properties.
---

# Skill: probe_synthesis

## Purpose
Create or select lightweight stress probes that test the mechanism hypothesis before expensive formal evaluation. Probes should produce evidence about the hypothesis, not just repeat syntax checks.

## When to use
Use after contract_check and before formal evaluation, especially when a candidate uses a new mechanism, changes a supported mechanism knob, has uncertain evidence, or needs a cheap robustness check.

## Inputs
- problem_card
- candidate_code
- parent_code
- mechanism_hypothesis
- agent_decision.arguments
- expert_notebook
- mechanism_archive
- available_training_instances_summary

## Outputs
For TSP-GLS the harness executes built-in probes and returns probe_results plus probe_penalty. When the LLM proposes focus areas, return JSON with probe_focus or probe_focuses so the harness can prioritize mechanism-relevant probes. For other problems return JSON only with a probes array containing probe_name, transformation, expected_property, cost, and reason.

## Procedure
Use problem invariances, small synthetic boundary cases, or cheap transformations of existing data. Choose probes that match the mechanism family and hypothesis:

- Normalization or scale-aware mechanisms: test distance or input scaling invariance.
- Length-aware mechanisms: test whether equally used long and short edges receive the expected relative pressure without extreme dominance.
- Rank or top-k mechanisms: test threshold stability, ties, and small-instance behavior.
- Saturation or bounded-growth mechanisms: test extreme usage counts for monotonic but non-explosive behavior.
- Ratio or utility mechanisms: test denominator stability and short-edge edge cases.

Always include the expected property and why it supports or refutes the mechanism hypothesis. A probe failure should suggest a repair target, not automatically condemn the whole mechanism family.

## Failure modes
Do not add a large benchmark, hidden test set, or expensive evaluator call. If no safe probe can be defined, return an empty probes list and explain the missing evidence.
