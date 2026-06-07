---
name: controlled_variation
description: Use to implement one mechanism-level experiment as executable heuristic code.
---

# Skill: controlled_variation

## Purpose
Generate one new heuristic candidate that implements a clear mechanism hypothesis and a clear experiment intent. The output should be an experiment, not an unconstrained rewrite.

## When to use
Use after the generation planner has assigned a parent and one concrete delta. If no explicit plan is available, infer the smallest defensible experiment from compact_context, mechanism_archive, and expert_notebook.

## Inputs
- problem_card
- parent_algorithm
- parent_code
- parent_objective
- parent_mechanism_summary
- compact mechanism memory
- agent_decision.arguments

## Outputs
Return a mechanism hypothesis inside braces using these labels: Hypothesis, Mechanism family, Controlled component, Parent delta, Expected effect, Preserve, Change, and Mechanism components. Then return executable Python code for the required heuristic function. Do not return JSON.

## Procedure
Preserve the function signature exactly. Use only information allowed by the problem card. Make the smallest change that can answer the stated experiment question. The generation plan may identify a controlled_component; change that component only and keep the other mechanism components fixed. Prefer normalized or bounded terms when using numeric constants. Keep the parent code structure unless the generation plan explicitly requests a mechanism-family change.

For TSP-GLS, describe these mechanism components when relevant: edge_selection, usage_signal, distance_normalization, penalty_transform, update_strength, and stability_control. Use training_diagnosis only as evidence about what to test; do not optimize for hidden test cases, add risk scores, or add validation-set weights.

## Failure modes
Reject changes that require unavailable information, rewrite the whole heuristic without an experiment reason, repeat a recently refuted delta, or produce code identical to the parent.
