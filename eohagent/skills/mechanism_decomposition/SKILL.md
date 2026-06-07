---
name: mechanism_decomposition
description: Use before changing a parent heuristic when the agent needs an explicit mechanism experiment plan.
---

# Skill: mechanism_decomposition

## Purpose
Turn a parent heuristic and accumulated evidence into an auditable mechanism-level experiment plan. This skill does not write code. It decides what kind of design move is justified before controlled_variation is used.

## When to use
Use when the parent has not been analyzed in the current context, when evaluator evidence conflicts, when a supported mechanism may deserve refinement, or when recent failures suggest that the next step should change direction.

## Inputs
- problem_card
- parent_algorithm
- parent_code
- parent_objective
- population_summary
- mechanism_archive
- expert_notebook
- compact_context and search_feedback
- last_result

## Outputs
Return JSON only. Include:
- dominant_mechanism
- evidence_status: one of supported, uncertain, refuted, seed, or mixed
- validated_components
- uncertain_components
- refuted_components
- design_knobs
- training_failure_pattern
- mechanism_components with edge_selection, usage_signal, distance_normalization, penalty_transform, update_strength, and stability_control when identifiable
- recommended_design_mode: use the closest label from exploit_supported, refine_uncertain, repair_failed_contract, explore_alternative, abandon_refuted
- freeform_intent
- evidence_used
- next_experiment with preserve, change, avoid, expected_effect, and stop_condition

## Procedure
Read the parent thought, code, objective, training_diagnosis, hypothesis verdict, mechanism archive, and notebook. Separate mechanism evidence from implementation details. Identify which mechanism component is being tested. A supported mechanism should usually be refined before replacing its core idea. An uncertain mechanism should be tested with a small isolated change. A promising but fragile mechanism should be simplified or repaired. A stagnant, crowded, or repeatedly refuted direction can justify exploration. A clearly refuted hypothesis should not be repeated with a cosmetic rewording.

Use the design-mode labels as experiment tags, not as restrictions on algorithm creativity. If none fits perfectly, choose the closest label and explain the nuance in freeform_intent.

## Failure modes
Do not infer that a concrete mechanism is good or bad from its name alone. Use objective deltas, duplicate outcomes, contract failures, and archive history. If evidence is insufficient, mark evidence_status as uncertain and propose a small diagnostic experiment.
