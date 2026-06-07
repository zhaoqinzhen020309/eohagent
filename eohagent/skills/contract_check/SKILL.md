---
name: contract_check
description: Use after code is generated to check signature, imports, output shape, and evaluator compatibility.
---

# Skill: contract_check

## Purpose
Check whether candidate code satisfies the task interface and can execute under the evaluator.

## When to use
Use after candidate generation or repair.

## Inputs
- problem_card
- candidate_code
- expected_function_name
- expected_signature
- optional mini_test_inputs

## Outputs
Return JSON only with passed, errors, warnings, and recommended_next_skill.

## Procedure
Check syntax, function name, inputs, imports, dummy runtime behavior, output type, output shape, and NaN/inf values.

## Failure modes
If unable to execute, return passed=false with a precise runtime_error.
