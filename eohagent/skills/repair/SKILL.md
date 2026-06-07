---
name: repair
description: Use when candidate code fails contract or runtime checks.
---

# Skill: repair

## Purpose
Repair a candidate based on precise failure reports while preserving its intended mechanism.

## When to use
Use after contract_check or formal evaluation reports a fixable issue.

## Inputs
- problem_card
- candidate_algorithm
- candidate_code
- failure_report
- parent_code
- expert_notebook

## Outputs
Return a one-sentence repair description inside braces, then executable Python code for the repaired heuristic function. The harness extracts the description as repaired_algorithm and the code as repaired_code.

## Procedure
Identify the exact failure, keep the signature unchanged, fix the smallest necessary part, and simplify or revert when the mechanism caused the failure.

## Failure modes
If more than two repairs fail, recommend reject.
