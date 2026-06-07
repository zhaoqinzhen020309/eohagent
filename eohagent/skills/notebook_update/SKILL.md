---
name: notebook_update
description: Use after candidate outcomes to convert raw results into reusable mechanism-level design memory.
---

# Skill: notebook_update

## Purpose
Keep expert_notebook as a compact expert design memory, not as a candidate log. Summaries should help the next agent decision choose whether to refine, repair, explore, or abandon a mechanism direction.

## When to use
Use after evaluated candidates, rejected candidates, duplicate outcomes, repair outcomes, or repeated failures. Also use after a generation when the population shows a clear mechanism pattern.

## Inputs
- candidate_outcome
- expert_notebook
- mechanism_archive
- compact_context
- recent_session_events
- evaluator evidence, probe evidence, and audit warnings when available

## Outputs
Return JSON only. Prefer these keys:
- lessons: short reusable mechanism principles
- mechanism_cards: mechanism-level evidence cards with hypothesis, family, training evidence, supported/refuted evidence, and next action
- delta_cards: controlled-component evidence cards with component, delta, outcome, evidence, and recommendation
- generation_attributions: generation-level conclusions about which controlled changes helped or hurt
- failed_hypotheses: mechanism-level failure summaries
- anti_patterns: repeated design mistakes to avoid
- successful_repairs: repair patterns that preserved a useful mechanism
- open_questions: focused questions for future experiments
- search_policy_hints: concise guidance such as refine supported mechanism, test uncertain knob, repair risk, or explore alternative

Each item should be short. Mention candidate ids only as evidence references, not as the main lesson.

## Procedure
Convert outcomes into verbal gradients. For a supported candidate, summarize the mechanism principle, training_diagnosis evidence, validated components, and one next refinement. For a refuted candidate, summarize the failed controlled component and likely reason without declaring the whole mechanism family bad. For duplicates, identify whether the design intent was not distinct enough. At generation end, compare variants inside the same experiment group and attribute which isolated delta helped or hurt.

Do not append raw event history. Merge repeated lessons into existing principles by increasing confidence or adding evidence references. Keep notebook memory bounded: prefer a stronger merged lesson over many similar candidate notes.

## Failure modes
If an event has no reusable design information, return empty arrays. Do not create a lesson from a single noisy outcome unless the mechanism evidence is clear.
