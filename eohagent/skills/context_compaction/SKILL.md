---
name: context_compaction
description: Use when context grows too long or a milestone should be summarized.
---

# Skill: context_compaction

## Purpose
Compress the active context when the runtime detects that the next LLM prompt is too large.

## When to use
Use when the context runtime interrupts a skill because the estimated prompt exceeds the configured context threshold.

## Inputs
- trigger information: blocked skill, estimated tokens, context window, threshold
- parent and current candidate summaries
- current population summary
- recent session events
- existing working memory
- expert notebook summary

## Outputs
Return JSON only with:
- resume_summary
- keep_rules
- avoid_rules
- promising_mechanisms
- failed_mechanisms
- active_hypotheses
- candidate_refs
- next_search_hint

## Procedure
Keep only information needed to resume the interrupted evolutionary search. Preserve useful mechanisms, failure causes, anti-patterns, active hypotheses, and candidate ids worth referencing. Prefer short mechanism descriptions and candidate references over full historical code. Do not copy raw logs.

## Failure modes
If uncertain whether to discard an item, keep a short summary or candidate id reference. Do not expand the summary to include complete historical traces.
