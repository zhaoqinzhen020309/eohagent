from __future__ import annotations

import json
import time
from copy import deepcopy

from eohagent.memory.budget import bounded_mechanism_archive, bounded_notebook, compact_candidate, short_text, shorten


class ContextOverflowError(RuntimeError):
    """Raised when a prompt remains over budget after forced compaction."""


class ContextRuntime:
    def __init__(self, config: dict, memory, client):
        self.config = config
        self.memory = memory
        self.client = client
        runtime_config = config.get("context_management", {})
        self.enabled = bool(runtime_config.get("enable", True))
        self.context_window = int(runtime_config.get("context_window", 32768))
        self.compact_threshold = float(runtime_config.get("compact_threshold", 0.70))
        self.chars_per_token = float(runtime_config.get("chars_per_token", 3.0))
        self.max_compaction_retries = int(runtime_config.get("max_compaction_retries", 2))
        self.recent_events = int(runtime_config.get("recent_events", 12))
        self.population_items = int(runtime_config.get("population_items", 8))
        self.notebook_items = int(runtime_config.get("notebook_items", 6))
        self.use_llm_compaction = bool(runtime_config.get("use_llm_compaction", False))
        self.record_memory = True
        self.event_sink = None

    def configure(self, record_memory: bool = True, event_sink=None):
        previous = (self.record_memory, self.event_sink)
        self.record_memory = record_memory
        self.event_sink = event_sink
        return previous

    def restore(self, previous) -> None:
        self.record_memory, self.event_sink = previous

    def ensure_prompt(self, purpose: str, prompt: str, rebuild_prompt, state=None) -> str:
        if not self.enabled or purpose == "context_compaction":
            return prompt

        current_prompt = prompt
        stats = self.prompt_stats(current_prompt)
        if stats["pressure"] < self.compact_threshold:
            return current_prompt
        if state is None:
            self._raise_overflow(purpose, stats, "no_state_for_bounded_context")

        self._emit({"event": "context_pressure", "purpose": purpose, **stats, "action": "deterministic_shrink"})
        print(
            f"    [context] purpose={purpose} estimated_tokens={stats['estimated_tokens']} "
            f"pressure={stats['pressure']:.2f} action=deterministic_shrink",
            flush=True,
        )
        self.apply_bounded_context(state, purpose=purpose, mode="bounded", stats=stats)
        current_prompt = rebuild_prompt()
        bounded_stats = self.prompt_stats(current_prompt)
        if bounded_stats["pressure"] < self.compact_threshold:
            return current_prompt

        if self.use_llm_compaction:
            self.compact(purpose=purpose, state=state, stats=bounded_stats)
            current_prompt = rebuild_prompt()
            llm_stats = self.prompt_stats(current_prompt)
            if llm_stats["pressure"] < self.compact_threshold:
                return current_prompt
            bounded_stats = llm_stats

        self._emit({"event": "context_pressure", "purpose": purpose, **bounded_stats, "action": "emergency_context"})
        print(
            f"    [context] purpose={purpose} estimated_tokens={bounded_stats['estimated_tokens']} "
            f"pressure={bounded_stats['pressure']:.2f} action=emergency_context",
            flush=True,
        )
        self.apply_bounded_context(state, purpose=purpose, mode="emergency", stats=bounded_stats)
        current_prompt = rebuild_prompt()
        emergency_stats = self.prompt_stats(current_prompt)
        if emergency_stats["pressure"] < self.compact_threshold:
            return current_prompt
        self._raise_overflow(purpose, emergency_stats, "failed_after_emergency_context")
        return current_prompt

    def prompt_stats(self, prompt: str) -> dict:
        estimated_tokens = max(1, int(len(prompt) / max(self.chars_per_token, 0.1)))
        pressure = estimated_tokens / max(self.context_window, 1)
        return {
            "estimated_tokens": estimated_tokens,
            "context_window": self.context_window,
            "pressure": round(pressure, 4),
            "prompt_chars": len(prompt),
        }

    def _raise_overflow(self, purpose: str, stats: dict, action: str) -> None:
        self._emit({"event": "context_overflow", "purpose": purpose, **stats, "action": action})
        raise ContextOverflowError(
            f"context pressure {stats['pressure']:.2f} remains above threshold "
            f"{self.compact_threshold:.2f} for {purpose}"
        )

    def apply_bounded_context(self, state, purpose: str, mode: str, stats: dict) -> dict:
        context = self._bounded_context(state, mode=mode)
        state.compact_context = context
        state.notebook = context.get("expert_notebook", {})
        state.mechanism_archive = context.get("mechanism_archive", {})
        if self.record_memory:
            self.memory.compact_context.write(context)
        self._emit(
            {
                "event": "context_compaction",
                "purpose": purpose,
                "strategy": mode,
                "estimated_tokens_before": stats["estimated_tokens"],
                "pressure_before": stats["pressure"],
                "working_memory": context.get("working_memory", {}),
            }
        )
        return context

    def _bounded_context(self, state, mode: str) -> dict:
        existing = deepcopy(state.compact_context or self.memory.compact_context.read())
        notebook_items = 1 if mode == "emergency" else min(self.notebook_items, 3)
        archive_items = 1 if mode == "emergency" else 3
        population_items = 2 if mode == "emergency" else min(self.population_items, 6)
        recent_failures = 1 if mode == "emergency" else 3
        notebook = bounded_notebook(self.memory.notebook.read(), notebook_items, text_limit=220)
        mechanism_archive = bounded_mechanism_archive(
            self.memory.mechanism_archive.read(),
            per_bucket=archive_items,
            family_limit=4 if mode == "emergency" else 8,
            text_limit=220,
        )
        current_best = existing.get("current_best") or compact_candidate(state.parent, include_code=False)
        return {
            "problem_card": existing.get("problem_card"),
            "current_best": shorten(current_best, 350),
            "working_memory": {
                "resume_summary": short_text(self._working_memory(existing).get("resume_summary"), 600 if mode == "bounded" else 240),
                "keep_rules": self._working_memory(existing).get("keep_rules", [])[-notebook_items:],
                "avoid_rules": self._working_memory(existing).get("avoid_rules", [])[-notebook_items:],
                "promising_mechanisms": self._working_memory(existing).get("promising_mechanisms", [])[-archive_items:],
                "failed_mechanisms": self._working_memory(existing).get("failed_mechanisms", [])[-archive_items:],
                "active_hypotheses": self._working_memory(existing).get("active_hypotheses", [])[-archive_items:],
                "candidate_refs": [state.parent.id] if getattr(state.parent, "id", None) else [],
                "next_search_hint": short_text(self._working_memory(existing).get("next_search_hint"), 240),
            },
            "expert_notebook": notebook,
            "mechanism_archive": mechanism_archive,
            "population_summary": state.population_summary[:population_items],
            "recent_failures": existing.get("recent_failures", [])[-recent_failures:],
            "next_focus": existing.get("next_focus", [])[-recent_failures:],
            "context_management": {
                "last_compacted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "strategy": mode,
                "threshold": self.compact_threshold,
                "context_window": self.context_window,
            },
        }

    def compact(self, purpose: str, state, stats: dict) -> dict:
        if state is None:
            raise ContextOverflowError(f"context compaction for {purpose} requires agent state")

        payload = self._compaction_payload(purpose, state, stats)
        system, user = self._build_compaction_prompt(payload)
        if hasattr(self.client, "prepare_json_prompt"):
            user = self.client.prepare_json_prompt(payload, [])
        elif hasattr(self.client, "prepare_prompt"):
            user = self.client.prepare_prompt(user, purpose="context_compaction")
        response = self.client.complete(user, system=system)
        data = self._extract_json(response)
        compacted = self._normalize_compaction(data)
        new_context = self._merge_compaction(state, compacted)
        state.compact_context = new_context
        if self.record_memory:
            self.memory.compact_context.write(new_context)
        self._emit(
            {
                "event": "context_compaction",
                "purpose": purpose,
                "estimated_tokens_before": stats["estimated_tokens"],
                "pressure_before": stats["pressure"],
                "resume_summary": compacted.get("resume_summary", ""),
                "candidate_refs": compacted.get("candidate_refs", []),
                "working_memory": compacted,
                "llm_usage": getattr(self.client, "last_usage", {}) or {},
                "prompt_compression": getattr(self.client, "last_prompt_compression", {}) or {},
            }
        )
        print(
            f"    [context] compaction_done purpose={purpose} "
            f"resume_chars={len(compacted.get('resume_summary', ''))}",
            flush=True,
        )
        return new_context

    def _compaction_payload(self, purpose: str, state, stats: dict) -> dict:
        recent_events = self.memory.session_log.recent(self.recent_events)
        notebook = self.memory.notebook.read()
        mechanism_archive = self.memory.mechanism_archive.read()
        compact_context = state.compact_context or self.memory.compact_context.read()
        return {
            "purpose": "context_compaction",
            "trigger": {
                "blocked_purpose": purpose,
                "estimated_tokens": stats["estimated_tokens"],
                "context_window": stats["context_window"],
                "pressure": stats["pressure"],
                "threshold": self.compact_threshold,
            },
            "problem": self.config.get("project", {}).get("problem"),
            "function_name": self.config.get("evaluator", {}).get("function_name"),
            "parent": self._candidate_ref(state.parent, include_code=True),
            "current_candidate": self._candidate_ref(state.current_candidate, include_code=True)
            if state.current_candidate
            else None,
            "population_summary": state.population_summary[: min(self.population_items, 6)],
            "recent_events": [self._event_summary(event) for event in recent_events],
            "existing_working_memory": self._working_memory(compact_context),
            "notebook_summary": bounded_notebook(notebook, min(self.notebook_items, 3)),
            "mechanism_archive": bounded_mechanism_archive(mechanism_archive, per_bucket=3, family_limit=8),
            "last_result": shorten(state.last_result, 500),
            "instruction": (
                "Compress only the information needed to continue this evolutionary heuristic search. "
                "Keep references to candidate ids instead of copying full old code. "
                "Preserve useful mechanisms, failed mechanisms, anti-patterns, and the next search hint."
            ),
        }

    @staticmethod
    def _build_compaction_prompt(payload: dict):
        system = (
            "You are the context_compaction skill inside an expert evolutionary-search agent.\n"
            "Return exactly one JSON object with these keys: resume_summary, keep_rules, avoid_rules, "
            "promising_mechanisms, failed_mechanisms, active_hypotheses, candidate_refs, next_search_hint.\n"
            "Keep the result compact. Do not include markdown, comments, or full historical code."
        )
        user = json.dumps(payload, ensure_ascii=True)
        return system, user

    @staticmethod
    def _extract_json(response: str) -> dict:
        from eohagent.llm.json_call import JSONCall

        return JSONCall.extract_json(response)

    def _normalize_compaction(self, data: dict) -> dict:
        normalized = {}
        normalized["resume_summary"] = str(data.get("resume_summary") or data.get("summary") or "")[:1800]
        for key in [
            "keep_rules",
            "avoid_rules",
            "promising_mechanisms",
            "failed_mechanisms",
            "active_hypotheses",
            "candidate_refs",
        ]:
            value = data.get(key, [])
            if isinstance(value, str):
                value = [value]
            if not isinstance(value, list):
                value = []
            normalized[key] = [self._short_string(item) for item in value[:8]]
        normalized["next_search_hint"] = str(data.get("next_search_hint") or "")[:600]
        return normalized

    def _merge_compaction(self, state, compacted: dict) -> dict:
        existing = deepcopy(state.compact_context or self.memory.compact_context.read())
        problem_card = existing.get("problem_card")
        current_best = existing.get("current_best")
        new_context = {
            "problem_card": problem_card,
            "current_best": current_best,
            "working_memory": compacted,
            "context_management": {
                "last_compacted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "threshold": self.compact_threshold,
                "context_window": self.context_window,
            },
        }
        for key in ["search_feedback", "recent_successes", "recent_failures", "next_focus", "mechanism_archive"]:
            if key in existing:
                new_context[key] = existing[key]
        return new_context

    def _working_memory(self, compact_context: dict) -> dict:
        if not isinstance(compact_context, dict):
            return {}
        working = compact_context.get("working_memory")
        if isinstance(working, dict):
            return working
        return {
            "resume_summary": compact_context.get("resume_summary")
            or compact_context.get("summary")
            or compact_context.get("current_best"),
            "keep_rules": compact_context.get("design_lessons", [])[-self.notebook_items :],
            "avoid_rules": compact_context.get("anti_patterns", [])[-self.notebook_items :],
            "promising_mechanisms": compact_context.get("recent_successes", [])[-3:],
            "failed_mechanisms": compact_context.get("recent_failures", [])[-3:],
            "active_hypotheses": compact_context.get("next_focus", [])[-3:],
            "candidate_refs": [],
            "next_search_hint": "Use compacted search feedback to continue.",
        }

    def _notebook_summary(self, notebook: dict) -> dict:
        if not isinstance(notebook, dict):
            return {}
        return {
            "validated_principles": notebook.get("validated_principles", [])[-self.notebook_items :],
            "anti_patterns": notebook.get("anti_patterns", [])[-self.notebook_items :],
            "failed_hypotheses": notebook.get("failed_hypotheses", [])[-self.notebook_items :],
            "open_questions": notebook.get("open_questions", [])[-self.notebook_items :],
        }

    def _candidate_ref(self, candidate, include_code: bool = False) -> dict:
        if candidate is None:
            return {}
        return compact_candidate(candidate, include_code=include_code, code_limit=2200)

    def _event_summary(self, event: dict) -> dict:
        keys = ["event", "skill", "result_event", "candidate_id", "objective", "contract_passed", "error", "errors"]
        return {key: self._shorten(event.get(key)) for key in keys if event.get(key) is not None}

    def _shorten(self, value):
        if isinstance(value, str):
            return self._truncate(value, 500)
        if isinstance(value, list):
            return [self._shorten(item) for item in value[:8]]
        if isinstance(value, dict):
            return {key: self._shorten(val) for key, val in list(value.items())[:12]}
        return value

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if value is None:
            return ""
        value = str(value)
        return value if len(value) <= limit else value[:limit] + "...[truncated]"

    def _short_string(self, value) -> str:
        return self._truncate(json.dumps(value, ensure_ascii=True) if isinstance(value, (dict, list)) else str(value), 500)

    def _emit(self, event: dict) -> None:
        if self.event_sink is not None:
            self.event_sink(event)
        elif self.record_memory:
            self.memory.session_log.append(event)
