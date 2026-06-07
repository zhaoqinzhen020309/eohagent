import json
from collections import Counter
from pathlib import Path

from eohagent.core.config import resolve_path
from eohagent.memory.candidate_store import CandidateStore
from eohagent.memory.compact_context import CompactContext
from eohagent.memory.compactor import Compactor
from eohagent.memory.expert_notebook import ExpertNotebook
from eohagent.memory.budget import bounded_mechanism_archive, bounded_notebook, short_hypothesis, shorten
from eohagent.memory.mechanism_archive import MechanismArchive
from eohagent.memory.provider import build_memory_provider
from eohagent.memory.session_log import SessionLog
from eohagent.core.training_diagnostics import compact_training_diagnosis


class MemoryManager:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = resolve_path(config, config["project"]["output_dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_log = SessionLog(self.output_dir)
        self.candidate_store = CandidateStore(self.output_dir)
        self.notebook = ExpertNotebook(self.output_dir)
        self.mechanism_archive = MechanismArchive(self.output_dir)
        self.compact_context = CompactContext(self.output_dir)
        self.compactor = Compactor()
        self.memory_provider = build_memory_provider(config, self.output_dir)
        self.problem_card_path = self.output_dir / "problem_card.json"

    def save_problem_card(self, problem_card: dict) -> None:
        self.problem_card_path.write_text(json.dumps(problem_card, indent=2), encoding="utf-8")

    def load_problem_card(self) -> dict:
        return json.loads(self.problem_card_path.read_text(encoding="utf-8"))

    def summarize_population(self, population: list) -> list[dict]:
        return [
            {
                "id": c.id,
                "objective": c.objective,
                "selection_score": c.selection_score,
                "mechanism_summary": c.mechanism_summary,
                "mechanism_hypothesis": c.mechanism_hypothesis,
                "mechanism_profile": c.mechanism_profile,
                "training_diagnosis": compact_training_diagnosis((c.evidence or {}).get("training_diagnosis")),
                "hypothesis_verdict": c.hypothesis_verdict,
                "arch_spec": (c.metadata or {}).get("arch_spec") if isinstance(c.metadata, dict) else None,
                "solver_tree": (c.metadata or {}).get("solver_tree") if isinstance(c.metadata, dict) else None,
            }
            for c in population
        ]

    def build_context_for_agent(self, state) -> dict:
        runtime = self.config.get("context_management", {})
        recent_limit = min(int(runtime.get("recent_events", 12)), 6)
        population_items = min(int(runtime.get("population_items", 8)), 6)
        notebook_items = min(int(runtime.get("notebook_items", 6)), 3)
        recent_events = self.session_log.recent(recent_limit)
        compact_context = state.compact_context or self.compact_context.read()
        notebook = self.notebook.read()
        return {
            "parent": {
                "id": state.parent.id,
                "objective": state.parent.objective,
                "mechanism_summary": str(state.parent.mechanism_summary or "")[:300],
                "mechanism_hypothesis": short_hypothesis(state.parent.mechanism_hypothesis),
                "hypothesis_verdict": state.parent.hypothesis_verdict,
                "algorithm": str(state.parent.algorithm or "")[:500],
                "code": state.parent.code,
            },
            "population_summary": self._prompt_population_summary(state.population_summary, population_items),
            "working_memory": self._working_context(compact_context),
            "expert_notebook": bounded_notebook(notebook, notebook_items),
            "mechanism_archive": bounded_mechanism_archive(self.mechanism_archive.read(), per_bucket=notebook_items),
            "search_feedback": self._search_feedback(state.population_summary, recent_events),
            "step_count": state.step_count,
            "last_result": shorten(state.last_result, 500),
        }

    def compact_if_needed(self, population: list) -> None:
        if self.config["memory"].get("enable_compaction", True):
            limit = int(self.config["memory"].get("keep_recent_events", 50))
            self.compactor.compact(self, population, self.session_log.recent(limit))

    def record_memory(self, kind: str, content: str, metadata: dict | None = None) -> None:
        self.memory_provider.record(kind, content, metadata or {})

    def record_experiment_memory(
        self,
        kind: str,
        candidate_id: str,
        generation: int,
        tree_descriptor,
        verdict: str,
        evidence_source: dict,
        content: str,
    ) -> dict:
        descriptor = [str(item) for item in tuple(tree_descriptor)]
        record = {
            "kind": str(kind),
            "candidate_id": str(candidate_id),
            "generation": int(generation),
            "tree_descriptor": descriptor,
            "verdict": str(verdict),
            "evidence_source": evidence_source if isinstance(evidence_source, dict) else {},
            "content": str(content),
        }
        self.record_memory(
            str(kind),
            json.dumps(record, ensure_ascii=True, sort_keys=True),
            {
                "candidate_id": record["candidate_id"],
                "generation": record["generation"],
                "tree_descriptor": record["tree_descriptor"],
                "verdict": record["verdict"],
                "source": "typed_experiment_memory",
            },
        )
        return record

    def retrieve_memories(self, query: str, limit: int = 5) -> list[dict]:
        return self.memory_provider.search(query, limit=limit)

    def record_reflection_memory(self, generation: int, reflection: dict) -> None:
        if not isinstance(reflection, dict):
            return
        for key in [
            "validated_principles",
            "failed_hypotheses",
            "supported_deltas",
            "refuted_deltas",
            "failure_patterns",
            "family_status",
            "next_generation_priorities",
        ]:
            values = reflection.get(key)
            if not isinstance(values, list):
                continue
            for value in values[:6]:
                self.record_memory(
                    key,
                    str(value)[:900],
                    {"generation": generation, "source": "generation_reflection"},
                )

    @staticmethod
    def _search_feedback(population_summary: list[dict], recent_events: list[dict]) -> dict:
        objectives = [p.get("objective") for p in population_summary if p.get("objective") is not None]
        rounded_objectives = [round(float(obj), 5) for obj in objectives]
        families = [
            ((p.get("mechanism_hypothesis") or {}).get("mechanism_family") if isinstance(p.get("mechanism_hypothesis"), dict) else None)
            or ((p.get("mechanism_profile") or {}).get("mechanism_family") if isinstance(p.get("mechanism_profile"), dict) else None)
            or "unspecified"
            for p in population_summary
        ]
        event_counts = Counter(event.get("event") for event in recent_events)
        skill_counts = Counter(event.get("skill") for event in recent_events if event.get("event") == "skill_called")
        formal_eval_objectives = [
            round(float(event["objective"]), 5)
            for event in recent_events
            if event.get("event") == "formal_eval" and event.get("objective") is not None
        ]
        failure_signatures = [
            (event.get("training_diagnosis") or {}).get("failure_signature")
            for event in recent_events
            if isinstance(event.get("training_diagnosis"), dict)
        ]
        failures = [
            {
                "event": event.get("event"),
                "skill": event.get("skill"),
                "candidate_id": event.get("candidate_id"),
                "errors": event.get("errors") or event.get("error"),
            }
            for event in recent_events
            if event.get("event") in {"candidate_duplicate", "objective_duplicate", "formal_eval_failed", "candidate_rejected"}
            or event.get("contract_passed") is False
        ][-8:]
        return {
            "population_size": len(population_summary),
            "unique_population_objectives": len(set(rounded_objectives)),
            "population_objective_counts": dict(Counter(rounded_objectives)),
            "population_family_counts": dict(Counter(families)),
            "best_objective": min(rounded_objectives) if rounded_objectives else None,
            "recent_event_counts": dict(event_counts),
            "recent_skill_counts": dict(skill_counts),
            "recent_formal_eval_count": event_counts.get("formal_eval", 0),
            "recent_duplicate_count": event_counts.get("candidate_duplicate", 0) + event_counts.get("objective_duplicate", 0),
            "recent_objective_counts": dict(Counter(formal_eval_objectives)),
            "recent_failure_signature_counts": dict(Counter(sig for sig in failure_signatures if sig)),
            "recent_failures": failures,
            "diagnosis": MemoryManager._diagnose_search(rounded_objectives, event_counts, skill_counts, families),
        }

    @staticmethod
    def _diagnose_search(objectives: list[float], event_counts: Counter, skill_counts: Counter, families: list[str] | None = None) -> list[str]:
        diagnosis = []
        if objectives and len(set(objectives)) <= 1:
            diagnosis.append("population_objective_diversity_is_low")
        if families:
            counts = Counter(families)
            if counts and counts.most_common(1)[0][1] > max(2, len(families) // 2):
                diagnosis.append("population_mechanism_family_is_crowded")
        if event_counts.get("candidate_duplicate", 0) + event_counts.get("objective_duplicate", 0) > event_counts.get("formal_eval", 0):
            diagnosis.append("recent_duplicates_exceed_successful_evaluations")
        if skill_counts.get("repair", 0) > skill_counts.get("controlled_variation", 0):
            diagnosis.append("repair_calls_exceed_variation_calls")
        if not diagnosis:
            diagnosis.append("no_obvious_search_stagnation_signal")
        return diagnosis

    @staticmethod
    def _prompt_population_summary(population_summary: list[dict], limit: int = 6) -> list[dict]:
        return [
            {
                "id": item.get("id"),
                "objective": item.get("objective"),
                "selection_score": item.get("selection_score"),
                "mechanism_summary": str(item.get("mechanism_summary", ""))[:220],
                "mechanism_hypothesis": short_hypothesis(item.get("mechanism_hypothesis")),
                "training_diagnosis": compact_training_diagnosis(item.get("training_diagnosis")),
                "hypothesis_verdict": item.get("hypothesis_verdict"),
                "arch_spec": item.get("arch_spec"),
            }
            for item in (population_summary or [])[:limit]
        ]

    @staticmethod
    def _working_context(compact_context: dict) -> dict:
        if not isinstance(compact_context, dict):
            return {}
        if isinstance(compact_context.get("working_memory"), dict):
            return {
                "current_best": compact_context.get("current_best"),
                "working_memory": compact_context.get("working_memory"),
                "search_feedback": compact_context.get("search_feedback"),
                "mechanism_archive": compact_context.get("mechanism_archive"),
                "next_focus": compact_context.get("next_focus"),
            }
        return {
            "current_best": compact_context.get("current_best"),
            "recent_successes": compact_context.get("recent_successes", [])[-3:],
            "recent_failures": compact_context.get("recent_failures", [])[-3:],
            "design_lessons": compact_context.get("design_lessons", [])[-6:],
            "mechanism_cards": compact_context.get("mechanism_cards", [])[-4:],
            "delta_cards": compact_context.get("delta_cards", [])[-4:],
            "generation_attributions": compact_context.get("generation_attributions", [])[-4:],
            "anti_patterns": compact_context.get("anti_patterns", [])[-6:],
            "mechanism_archive": compact_context.get("mechanism_archive"),
            "next_focus": compact_context.get("next_focus", [])[-3:],
        }

    @staticmethod
    def _shorten(value):
        if isinstance(value, str):
            return value[:1000]
        if isinstance(value, list):
            return [MemoryManager._shorten(item) for item in value[:8]]
        if isinstance(value, dict):
            return {
                key: MemoryManager._shorten(val)
                for key, val in value.items()
                if key != "raw_response"
            }
        return value
