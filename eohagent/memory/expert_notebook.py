import json
from pathlib import Path


class ExpertNotebook:
    def __init__(self, output_dir: Path):
        self.path = output_dir / "expert_notebook.json"

    def default(self) -> dict:
        return {
            "validated_principles": [
                "Start from a simple baseline-like mechanism.",
                "Treat each code candidate as an implementation of an explicit mechanism hypothesis.",
                "Use training-diagnosis evidence to decide whether a mechanism should be refined, compared, or abandoned.",
            ],
            "mechanism_cards": [],
            "delta_cards": [],
            "generation_attributions": [],
            "anti_patterns": [
                "Unexplained hard-coded constants",
                "Many unrelated nonlinear terms",
                "Branching rules tied to a narrow training distribution",
                "Candidate records without a reusable mechanism-level lesson",
            ],
            "successful_repairs": [],
            "failed_hypotheses": [],
            "open_questions": ["Can a small correction term improve the baseline without damaging robustness?"],
        }

    def write_default(self) -> None:
        if not self.path.exists():
            self.write(self.default())

    def read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else self.default()

    def write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def merge_update(self, update: dict) -> None:
        data = self.read()
        key_map = {
            "lessons": "validated_principles",
            "design_lessons": "validated_principles",
            "validated_principles": "validated_principles",
            "mechanism_cards": "mechanism_cards",
            "delta_cards": "delta_cards",
            "generation_attributions": "generation_attributions",
            "supported_deltas": "delta_cards",
            "refuted_deltas": "delta_cards",
            "ambiguous_deltas": "delta_cards",
            "failure_patterns": "generation_attributions",
            "family_status": "generation_attributions",
            "next_generation_priorities": "open_questions",
            "anti_patterns": "anti_patterns",
            "successful_repairs": "successful_repairs",
            "failed_hypotheses": "failed_hypotheses",
            "open_questions": "open_questions",
        }
        for source_key, target_key in key_map.items():
            data.setdefault(target_key, [])
            for item in update.get(source_key, []) or []:
                if item not in data[target_key]:
                    data[target_key].append(item)
        self._cap(data, limit=100)
        self.write(data)

    def add_lesson_from_candidate(self, candidate, state=None, outcome: str = "observed") -> None:
        if candidate is None:
            return
        lesson = self._mechanism_lesson(candidate, outcome)
        data = self.read()
        if outcome == "evaluated" and candidate.objective is not None and candidate.hypothesis_verdict == "supported":
            target = "mechanism_cards"
        elif outcome == "repair":
            target = "successful_repairs"
        else:
            target = "delta_cards"
        data.setdefault(target, [])
        if lesson not in data[target]:
            data[target].append(lesson)
        if outcome in {"duplicate", "objective_duplicate"}:
            data.setdefault("anti_patterns", [])
            pattern = f"{outcome}: candidate did not add a distinct EoH population member"
            if pattern not in data["anti_patterns"]:
                data["anti_patterns"].append(pattern)
        self.write(data)

    def add_generation_reflection(self, generation: int, population: list, offspring: list) -> None:
        if not offspring:
            return
        data = self.read()
        supported = [
            c for c in offspring
            if c.objective is not None and c.hypothesis_verdict == "supported"
        ]
        refuted = [
            c for c in offspring
            if c.hypothesis_verdict == "refuted" or c.status in {"eval_failed", "objective_duplicate", "duplicate"}
        ]
        if supported:
            best = min(supported, key=lambda c: c.selection_score if c.selection_score is not None else c.objective)
            principle = self._design_principle(best, generation)
            self._append_unique(data.setdefault("generation_attributions", []), principle)
        for candidate in refuted[:3]:
            self._append_unique(data.setdefault("delta_cards", []), self._mechanism_lesson(candidate, candidate.status or "refuted"))
        family_counts = {}
        for candidate in population:
            family = self._family(candidate)
            family_counts[family] = family_counts.get(family, 0) + 1
        crowded = [family for family, count in family_counts.items() if count > max(2, len(population) // 3)]
        for family in crowded:
            self._append_unique(
                data.setdefault("generation_attributions", []),
                f"Generation {generation}: population is crowded in {family}; test a different mechanism family before further local tuning.",
            )
        self._cap(data, limit=80)
        self.write(data)

    @staticmethod
    def _mechanism_lesson(candidate, outcome: str) -> dict:
        hypothesis = candidate.mechanism_hypothesis or {}
        text = hypothesis.get("text") or candidate.mechanism_summary or candidate.algorithm
        family = hypothesis.get("mechanism_family") or "unspecified"
        delta = None
        diagnosis = {}
        if isinstance(candidate.evidence, dict):
            delta = candidate.evidence.get("delta_vs_parent")
            diagnosis = candidate.evidence.get("training_diagnosis") or {}
        plan = candidate.metadata.get("generation_plan") if isinstance(candidate.metadata, dict) else {}
        return {
            "candidate_id": candidate.id,
            "outcome": outcome,
            "summary": f"{outcome}: {candidate.id}; family={family}; hypothesis={str(text)[:160]}",
            "family": family,
            "hypothesis": str(text)[:260],
            "controlled_component": plan.get("controlled_component") if isinstance(plan, dict) else None,
            "delta": hypothesis.get("parent_delta") or (plan.get("delta") if isinstance(plan, dict) else None),
            "objective": candidate.objective,
            "delta_vs_parent": delta,
            "verdict": candidate.hypothesis_verdict,
            "training_diagnosis": {
                "tail_gap": diagnosis.get("tail_gap"),
                "failure_signature": diagnosis.get("failure_signature"),
                "worst_training_instances": (diagnosis.get("worst_training_instances") or [])[:3],
            },
        }

    @staticmethod
    def _design_principle(candidate, generation: int) -> dict:
        hypothesis = candidate.mechanism_hypothesis or {}
        family = hypothesis.get("mechanism_family") or "unspecified"
        text = hypothesis.get("text") or candidate.mechanism_summary or candidate.algorithm
        evidence = candidate.evidence or {}
        evaluator = evidence.get("formal_evaluator", {}) if isinstance(evidence, dict) else {}
        diagnosis = evidence.get("training_diagnosis", {}) if isinstance(evidence, dict) else {}
        return {
            "generation": generation,
            "summary": f"Generation {generation}: supported family={family}; hypothesis={str(text)[:160]}",
            "family": family,
            "hypothesis": str(text)[:220],
            "objective": candidate.objective,
            "delta_vs_parent": evidence.get("delta_vs_parent") if isinstance(evidence, dict) else None,
            "std_gap": evaluator.get("std_gap") if isinstance(evaluator, dict) else None,
            "tail_gap": diagnosis.get("tail_gap") if isinstance(diagnosis, dict) else None,
            "failure_signature": diagnosis.get("failure_signature") if isinstance(diagnosis, dict) else None,
        }

    @staticmethod
    def _family(candidate) -> str:
        hypothesis = candidate.mechanism_hypothesis or {}
        if isinstance(hypothesis, dict):
            return hypothesis.get("mechanism_family") or "unspecified"
        return "unspecified"

    @staticmethod
    def _append_unique(items: list, value: str) -> None:
        if value and value not in items:
            items.append(value)

    @staticmethod
    def _cap(data: dict, limit: int) -> None:
        for key in [
            "validated_principles",
            "mechanism_cards",
            "delta_cards",
            "generation_attributions",
            "anti_patterns",
            "successful_repairs",
            "failed_hypotheses",
            "open_questions",
        ]:
            if isinstance(data.get(key), list) and len(data[key]) > limit:
                data[key] = data[key][-limit:]
