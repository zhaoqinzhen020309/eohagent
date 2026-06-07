import json
from pathlib import Path

from eohagent.core.mechanism import canonicalize_family
from eohagent.core.training_diagnostics import compact_training_diagnosis


class MechanismArchive:
    def __init__(self, output_dir: Path):
        self.path = output_dir / "mechanism_archive.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.write(self.default())

    @staticmethod
    def default() -> dict:
        return {
            "active_hypotheses": [],
            "supported_hypotheses": [],
            "refuted_hypotheses": [],
            "uncertain_hypotheses": [],
            "mechanism_families": {},
            "architecture_tree": [],
        }

    def read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else self.default()

    def write(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def record_candidate(self, candidate, outcome: str) -> None:
        hypothesis = getattr(candidate, "mechanism_hypothesis", None) or {}
        text = hypothesis.get("text") or getattr(candidate, "mechanism_summary", None)
        if not text:
            return
        family = canonicalize_family(hypothesis.get("mechanism_family"), text)
        entry = {
            "candidate_id": candidate.id,
            "hypothesis_id": hypothesis.get("id"),
            "text": text,
            "mechanism_family": family,
            "mechanism_components": hypothesis.get("mechanism_components")
            or (getattr(candidate, "mechanism_profile", None) or {}).get("mechanism_components"),
            "parent_delta": hypothesis.get("parent_delta") or getattr(candidate, "parent_delta", None),
            "expected_effect": hypothesis.get("expected_effect"),
            "objective": candidate.objective,
            "selection_score": candidate.selection_score,
            "evaluation_evidence": (candidate.metadata or {}).get("evaluation_evidence"),
            "training_diagnosis": compact_training_diagnosis((candidate.metadata or {}).get("training_diagnosis")),
            "verdict": getattr(candidate, "hypothesis_verdict", None),
            "outcome": outcome,
        }
        data = self.read()
        target = self._target_list(candidate, outcome)
        self._append_unique(data.setdefault(target, []), entry)
        families = data.setdefault("mechanism_families", {})
        family_data = families.setdefault(
            family,
            {
                "count": 0,
                "best_objective": None,
                "best_candidate_id": None,
                "best_selection_score": None,
                "supported": 0,
                "refuted": 0,
                "uncertain": 0,
                "recent_candidates": [],
            },
        )
        family_data["count"] += 1
        verdict = entry.get("verdict") or "uncertain"
        if verdict in {"supported", "refuted", "uncertain"}:
            family_data[verdict] = family_data.get(verdict, 0) + 1
        if candidate.objective is not None:
            best = family_data.get("best_objective")
            if best is None or float(candidate.objective) < float(best):
                family_data["best_objective"] = candidate.objective
                family_data["best_candidate_id"] = candidate.id
        if candidate.selection_score is not None:
            best_score = family_data.get("best_selection_score")
            if best_score is None or float(candidate.selection_score) < float(best_score):
                family_data["best_selection_score"] = candidate.selection_score
        recent = family_data.setdefault("recent_candidates", [])
        self._append_unique(recent, {"candidate_id": candidate.id, "objective": candidate.objective, "outcome": outcome})
        del recent[:-8]
        self._record_architecture_edge(data, candidate, entry)
        self.write(data)

    def _record_architecture_edge(self, data: dict, candidate, entry: dict) -> None:
        metadata = getattr(candidate, "metadata", None) or {}
        plan = metadata.get("generation_plan") or metadata.get("planned_experiment")
        if not isinstance(plan, dict) or not plan:
            return
        parent_id = (getattr(candidate, "parents", None) or [plan.get("parent_id") or None])[0]
        edge = {
            "parent_id": parent_id,
            "child_id": candidate.id,
            "changed_layer": plan.get("controlled_component"),
            "delta": plan.get("delta"),
            "group_id": plan.get("group_id"),
            "objective": candidate.objective,
            "selection_score": candidate.selection_score,
            "verdict": entry.get("verdict"),
            "outcome": entry.get("outcome"),
            "training_diagnosis": entry.get("training_diagnosis"),
        }
        tree = data.setdefault("architecture_tree", [])
        key = (edge["parent_id"], edge["child_id"], edge["changed_layer"], edge["delta"])
        for item in tree:
            if (item.get("parent_id"), item.get("child_id"), item.get("changed_layer"), item.get("delta")) == key:
                return
        tree.append(edge)
        del tree[:-300]

    @staticmethod
    def _target_list(candidate, outcome: str) -> str:
        verdict = getattr(candidate, "hypothesis_verdict", None)
        if verdict == "supported":
            return "supported_hypotheses"
        if verdict == "refuted" or outcome in {"eval_failed", "rejected"}:
            return "refuted_hypotheses"
        if verdict == "uncertain" or outcome in {"duplicate", "objective_duplicate"}:
            return "uncertain_hypotheses"
        if outcome == "evaluated":
            return "supported_hypotheses"
        return "active_hypotheses"

    @staticmethod
    def _append_unique(items: list, entry: dict) -> None:
        key = (entry.get("candidate_id"), entry.get("hypothesis_id"), entry.get("outcome"))
        for item in items:
            if (item.get("candidate_id"), item.get("hypothesis_id"), item.get("outcome")) == key:
                return
        items.append(entry)
