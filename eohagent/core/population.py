from eohagent.core.mechanism import canonicalize_family
from eohagent.core.qd_archive import QDArchive


class PopulationManager:
    def __init__(
        self,
        pop_size: int,
        objective_direction: str = "minimize",
        deduplicate_code: bool = True,
        deduplicate_objective: bool = True,
        use_selection_score: bool = True,
        diversity_weight: float = 0.0,
        survival_selection: str = "greedy",
    ):
        self.pop_size = pop_size
        self.objective_direction = objective_direction
        self.deduplicate_code = deduplicate_code
        self.deduplicate_objective = deduplicate_objective
        self.use_selection_score = use_selection_score
        self.diversity_weight = float(diversity_weight)
        self.survival_selection = survival_selection

    def update(self, population: list, offspring: list) -> list:
        candidates = [c for c in list(population) + list(offspring) if c.objective is not None]
        candidates = self._prepare_candidates(candidates)
        ranked = self._rank(candidates)
        if self.survival_selection == "family_champion":
            return self._select_family_champions(ranked)
        if self.survival_selection == "quality_diversity":
            return self._select_quality_diversity(ranked)
        if self.survival_selection == "qd_archive":
            return self._select_qd_archive(ranked)
        if self.survival_selection == "qd_cells":
            return self._select_qd_cells(ranked)
        if self.survival_selection == "greedy" and self.diversity_weight == 0:
            return ranked[: self.pop_size]
        if self.survival_selection not in {"greedy", "diversity_penalty"}:
            raise ValueError(f"unsupported survival_selection: {self.survival_selection!r}")
        selected = []
        family_counts = {}
        while ranked and len(selected) < self.pop_size:
            best_index = 0
            best_key = self._diversity_sort_key(ranked[0], family_counts)
            for index, candidate in enumerate(ranked[1:], start=1):
                key = self._diversity_sort_key(candidate, family_counts)
                if self._key_better(key, best_key):
                    best_index = index
                    best_key = key
            chosen = ranked.pop(best_index)
            selected.append(chosen)
            family = self._family(chosen)
            family_counts[family] = family_counts.get(family, 0) + 1
        return selected

    def _select_family_champions(self, ranked: list) -> list:
        selected = []
        selected_ids = set()

        def append(candidate) -> None:
            if len(selected) >= self.pop_size or candidate.id in selected_ids:
                return
            selected.append(candidate)
            selected_ids.add(candidate.id)

        if ranked:
            append(ranked[0])

        family_best = {}
        for candidate in ranked:
            family = self._family(candidate)
            if family not in family_best:
                family_best[family] = candidate
        for candidate in self._rank(family_best.values()):
            append(candidate)
        for candidate in ranked:
            append(candidate)
        return selected

    def _select_qd_cells(self, ranked: list) -> list:
        archive = QDArchive(objective_direction=self.objective_direction, descriptor_mode="mechanism")
        archive.update(ranked)
        return archive.elites()[: self.pop_size]

    def _select_quality_diversity(self, ranked: list) -> list:
        selected = []
        selected_ids = set()

        def append(candidate) -> None:
            if len(selected) >= self.pop_size or candidate.id in selected_ids:
                return
            selected.append(candidate)
            selected_ids.add(candidate.id)

        for front in self._pareto_fronts(ranked):
            niche_best = {}
            for candidate in self._rank(front):
                key = self._niche(candidate)
                if key not in niche_best:
                    niche_best[key] = candidate
            for candidate in self._rank(niche_best.values()):
                append(candidate)
            for candidate in self._rank(front):
                append(candidate)
            if len(selected) >= self.pop_size:
                break
        return selected

    def _select_qd_archive(self, ranked: list) -> list:
        selected = []
        selected_ids = set()

        def append(candidate) -> None:
            if len(selected) >= self.pop_size or candidate.id in selected_ids:
                return
            selected.append(candidate)
            selected_ids.add(candidate.id)

        archive = QDArchive(objective_direction=self.objective_direction)
        archive.update(ranked)
        for candidate in archive.elites():
            append(candidate)
        for candidate in ranked:
            append(candidate)
        return selected

    def _prepare_candidates(self, candidates: list) -> list:
        prepared = []
        unique_by_objective = {}
        for candidate in candidates:
            key = self._objective_key(candidate.objective)
            candidate.objective = key
            if candidate.selection_score is None or not self.use_selection_score:
                candidate.selection_score = key
            if not self.deduplicate_objective:
                prepared.append(candidate)
                continue
            existing = unique_by_objective.get(key)
            if existing is None or self._is_better(candidate, existing):
                unique_by_objective[key] = candidate
        if self.deduplicate_objective:
            return list(unique_by_objective.values())
        return prepared

    def _pareto_fronts(self, candidates: list) -> list[list]:
        remaining = list(candidates)
        fronts = []
        while remaining:
            front = []
            for candidate in remaining:
                if not any(self._dominates(other, candidate) for other in remaining if other is not candidate):
                    front.append(candidate)
            fronts.append(self._rank(front))
            front_ids = {candidate.id for candidate in front}
            remaining = [candidate for candidate in remaining if candidate.id not in front_ids]
        return fronts

    def _dominates(self, left, right) -> bool:
        left_metrics = self._pareto_metrics(left)
        right_metrics = self._pareto_metrics(right)
        if self.objective_direction == "maximize":
            return all(a >= b for a, b in zip(left_metrics, right_metrics)) and any(
                a > b for a, b in zip(left_metrics, right_metrics)
            )
        return all(a <= b for a, b in zip(left_metrics, right_metrics)) and any(
            a < b for a, b in zip(left_metrics, right_metrics)
        )

    @staticmethod
    def _pareto_metrics(candidate) -> tuple[float, float, float, float]:
        evidence = getattr(candidate, "evidence", None) or {}
        formal = evidence.get("formal_evaluator") if isinstance(evidence, dict) else {}
        diagnosis = evidence.get("training_diagnosis") if isinstance(evidence, dict) else {}
        if not isinstance(formal, dict):
            formal = {}
        if not isinstance(diagnosis, dict):
            diagnosis = {}
        objective = float(candidate.objective)
        tail_gap = diagnosis.get("tail_gap", formal.get("max_gap", objective))
        max_gap = diagnosis.get("max_gap", formal.get("max_gap", tail_gap))
        std_gap = diagnosis.get("std_gap", formal.get("std_gap", 0.0))
        return (
            objective,
            float(tail_gap if tail_gap is not None else objective),
            float(max_gap if max_gap is not None else objective),
            float(std_gap if std_gap is not None else 0.0),
        )

    @staticmethod
    def _niche(candidate) -> tuple:
        metadata = getattr(candidate, "metadata", None) or {}
        arch_spec = metadata.get("arch_spec") if isinstance(metadata, dict) else None
        solver_tree = metadata.get("solver_tree") if isinstance(metadata, dict) else None
        if isinstance(solver_tree, dict):
            try:
                from eohagent.problems.tsp_gls_solver_tree.tree_schema import solver_tree_descriptor

                return solver_tree_descriptor(solver_tree)
            except Exception as exc:
                raise ValueError(f"invalid SOLVER_TREE niche metadata for candidate {candidate.id}: {exc}") from exc
        if isinstance(arch_spec, dict):
            try:
                from eohagent.problems.tsp_gls_arch_spec.spec_schema import arch_spec_niche_key

                return ("arch_spec",) + arch_spec_niche_key(arch_spec)
            except Exception as exc:
                raise ValueError(f"invalid ARCH_SPEC niche metadata for candidate {candidate.id}: {exc}") from exc
        evidence = getattr(candidate, "evidence", None) or {}
        diagnosis = evidence.get("training_diagnosis") if isinstance(evidence, dict) else {}
        failure_signature = diagnosis.get("failure_signature") if isinstance(diagnosis, dict) else None
        return ("family", PopulationManager._family(candidate), failure_signature)

    def _rank(self, candidates) -> list:
        return sorted(candidates, key=self._base_sort_key, reverse=self.objective_direction == "maximize")

    def _is_better(self, candidate, existing) -> bool:
        candidate_key = self._base_sort_key(candidate)
        existing_key = self._base_sort_key(existing)
        if self.objective_direction == "maximize":
            return candidate_key > existing_key
        return candidate_key < existing_key

    def _base_sort_key(self, candidate) -> float:
        if self.use_selection_score and candidate.selection_score is not None:
            return float(candidate.selection_score)
        return float(candidate.objective)

    def _diversity_sort_key(self, candidate, family_counts: dict[str, int]) -> float:
        base = self._base_sort_key(candidate)
        penalty = self.diversity_weight * family_counts.get(self._family(candidate), 0)
        return base - penalty if self.objective_direction == "maximize" else base + penalty

    def _key_better(self, left: float, right: float) -> bool:
        if self.objective_direction == "maximize":
            return left > right
        return left < right

    @staticmethod
    def _family(candidate) -> str:
        hypothesis = getattr(candidate, "mechanism_hypothesis", None) or {}
        if isinstance(hypothesis, dict) and hypothesis.get("mechanism_family"):
            return canonicalize_family(str(hypothesis["mechanism_family"]), hypothesis.get("text"))
        profile = getattr(candidate, "mechanism_profile", None) or {}
        if isinstance(profile, dict) and profile.get("mechanism_family"):
            return canonicalize_family(str(profile["mechanism_family"]), profile.get("hypothesis_text"))
        return "unspecified"

    @staticmethod
    def _objective_key(objective) -> float:
        return round(float(objective), 5)
