from __future__ import annotations

import concurrent.futures
import time

from eohagent.bootstrap.baseline_loader import BaselineLoader
from eohagent.bootstrap.problem_card import build_problem_card
from eohagent.core.candidate import Candidate
from eohagent.core.evaluator_bridge import build_evaluator
from eohagent.core.training_diagnostics import build_training_diagnosis


_BOOTSTRAP_EVALUATOR = None


def _evaluate_seed_worker(config: dict, seed_data: dict) -> dict:
    global _BOOTSTRAP_EVALUATOR
    seed = Candidate.from_dict(seed_data)
    start = time.time()
    if _BOOTSTRAP_EVALUATOR is None:
        _BOOTSTRAP_EVALUATOR = build_evaluator(config)
    objective = round(float(_BOOTSTRAP_EVALUATOR.evaluate(seed.code)), 5)
    return {
        "candidate_id": seed.id,
        "objective": objective,
        "evidence": getattr(_BOOTSTRAP_EVALUATOR, "last_evidence", {}) or {},
        "elapsed_sec": round(time.time() - start, 2),
    }


class ExpertBootstrap:
    def __init__(self, config: dict, evaluator, memory, progress=None):
        self.config = config
        self.evaluator = evaluator
        self.memory = memory
        self.progress = progress

    def initialize_population(self) -> list:
        seeds = BaselineLoader(self.config).load()
        results = self._evaluate_seeds(seeds)
        for seed in seeds:
            result = results[seed.id]
            seed.objective = result["objective"]
            seed.selection_score = seed.objective
            seed.contract_passed = True
            seed.hypothesis_verdict = "seed"
            evaluator_evidence = result.get("evidence", {})
            diagnosis = build_training_diagnosis(evaluator_evidence)
            seed.evidence = {
                "train_objective": seed.objective,
                "formal_evaluator": evaluator_evidence,
                "training_diagnosis": diagnosis,
            }
            seed.metadata["evaluation_evidence"] = evaluator_evidence
            seed.metadata["training_diagnosis"] = diagnosis
            self.memory.candidate_store.save(seed)
            self.memory.mechanism_archive.record_candidate(seed, "seed")
            family = (seed.mechanism_hypothesis or {}).get("mechanism_family")
            self.memory.session_log.append(
                {
                    "event": "bootstrap_loaded_seed",
                    "candidate_id": seed.id,
                    "objective": seed.objective,
                    "mechanism_family": family,
                    "hypothesis": (seed.mechanism_hypothesis or {}).get("text"),
                }
            )

        primary_seed = seeds[0]
        problem_card = build_problem_card(self.config, primary_seed)
        self.memory.save_problem_card(problem_card)
        self.memory.notebook.write_default()
        self.memory.compact_context.write_default(problem_card, primary_seed)
        return seeds

    def _evaluate_seeds(self, seeds: list[Candidate]) -> dict[str, dict]:
        if not seeds:
            return {}
        n_proc = max(
            1,
            int(
                self.config.get("bootstrap", {}).get(
                    "n_proc",
                    self.config.get("evolution", {}).get("n_proc", 1),
                )
            ),
        )
        n_jobs = min(n_proc, len(seeds))
        if n_jobs == 1:
            return self._evaluate_seeds_serial(seeds)
        return self._evaluate_seeds_parallel(seeds, n_jobs)

    def _evaluate_seeds_serial(self, seeds: list[Candidate]) -> dict[str, dict]:
        results = {}
        for index, seed in enumerate(seeds, start=1):
            self._progress("bootstrap_seed_start", seed_index=index, seed_total=len(seeds), candidate_id=seed.id)
            start = time.time()
            objective = round(float(self.evaluator.evaluate(seed.code)), 5)
            result = {
                "candidate_id": seed.id,
                "objective": objective,
                "evidence": getattr(self.evaluator, "last_evidence", {}) or {},
                "elapsed_sec": round(time.time() - start, 2),
            }
            results[seed.id] = result
            self._progress(
                "bootstrap_seed_done",
                seed_index=index,
                seed_total=len(seeds),
                candidate_id=seed.id,
                objective=objective,
                elapsed_sec=result["elapsed_sec"],
            )
        return results

    def _evaluate_seeds_parallel(self, seeds: list[Candidate], n_jobs: int) -> dict[str, dict]:
        self._progress("bootstrap_parallel_start", seed_total=len(seeds), n_jobs=n_jobs)
        worker_config = dict(self.config)
        worker_config.pop("_repo_root_path", None)
        results = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_seed = {}
            for index, seed in enumerate(seeds, start=1):
                self._progress("bootstrap_seed_start", seed_index=index, seed_total=len(seeds), candidate_id=seed.id)
                future = executor.submit(_evaluate_seed_worker, worker_config, seed.to_dict())
                future_to_seed[future] = (index, seed)
            for future in concurrent.futures.as_completed(future_to_seed):
                index, seed = future_to_seed[future]
                try:
                    result = future.result()
                except Exception as exc:
                    self._progress(
                        "bootstrap_seed_failed",
                        seed_index=index,
                        seed_total=len(seeds),
                        candidate_id=seed.id,
                        error=str(exc),
                    )
                    raise RuntimeError(f"Bootstrap seed evaluation failed for {seed.id}: {exc}") from exc
                results[seed.id] = result
                self._progress(
                    "bootstrap_seed_done",
                    seed_index=index,
                    seed_total=len(seeds),
                    candidate_id=seed.id,
                    objective=result["objective"],
                    elapsed_sec=result["elapsed_sec"],
                )
        self._progress("bootstrap_parallel_done", seed_total=len(seeds), n_jobs=n_jobs)
        return results

    def _progress(self, event: str, **fields) -> None:
        if self.progress is not None:
            self.progress(event, **fields)
            return
        message = f"[{event}] " + " ".join(f"{key}={value}" for key, value in fields.items())
        print(message, flush=True)
