from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
from llm4ad.base import Evaluation

from .template import task_description, template_program


__all__ = ["TSPGLSOPROGapEvaluation"]


def _find_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "examples" / "user_tsp_gls_OPRO").exists():
            return parent
    raise FileNotFoundError("Could not find repository root containing examples/user_tsp_gls_OPRO")


class TSPGLSOPROGapEvaluation(Evaluation):
    """LLM4AD evaluator aligned with the eohagent TSP-GLS training setup."""

    def __init__(
        self,
        *,
        example_dir: str | Path | None = None,
        n_inst_eva: int = 64,
        time_limit: int = 60,
        ite_max: int = 1000,
        perturbation_moves: int = 1,
        timeout_seconds: int = 500,
    ) -> None:
        super().__init__(
            template_program=template_program,
            task_description=task_description,
            use_numba_accelerate=False,
            timeout_seconds=timeout_seconds,
        )
        self.repo_root = _find_repo_root()
        self.example_dir = Path(example_dir) if example_dir is not None else self.repo_root / "examples" / "user_tsp_gls_OPRO"
        if not self.example_dir.is_absolute():
            self.example_dir = self.repo_root / self.example_dir
        self.n_inst_eva = int(n_inst_eva)
        self.time_limit = int(time_limit)
        self.ite_max = int(ite_max)
        self.perturbation_moves = int(perturbation_moves)
        self.instance_path = self.example_dir / "TrainingData" / "TSPAEL64.pkl"
        self._ensure_import_paths()

        from utils import readTSPRandom

        coords, instances, opt_costs = readTSPRandom.read_instance_all(str(self.instance_path))
        if self.n_inst_eva > len(instances):
            raise ValueError(f"n_inst_eva={self.n_inst_eva} exceeds available training instances={len(instances)}")
        self.coords = coords
        self.instances = instances
        self.opt_costs = opt_costs

    def evaluate_program(self, program_str: str, callable_func: callable, **kwargs) -> float:
        return self._evaluate_callable(callable_func)

    def evaluate(self, callable_func: callable) -> float:
        """Compatibility shim for LLM4AD methods that expect evaluator.evaluate(callable)."""
        return self._evaluate_callable(callable_func)

    def _evaluate_callable(self, callable_func: callable) -> float:
        if callable_func is None:
            raise RuntimeError("TSPGLSOPROGapEvaluation requires a callable update_edge_distance function")
        self._ensure_import_paths()

        from gls.gls_run import solve_instance

        heuristic = types.SimpleNamespace(update_edge_distance=callable_func)
        gaps = np.zeros(self.n_inst_eva, dtype=float)
        for index in range(self.n_inst_eva):
            gaps[index] = float(
                solve_instance(
                    index,
                    self.opt_costs[index],
                    self.instances[index],
                    self.coords[index],
                    self.time_limit,
                    self.ite_max,
                    self.perturbation_moves,
                    heuristic,
                )
            )
        mean_gap = float(np.mean(gaps))
        return -mean_gap

    def _ensure_import_paths(self) -> None:
        for path in (self.repo_root, self.example_dir):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
