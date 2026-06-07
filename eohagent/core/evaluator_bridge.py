from __future__ import annotations

import re
import subprocess
import sys
import types
import warnings
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

from .config import resolve_path


def normalize_objective(value: Any) -> float:
    if value is None:
        raise RuntimeError("Evaluator returned None")
    if isinstance(value, dict):
        nums = [float(v) for v in value.values()]
        if not nums:
            raise RuntimeError("Evaluator returned an empty objective dict")
        return float(mean(nums))
    if isinstance(value, (list, tuple)):
        nums = [float(v) for v in value]
        if not nums:
            raise RuntimeError("Evaluator returned an empty objective list")
        return float(mean(nums))
    return float(value)


class ImportEoHEvaluatorBridge:
    def __init__(self, config: dict):
        self.config = config
        self.problem = config["evaluator"].get("problem") or config["project"]["problem"]
        self.last_evidence: dict[str, Any] = {}
        eoh_src = Path(config["_repo_root"]) / "eoh" / "src"
        if str(eoh_src) not in sys.path:
            sys.path.insert(0, str(eoh_src))
        self._evaluator = self._load_problem()

    def _load_problem(self):
        with redirect_stdout(StringIO()):
            if self.problem == "bp_online":
                from eoh.problems.optimization.bp_online.run import BPONLINE

                return BPONLINE()
            if self.problem == "tsp_construct":
                from eoh.problems.optimization.tsp_greedy.run import TSPCONST

                return TSPCONST()
            if self.problem == "tsp_gls":
                gls_dir = Path(self.config["_repo_root"]) / "examples" / "user_tsp_gls_OPRO"
                if str(gls_dir) not in sys.path:
                    sys.path.insert(0, str(gls_dir))
                from prob import TSPGLS

                evaluator = TSPGLS()
                evaluator_config = self.config.get("evaluator", {})
                if "n_inst_eva" in evaluator_config:
                    evaluator.n_inst_eva = int(evaluator_config["n_inst_eva"])
                if "time_limit" in evaluator_config:
                    evaluator.time_limit = int(evaluator_config["time_limit"])
                if "ite_max" in evaluator_config:
                    evaluator.ite_max = int(evaluator_config["ite_max"])
                if "perturbation_moves" in evaluator_config:
                    evaluator.perturbation_moves = int(evaluator_config["perturbation_moves"])
                return evaluator
            if self.problem in {"tsp_gls_arch", "tsp_gls_arch_spec", "tsp_gls_solver_tree"}:
                gls_dir = Path(self.config["_repo_root"]) / "examples" / "user_tsp_gls_OPRO"
                if str(gls_dir) not in sys.path:
                    sys.path.insert(0, str(gls_dir))
                return self._load_tsp_gls_arch_problem(gls_dir)
        raise ValueError(f"Unsupported import evaluator problem: {self.problem}")

    def _load_tsp_gls_arch_problem(self, gls_dir: Path):
        from utils import readTSPRandom

        evaluator_config = self.config.get("evaluator", {})
        data_path = gls_dir / "TrainingData" / "TSPAEL64.pkl"
        coords, instances, opt_costs = readTSPRandom.read_instance_all(str(data_path))
        return types.SimpleNamespace(
            n_inst_eva=int(evaluator_config.get("n_inst_eva", 64)),
            time_limit=int(evaluator_config.get("time_limit", 60)),
            ite_max=int(evaluator_config.get("ite_max", 1000)),
            perturbation_moves=int(evaluator_config.get("perturbation_moves", 1)),
            instance_path=str(data_path),
            coords=coords,
            instances=instances,
            opt_costs=opt_costs,
        )

    def evaluate(self, code: str) -> float:
        with redirect_stdout(StringIO()):
            if self.problem == "tsp_gls":
                raw = self._evaluate_tsp_gls_with_evidence(code)
            elif self.problem == "tsp_gls_arch":
                raw = self._evaluate_tsp_gls_arch_with_evidence(code)
            elif self.problem == "tsp_gls_arch_spec":
                raw = self._evaluate_tsp_gls_arch_spec_with_evidence(code)
            elif self.problem == "tsp_gls_solver_tree":
                raw = self._evaluate_tsp_gls_solver_tree_with_evidence(code)
            else:
                self.last_evidence = {}
                raw = self._evaluator.evaluate(code)
        return normalize_objective(raw)

    def _evaluate_tsp_gls_with_evidence(self, code: str):
        from gls.gls_run import solve_instance

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            heuristic_module = types.ModuleType("heuristic_module")
            try:
                exec(code, heuristic_module.__dict__)
            except Exception as exc:
                self.last_evidence = {"error": f"{type(exc).__name__}: {exc}"}
                raise RuntimeError(f"tsp_gls evaluator failed while loading candidate: {exc}") from exc
            sys.modules[heuristic_module.__name__] = heuristic_module
            evaluator = self._evaluator
            gaps = []
            for index in range(evaluator.n_inst_eva):
                try:
                    gap = solve_instance(
                        index,
                        evaluator.opt_costs[index],
                        evaluator.instances[index],
                        evaluator.coords[index],
                        evaluator.time_limit,
                        evaluator.ite_max,
                        evaluator.perturbation_moves,
                        heuristic_module,
                    )
                except Exception as exc:
                    self.last_evidence = {"error": f"instance {index}: {type(exc).__name__}: {exc}"}
                    raise RuntimeError(f"tsp_gls evaluator failed on training instance {index}: {exc}") from exc
                gaps.append(float(gap))
            self.last_evidence = self._gap_evidence(gaps)
            return self.last_evidence["mean_gap"]

    def _evaluate_tsp_gls_arch_spec_with_evidence(self, code: str):
        from eohagent.problems.tsp_gls_arch_spec.gls_arch_spec_run import solve_instance

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            heuristic_module = types.ModuleType("heuristic_module")
            try:
                exec(code, heuristic_module.__dict__)
            except Exception as exc:
                self.last_evidence = {"error": f"{type(exc).__name__}: {exc}"}
                raise RuntimeError(f"tsp_gls_arch_spec evaluator failed while loading candidate: {exc}") from exc
            sys.modules[heuristic_module.__name__] = heuristic_module
            evaluator = self._evaluator
            gaps = []
            for index in range(evaluator.n_inst_eva):
                try:
                    gap = solve_instance(
                        index,
                        evaluator.opt_costs[index],
                        evaluator.instances[index],
                        evaluator.coords[index],
                        evaluator.time_limit,
                        evaluator.ite_max,
                        evaluator.perturbation_moves,
                        heuristic_module,
                    )
                except Exception as exc:
                    self.last_evidence = {"error": f"instance {index}: {type(exc).__name__}: {exc}"}
                    raise RuntimeError(f"tsp_gls_arch_spec evaluator failed on training instance {index}: {exc}") from exc
                gaps.append(float(gap))
            self.last_evidence = self._gap_evidence(gaps)
            return self.last_evidence["mean_gap"]

    def _evaluate_tsp_gls_solver_tree_with_evidence(self, code: str):
        from eohagent.problems.tsp_gls_solver_tree.gls_tree_run import solve_instance

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            heuristic_module = types.ModuleType("heuristic_module")
            try:
                exec(code, heuristic_module.__dict__)
            except Exception as exc:
                self.last_evidence = {"error": f"{type(exc).__name__}: {exc}"}
                raise RuntimeError(f"tsp_gls_solver_tree evaluator failed while loading candidate: {exc}") from exc
            sys.modules[heuristic_module.__name__] = heuristic_module
            evaluator = self._evaluator
            gaps = []
            for index in range(evaluator.n_inst_eva):
                try:
                    gap = solve_instance(
                        index,
                        evaluator.opt_costs[index],
                        evaluator.instances[index],
                        evaluator.coords[index],
                        evaluator.time_limit,
                        evaluator.ite_max,
                        evaluator.perturbation_moves,
                        heuristic_module,
                    )
                except Exception as exc:
                    self.last_evidence = {"error": f"instance {index}: {type(exc).__name__}: {exc}"}
                    raise RuntimeError(f"tsp_gls_solver_tree evaluator failed on training instance {index}: {exc}") from exc
                gaps.append(float(gap))
            self.last_evidence = self._gap_evidence(gaps)
            return self.last_evidence["mean_gap"]

    def _evaluate_tsp_gls_arch_with_evidence(self, code: str):
        from eohagent.problems.tsp_gls_arch.gls_arch_run import solve_instance

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            heuristic_module = types.ModuleType("heuristic_module")
            try:
                exec(code, heuristic_module.__dict__)
            except Exception as exc:
                self.last_evidence = {"error": f"{type(exc).__name__}: {exc}"}
                raise RuntimeError(f"tsp_gls_arch evaluator failed while loading candidate: {exc}") from exc
            sys.modules[heuristic_module.__name__] = heuristic_module
            evaluator = self._evaluator
            gaps = []
            for index in range(evaluator.n_inst_eva):
                try:
                    gap = solve_instance(
                        index,
                        evaluator.opt_costs[index],
                        evaluator.instances[index],
                        evaluator.coords[index],
                        evaluator.time_limit,
                        evaluator.ite_max,
                        evaluator.perturbation_moves,
                        heuristic_module,
                    )
                except Exception as exc:
                    self.last_evidence = {"error": f"instance {index}: {type(exc).__name__}: {exc}"}
                    raise RuntimeError(f"tsp_gls_arch evaluator failed on training instance {index}: {exc}") from exc
                gaps.append(float(gap))
            self.last_evidence = self._gap_evidence(gaps)
            return self.last_evidence["mean_gap"]

    @staticmethod
    def _gap_evidence(gaps: list[float]) -> dict[str, Any]:
        arr = np.asarray(gaps, dtype=float)
        if arr.size == 0:
            return {"mean_gap": None, "per_instance_gap": []}
        order = np.argsort(arr)
        worst = order[-5:][::-1].tolist()
        best = order[:5].tolist()
        return {
            "mean_gap": round(float(np.mean(arr)), 8),
            "std_gap": round(float(np.std(arr)), 8),
            "max_gap": round(float(np.max(arr)), 8),
            "min_gap": round(float(np.min(arr)), 8),
            "worst_instances": [{"index": int(i), "gap": round(float(arr[i]), 8)} for i in worst],
            "best_instances": [{"index": int(i), "gap": round(float(arr[i]), 8)} for i in best],
            "per_instance_gap": [round(float(value), 8) for value in arr.tolist()],
        }


class SubprocessEvaluatorBridge:
    def __init__(self, config: dict):
        evaluator_config = config["evaluator"]
        self.eval_dir = resolve_path(config, evaluator_config["eval_dir"])
        self.heuristic_path = self.eval_dir / evaluator_config.get("heuristic_filename", "heuristic.py")
        self.run_command = evaluator_config.get("run_command", "python runEval.py").split()
        self.timeout = int(evaluator_config.get("timeout", 300))
        self.problem = config["project"].get("problem", "")
        self.last_evidence: dict[str, Any] = {}

    def evaluate(self, code: str) -> float:
        self.heuristic_path.write_text(code, encoding="utf-8")
        result = subprocess.run(
            self.run_command,
            cwd=str(self.eval_dir),
            capture_output=True,
            text=True,
            timeout=self.timeout,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        return self.parse_objective(result.stdout)

    def parse_objective(self, stdout: str) -> float:
        if self.problem == "bp_online":
            values = [float(x) for x in re.findall(r"Excess:\s*([-+]?\d+(?:\.\d+)?)%", stdout)]
        elif self.problem == "tsp_construct":
            values = [float(x) for x in re.findall(r"is:\s*([-+]?\d+(?:\.\d+)?)", stdout)]
        elif self.problem in {"tsp_gls", "tsp_gls_arch", "tsp_gls_arch_spec"}:
            values = [float(x) for x in re.findall(r"Gap:\s*([-+]?\d+(?:\.\d+)?)%", stdout)]
        else:
            values = [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", stdout)]
        if not values:
            raise RuntimeError(f"Could not parse objective from evaluator output:\n{stdout}")
        return float(mean(values))


def build_evaluator(config: dict):
    mode = config["evaluator"].get("mode", "import")
    if mode == "import":
        return ImportEoHEvaluatorBridge(config)
    if mode == "subprocess":
        return SubprocessEvaluatorBridge(config)
    raise ValueError(f"Unsupported evaluator mode: {mode}")
