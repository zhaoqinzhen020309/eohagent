from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


METHOD_CHOICES = ("eoh", "reevo", "mcts_ahd", "lhns", "hillclimb", "randsample", "funsearch")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    llm4ad_root = repo_root / "LLM4AD"
    add_path(repo_root)
    add_path(llm4ad_root)

    output_dir = resolve_path(repo_root, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = build_task(args)
    llm = build_llm(args)
    profiler = build_profiler(args.method, output_dir)
    method = build_method(args, llm, task, profiler)

    if args.seed_file:
        inject_seeds(method, resolve_path(repo_root, args.seed_file), args.seed_limit)

    write_run_config(output_dir, args)
    method.run()

    population_path = export_population(output_dir, args.method, args.export_top_k)
    print(f"[baseline] exported population={population_path}", flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one fair LLM4AD TSP-GLS baseline experiment.")
    parser.add_argument("--method", required=True, choices=METHOD_CHOICES)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=os.environ.get("LLM4AD_MODEL", "deepseek-v4-flash"))
    parser.add_argument("--api-host", default=os.environ.get("LLM4AD_API_HOST", "api.deepseek.com"))
    parser.add_argument("--api-key", default=os.environ.get("AGENT_EOHS_API_KEY") or os.environ.get("LLM4AD_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--llm-timeout", type=int, default=600)
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--max-generations", type=int, default=20)
    parser.add_argument("--pop-size", type=int, default=10)
    parser.add_argument("--selection-num", type=int, default=2, help="Number of parent heuristics selected by EoH crossover prompts.")
    parser.add_argument("--num-samplers", type=int, default=5)
    parser.add_argument("--num-evaluators", type=int, default=5)
    parser.add_argument("--eval-timeout", type=int, default=500)
    parser.add_argument("--n-inst-eva", type=int, default=64)
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--ite-max", type=int, default=1000)
    parser.add_argument("--perturbation-moves", type=int, default=1)
    parser.add_argument("--example-dir", default="examples/user_tsp_gls_OPRO")
    parser.add_argument("--seed-file", default=None)
    parser.add_argument("--seed-limit", type=int, default=None)
    parser.add_argument("--export-top-k", type=int, default=10)
    parser.add_argument("--debug-mode", action="store_true")
    parser.add_argument("--eval-workers", choices=("thread", "process"), default="thread")
    return parser.parse_args()


def build_task(args: argparse.Namespace):
    from llm4ad.task.optimization.tsp_gls_opro_gap import TSPGLSOPROGapEvaluation

    return TSPGLSOPROGapEvaluation(
        example_dir=args.example_dir,
        n_inst_eva=args.n_inst_eva,
        time_limit=args.time_limit,
        ite_max=args.ite_max,
        perturbation_moves=args.perturbation_moves,
        timeout_seconds=args.eval_timeout,
    )


def build_llm(args: argparse.Namespace):
    if not args.api_key:
        raise RuntimeError("Missing API key. Set AGENT_EOHS_API_KEY, LLM4AD_API_KEY, or pass --api-key.")
    from llm4ad.tools.llm.llm_api_https import HttpsApi

    return HttpsApi(
        host=normalize_host(args.api_host),
        key=args.api_key,
        model=args.model,
        timeout=args.llm_timeout,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )


def build_profiler(method: str, output_dir: Path):
    log_dir = str(output_dir / "llm4ad_logs" / method)
    if method == "eoh":
        from llm4ad.method.eoh import EoHProfiler

        return EoHProfiler(log_dir=log_dir, log_style="simple", create_random_path=False)
    if method == "reevo":
        from llm4ad.method.reevo import ReEvoProfiler

        return ReEvoProfiler(log_dir=log_dir, log_style="simple", create_random_path=False)
    if method == "mcts_ahd":
        from llm4ad.method.mcts_ahd import MAProfiler

        return MAProfiler(log_dir=log_dir, log_style="simple", create_random_path=False)
    if method == "lhns":
        from llm4ad.method.lhns import LHNSProfiler

        return LHNSProfiler(log_dir=log_dir, log_style="simple", create_random_path=False)
    if method == "hillclimb":
        from llm4ad.method.hillclimb import HillClimbProfiler

        return HillClimbProfiler(log_dir=log_dir, log_style="simple", create_random_path=False)
    if method == "randsample":
        from llm4ad.method.randsample import RandSampleProfiler

        return RandSampleProfiler(log_dir=log_dir, log_style="simple", create_random_path=False)
    if method == "funsearch":
        from llm4ad.method.funsearch.profiler import FunSearchProfiler

        return FunSearchProfiler(log_dir=log_dir, log_style="simple", create_random_path=False)
    raise ValueError(f"Unsupported method: {method}")


def build_method(args: argparse.Namespace, llm, task, profiler):
    common = {
        "llm": llm,
        "evaluation": task,
        "profiler": profiler,
        "max_sample_nums": args.max_samples,
        "num_samplers": args.num_samplers,
        "num_evaluators": args.num_evaluators,
        "resume_mode": bool(args.seed_file),
        "debug_mode": args.debug_mode,
        "multi_thread_or_process_eval": args.eval_workers,
    }
    if args.method == "eoh":
        from llm4ad.method.eoh import EoH

        return EoH(
            max_generations=args.max_generations,
            pop_size=args.pop_size,
            selection_num=args.selection_num,
            **common,
        )
    if args.method == "reevo":
        from llm4ad.method.reevo import ReEvo

        return ReEvo(pop_size=args.pop_size, **common)
    if args.method == "mcts_ahd":
        if args.seed_file:
            raise RuntimeError("mcts_ahd ignores resume_mode in LLM4AD; run it without --seed-file for a clean baseline.")
        from llm4ad.method.mcts_ahd import MCTS_AHD

        return MCTS_AHD(init_size=min(4, args.pop_size), pop_size=args.pop_size, **common)
    if args.method == "lhns":
        from llm4ad.method.lhns import LHNS

        common.pop("resume_mode")
        return LHNS(elite_set_size=args.pop_size, resume_mode=bool(args.seed_file), **common)
    if args.method == "hillclimb":
        if args.seed_file:
            raise RuntimeError("hillclimb has no population resume hook; run it without --seed-file for a clean baseline.")
        from llm4ad.method.hillclimb import HillClimb

        return HillClimb(**common)
    if args.method == "randsample":
        if args.seed_file:
            raise RuntimeError("randsample has no parent population; run it without --seed-file for a clean baseline.")
        from llm4ad.method.randsample import RandSample

        return RandSample(**common)
    if args.method == "funsearch":
        if args.seed_file:
            raise RuntimeError("funsearch seed injection is not implemented for the LLM4AD program database.")
        from llm4ad.method.funsearch import FunSearch

        return FunSearch(**common)
    raise ValueError(f"Unsupported method: {args.method}")


def inject_seeds(method, seed_file: Path, seed_limit: int | None) -> None:
    seeds = json.loads(seed_file.read_text(encoding="utf-8"))
    if seed_limit is not None:
        seeds = seeds[:seed_limit]
    if not seeds:
        raise RuntimeError(f"No seeds loaded from {seed_file}")

    for index, seed in enumerate(seeds, start=1):
        code = seed["code"]
        score, eval_time = method._evaluator.evaluate_program_record_time(program=code)
        if score is None:
            raise RuntimeError(f"Seed {seed.get('id', index)} evaluated to None")
        function = seed_to_function(method, code)
        function.score = score
        function.evaluate_time = eval_time
        function.algorithm = seed.get("algorithm", seed.get("id", f"seed_{index}"))
        if method._profiler is not None:
            method._profiler.register_function(function, program=code)

        if hasattr(method, "_population"):
            method._population._population.append(function)
        elif hasattr(method, "_elite_set"):
            method._elite_set.register_function(function)
            if getattr(method._best_function, "score", None) is None or function.score > method._best_function.score:
                method._current_function = function
                method._best_function = function
        else:
            raise RuntimeError(f"Seed injection is not supported for {method.__class__.__name__}")


def seed_to_function(method, code: str):
    if hasattr(method, "_elite_set"):
        from llm4ad.method.lhns.func_ruin import LHNSTextFunctionProgramConverter

        function = LHNSTextFunctionProgramConverter.text_to_function(code)
        if function is not None:
            function.features = ""
    else:
        from llm4ad.base import TextFunctionProgramConverter

        function = TextFunctionProgramConverter.text_to_function(code)
    if function is None:
        raise RuntimeError("Seed code is not a single valid function")
    return function


def export_population(output_dir: Path, method: str, top_k: int) -> Path:
    records = load_profiler_records(output_dir / "llm4ad_logs" / method / "samples")
    valid_records = [record for record in records if is_finite_score(record.get("score"))]
    if not valid_records:
        raise RuntimeError(f"No finite-score records found under {output_dir}")
    valid_records.sort(key=lambda record: float(record["score"]), reverse=True)
    selected = valid_records[:top_k]

    population = []
    for index, record in enumerate(selected, start=1):
        score = float(record["score"])
        code = record.get("program") or with_numpy_import(record["function"])
        population.append(
            {
                "id": f"{method}_{index:04d}",
                "algorithm": f"LLM4AD {method} baseline; score={score}",
                "objective": round(-score, 8),
                "score": score,
                "status": "evaluated",
                "code": code,
            }
        )

    pop_dir = output_dir / "results" / "pops"
    pop_dir.mkdir(parents=True, exist_ok=True)
    population_path = pop_dir / "population_final.json"
    population_path.write_text(json.dumps(population, indent=2), encoding="utf-8")
    return population_path


def load_profiler_records(samples_dir: Path) -> list[dict[str, Any]]:
    if not samples_dir.exists():
        raise RuntimeError(f"Profiler samples directory does not exist: {samples_dir}")
    records: list[dict[str, Any]] = []
    for path in sorted(samples_dir.glob("samples_*~*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    return records


def is_finite_score(score: Any) -> bool:
    try:
        return math.isfinite(float(score))
    except (TypeError, ValueError):
        return False


def with_numpy_import(function_code: str) -> str:
    return "import numpy as np\n\n" + function_code.strip() + "\n"


def write_run_config(output_dir: Path, args: argparse.Namespace) -> None:
    config = {key: value for key, value in vars(args).items() if key != "api_key"}
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")


def normalize_host(value: str) -> str:
    return value.removeprefix("https://").removeprefix("http://").rstrip("/")


def add_path(path: Path) -> None:
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)


def resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
