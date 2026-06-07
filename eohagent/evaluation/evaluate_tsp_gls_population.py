from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import types
from pathlib import Path

import numpy as np


OPTIMAL_SOLUTIONS = {
    "eil51.tsp": 426,
    "berlin52.tsp": 7542,
    "st70.tsp": 675,
    "eil76.tsp": 538,
    "pr76.tsp": 108159,
    "rat99.tsp": 1211,
    "rd100.tsp": 7910.396,
    "kroA100.tsp": 21282,
    "kroB100.tsp": 22141,
    "kroC100.tsp": 20749,
    "kroD100.tsp": 21294,
    "kroE100.tsp": 22068,
    "eil101.tsp": 629,
    "lin105.tsp": 14379,
    "pr107.tsp": 44303,
    "pr124.tsp": 59030.0,
    "bier127.tsp": 118282.0,
    "ch130.tsp": 6110,
    "pr136.tsp": 96772,
    "pr144.tsp": 58537,
    "ch150.tsp": 6528,
    "kroA150.tsp": 26524.0,
    "kroB150.tsp": 26130,
    "pr152.tsp": 73682,
    "u159.tsp": 42080.0,
    "rat195.tsp": 2323,
    "d198.tsp": 15780,
    "kroA200.tsp": 29368,
    "kroB200.tsp": 29437.0,
}


DEFAULT_INSTANCES = list(OPTIMAL_SOLUTIONS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate eohagent TSP-GLS population on TSPLib.")
    parser.add_argument("population_json", help="Path to population_generation_*.json")
    parser.add_argument("--example-dir", default="./examples/user_tsp_gls_OPRO")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N individuals.")
    parser.add_argument("--instances", nargs="*", default=None, help="Subset of TSPLib instance filenames.")
    parser.add_argument("--time-limit", type=int, default=60)
    parser.add_argument("--ite-max", type=int, default=1000)
    parser.add_argument("--perturbation-moves", type=int, default=1)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    add_repo_path(repo_root)
    population_path = resolve_path(repo_root, args.population_json)
    example_dir = resolve_path(repo_root, args.example_dir)
    output_json = Path(args.output_json) if args.output_json else population_path.with_name(population_path.stem + "_tsplib_eval.json")
    output_csv = Path(args.output_csv) if args.output_csv else population_path.with_name(population_path.stem + "_tsplib_eval.csv")

    if not output_json.is_absolute():
        output_json = repo_root / output_json
    if not output_csv.is_absolute():
        output_csv = repo_root / output_csv
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    add_example_path(example_dir)
    from eohagent.problems.tsp_gls_arch import gls_arch_run
    from eohagent.problems.tsp_gls_arch_spec import gls_arch_spec_run
    from eohagent.problems.tsp_gls_solver_tree import gls_tree_run
    from utils import readTSPLib

    population = json.loads(population_path.read_text(encoding="utf-8"))
    if args.limit is not None:
        population = population[: args.limit]
    instances = args.instances or DEFAULT_INSTANCES

    results = []
    print(f"[eval] population={population_path} individuals={len(population)} instances={len(instances)}", flush=True)

    for index, candidate in enumerate(population, start=1):
        candidate_id = candidate.get("id", f"candidate_{index}")
        training_objective = candidate.get("objective")
        print(f"[eval] candidate {index}/{len(population)} id={candidate_id} train_obj={training_objective}", flush=True)
        started = time.time()
        instance_results = evaluate_candidate(
            candidate,
            example_dir,
            instances,
            readTSPLib.read_instance,
            gls_arch_run.solve_instance,
            gls_arch_spec_run.solve_instance,
            gls_tree_run.solve_instance,
            time_limit=args.time_limit,
            ite_max=args.ite_max,
            perturbation_moves=args.perturbation_moves,
        )
        valid = [value for value in instance_results.values() if value is not None]
        average_gap = float(np.mean(valid)) if valid else None
        row = {
            "id": candidate_id,
            "algorithm": candidate.get("algorithm"),
            "training_objective": training_objective,
            "tsplib_average_gap": average_gap,
            "elapsed_sec": round(time.time() - started, 2),
            "instance_gaps": instance_results,
        }
        results.append(row)
        print(f"[eval] candidate {candidate_id} average_gap={average_gap} elapsed={row['elapsed_sec']}s", flush=True)

    output_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_csv(output_csv, results)
    print(f"[eval] wrote {output_json}", flush=True)
    print(f"[eval] wrote {output_csv}", flush=True)
    return 0


def evaluate_candidate(
    candidate: dict,
    example_dir: Path,
    instances: list[str],
    read_instance,
    arch_solve_instance,
    arch_spec_solve_instance,
    solver_tree_solve_instance,
    time_limit: int,
    ite_max: int,
    perturbation_moves: int,
) -> dict[str, float | None]:
    module = types.ModuleType(f"tsp_gls_candidate_{candidate.get('id', 'unknown')}")
    exec(candidate["code"], module.__dict__)
    if hasattr(module, "SOLVER_TREE") and hasattr(module, "CandidateGLSArchitecture"):
        active_solver = solver_tree_solve_instance
    elif hasattr(module, "ARCH_SPEC") and hasattr(module, "CandidateGLSArchitecture"):
        active_solver = arch_spec_solve_instance
    elif hasattr(module, "CandidateGLSArchitecture"):
        active_solver = arch_solve_instance
    elif hasattr(module, "update_edge_distance"):
        from gls import gls_run

        active_solver = gls_run.solve_instance
    else:
        raise RuntimeError(f"{candidate.get('id')} defines neither update_edge_distance nor CandidateGLSArchitecture")

    results = {}
    for instance in instances:
        print(f"  [instance] {instance}", flush=True)
        try:
            graph, scale_factor = read_instance(str(example_dir / "TestingData" / "TSPLib" / instance))
            distance_matrix, coords = networkx_to_matrix_coords(graph, scale_factor)
            gap = active_solver(
                0,
                OPTIMAL_SOLUTIONS[instance],
                distance_matrix,
                coords,
                time_limit,
                ite_max,
                perturbation_moves,
                module,
            )
            results[instance] = round(float(gap), 5)
        except Exception as exc:
            print(f"  [instance] {instance} failed: {exc}", flush=True)
            results[instance] = None
    return results


def networkx_to_matrix_coords(graph, scale_factor):
    nodes = sorted(graph.nodes())
    n = len(nodes)
    distance_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = graph[nodes[i]][nodes[j]]["weight"] * scale_factor
    np.random.seed(2024)
    coords = np.random.rand(n, 2) * 100
    return distance_matrix, coords


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["id", "training_objective", "tsplib_average_gap", "elapsed_sec", "instance_gaps"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "training_objective": row["training_objective"],
                    "tsplib_average_gap": row["tsplib_average_gap"],
                    "elapsed_sec": row["elapsed_sec"],
                    "instance_gaps": json.dumps(row["instance_gaps"], sort_keys=True),
                }
            )


def add_example_path(example_dir: Path) -> None:
    if str(example_dir) not in sys.path:
        sys.path.insert(0, str(example_dir))


def add_repo_path(repo_root: Path) -> None:
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


def resolve_path(repo_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return repo_root / path


if __name__ == "__main__":
    raise SystemExit(main())
