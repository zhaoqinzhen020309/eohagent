# -------
# Evaluate every heuristic in an EoHAgent TSP population file.
# This keeps the original single-heuristic runEval.py untouched.
# -------

import argparse
import csv
import importlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

from evaluation import Evaluation


EVAL_DIR = Path(__file__).resolve().parent
REPO_ROOT = EVAL_DIR.parents[2]
HEURISTIC_FILE = EVAL_DIR / "heuristic.py"


def repo_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def load_population(path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Population file must contain a list: {path}")
    return data


def write_heuristic(code):
    if not isinstance(code, str) or not code.strip():
        raise TypeError("candidate code must be a non-empty string")
    HEURISTIC_FILE.write_text(code.rstrip() + "\n", encoding="utf-8")
    importlib.invalidate_caches()
    sys.modules.pop("heuristic", None)


def evaluate_one_size(size, n_test, debug_mode):
    instance_file = EVAL_DIR / "testingdata" / f"instance_data_{size}.pkl"
    with instance_file.open("rb") as f:
        instance_dataset = pickle.load(f)
    evaluator = Evaluation(size, instance_dataset, n_test, debug_mode)
    start = time.time()
    objective = evaluator.evaluate()
    return objective, time.time() - start


def summarize(rows):
    ok_rows = [r for r in rows if r["error"] == ""]
    if not ok_rows:
        print("No candidate was evaluated successfully.")
        return

    by_size = {}
    for row in ok_rows:
        by_size.setdefault(row["size"], []).append(row)

    print("\nBest result by test size:")
    for size in sorted(by_size):
        best = min(by_size[size], key=lambda r: r["test_objective"])
        print(
            f"  size={size}: rank={best['rank']} id={best['candidate_id']} "
            f"objective={best['test_objective']:.6f}"
        )

    by_candidate = {}
    for row in ok_rows:
        by_candidate.setdefault(row["rank"], []).append(row)

    complete = [
        (rank, rows_for_candidate)
        for rank, rows_for_candidate in by_candidate.items()
        if len(rows_for_candidate) == len(by_size)
    ]
    if complete:
        best_rank, best_rows = min(
            complete,
            key=lambda item: sum(r["test_objective"] for r in item[1]) / len(item[1]),
        )
        avg = sum(r["test_objective"] for r in best_rows) / len(best_rows)
        candidate_id = best_rows[0]["candidate_id"]
        print(
            f"Best average across sizes: rank={best_rank} "
            f"id={candidate_id} objective={avg:.6f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all candidates in a saved TSP population JSON file."
    )
    parser.add_argument(
        "--population",
        default="eohagent/results/tsp_qwen_20x20/results/pops/population_generation_20.json",
        help="Population JSON path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--output-csv",
        default="eohagent/results/tsp_qwen_20x20/population_eval_generation_20.csv",
        help="CSV output path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--output-json",
        default="eohagent/results/tsp_qwen_20x20/population_eval_generation_20.json",
        help="JSON output path. Relative paths are resolved from the repository root.",
    )
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[20, 50, 100],
        help="TSP test sizes to evaluate.",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=64,
        help="Number of test instances per size.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable evaluator debug mode.",
    )
    parser.add_argument(
        "--keep-last",
        action="store_true",
        help="Keep the final candidate code in heuristic.py instead of restoring the original file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    population_path = repo_path(args.population)
    output_csv = repo_path(args.output_csv)
    output_json = repo_path(args.output_json)

    population = load_population(population_path)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    original_heuristic = HEURISTIC_FILE.read_bytes()
    rows = []

    os.chdir(EVAL_DIR)
    print(f"Evaluating {len(population)} candidates from {population_path}")
    print(f"Test sizes: {args.sizes}; n_test={args.n_test}")

    try:
        for rank, candidate in enumerate(population, start=1):
            candidate_id = candidate.get("id", f"candidate_{rank}")
            print(f"\n[{rank}/{len(population)}] {candidate_id}")

            try:
                write_heuristic(candidate.get("code"))
            except Exception as exc:
                for size in args.sizes:
                    rows.append(
                        {
                            "rank": rank,
                            "candidate_id": candidate_id,
                            "train_objective": candidate.get("objective"),
                            "selection_score": candidate.get("selection_score"),
                            "risk_score": candidate.get("risk_score"),
                            "status": candidate.get("status"),
                            "generation": candidate.get("generation"),
                            "size": size,
                            "test_objective": "",
                            "timecost": "",
                            "error": str(exc),
                        }
                    )
                print(f"  skipped: {exc}")
                continue

            for size in args.sizes:
                try:
                    objective, timecost = evaluate_one_size(size, args.n_test, args.debug)
                    rows.append(
                        {
                            "rank": rank,
                            "candidate_id": candidate_id,
                            "train_objective": candidate.get("objective"),
                            "selection_score": candidate.get("selection_score"),
                            "risk_score": candidate.get("risk_score"),
                            "status": candidate.get("status"),
                            "generation": candidate.get("generation"),
                            "size": size,
                            "test_objective": objective,
                            "timecost": timecost,
                            "error": "",
                        }
                    )
                    print(
                        f"  size={size}: objective={objective:.6f} "
                        f"timecost={timecost:.3f}s"
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "rank": rank,
                            "candidate_id": candidate_id,
                            "train_objective": candidate.get("objective"),
                            "selection_score": candidate.get("selection_score"),
                            "risk_score": candidate.get("risk_score"),
                            "status": candidate.get("status"),
                            "generation": candidate.get("generation"),
                            "size": size,
                            "test_objective": "",
                            "timecost": "",
                            "error": repr(exc),
                        }
                    )
                    print(f"  size={size}: failed: {exc}")
    finally:
        if not args.keep_last:
            HEURISTIC_FILE.write_bytes(original_heuristic)
            importlib.invalidate_caches()
            sys.modules.pop("heuristic", None)

    fieldnames = [
        "rank",
        "candidate_id",
        "train_objective",
        "selection_score",
        "risk_score",
        "status",
        "generation",
        "size",
        "test_objective",
        "timecost",
        "error",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    print(f"\nWrote CSV: {output_csv}")
    print(f"Wrote JSON: {output_json}")
    summarize(rows)


if __name__ == "__main__":
    main()
