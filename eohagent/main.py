from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eohagent.core.config import load_config
from eohagent.core.engine import ExpertEoHEngine
from eohagent.core.experiment_guard import build_smoke_config
from eohagent.core.langgraph_workflow import LangGraphWorkflowRunner


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the expert-harnessed EoH framework.")
    parser.add_argument("--config", default="eohagent/config/bp_online.yaml", help="Path to a YAML config.")
    parser.add_argument("--smoke", action="store_true", help="Only bootstrap and evaluate baseline seeds.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.smoke:
        config = build_smoke_config(config)
    else:
        config.setdefault("run", {})
        config["run"].setdefault("mode", "run")
    engine = ExpertEoHEngine(config)
    if config.get("agent", {}).get("workflow") == "langgraph":
        population = LangGraphWorkflowRunner(engine).run(smoke=args.smoke)
    else:
        population = engine.smoke() if args.smoke else engine.run()
    print("Final population:")
    for candidate in population:
        print(f"{candidate.id}: objective={candidate.objective}, status={candidate.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
