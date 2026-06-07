from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
from pathlib import Path

from eohagent.core.config import resolve_path


EXPERIMENT_STATE_MARKERS = {
    "candidate_store.json",
    "compact_context.json",
    "expert_notebook.json",
    "mechanism_archive.json",
    "problem_card.json",
    "progress.jsonl",
    "run_manifest.json",
    "session_log.jsonl",
    "results",
}


def prepare_output_dir(config: dict) -> dict:
    output_dir = resolve_path(config, config["project"]["output_dir"])
    run_config = config.get("run", {})
    mode = str(run_config.get("mode") or "run")
    resume = bool(run_config.get("resume", False))
    allow_dirty = bool(run_config.get("allow_dirty_output_dir", False))
    dirty = dirty_output_markers(output_dir)
    if mode != "smoke" and dirty and not resume and not allow_dirty:
        joined = ", ".join(dirty)
        raise RuntimeError(
            f"output_dir contains existing experiment state: {output_dir} ({joined}). "
            "Use a fresh output_dir, delete the directory, or set run.resume=true explicitly."
        )
    if resume:
        _verify_resume_manifest(config, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_run_manifest(config, mode)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return manifest


def dirty_output_markers(output_dir: Path) -> list[str]:
    if not output_dir.exists():
        return []
    return sorted(name for name in EXPERIMENT_STATE_MARKERS if (output_dir / name).exists())


def build_smoke_config(config: dict) -> dict:
    smoke_config = deepcopy(config)
    base_output = resolve_path(smoke_config, smoke_config["project"]["output_dir"])
    stamp = time.strftime("%Y%m%d_%H%M%S")
    smoke_output = base_output / "_smoke" / stamp
    smoke_config["project"]["formal_output_dir"] = str(base_output)
    smoke_config["project"]["output_dir"] = str(smoke_output)
    smoke_config.setdefault("run", {})
    smoke_config["run"]["mode"] = "smoke"
    smoke_config["run"]["resume"] = False
    smoke_config["run"]["allow_dirty_output_dir"] = False
    return smoke_config


def build_run_manifest(config: dict, mode: str) -> dict:
    project = config.get("project", {})
    evaluator = config.get("evaluator", {})
    llm = config.get("llm", {})
    evolution = config.get("evolution", {})
    return {
        "mode": mode,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config_hash": _config_hash(config),
        "project": {
            "name": project.get("name"),
            "problem": project.get("problem"),
            "output_dir": project.get("output_dir"),
        },
        "evaluator": {
            "problem": evaluator.get("problem"),
            "function_name": evaluator.get("function_name"),
            "architecture_class_name": evaluator.get("architecture_class_name"),
            "n_inst_eva": evaluator.get("n_inst_eva"),
            "time_limit": evaluator.get("time_limit"),
            "ite_max": evaluator.get("ite_max"),
            "perturbation_moves": evaluator.get("perturbation_moves"),
        },
        "llm": {
            "provider": llm.get("provider"),
            "model": llm.get("model"),
            "api_base": llm.get("api_base"),
            "thinking": llm.get("thinking"),
            "temperature": llm.get("temperature"),
        },
        "evolution": {
            "pop_size": evolution.get("pop_size"),
            "generations": evolution.get("generations"),
            "offspring_per_generation": evolution.get("offspring_per_generation"),
            "survival_selection": evolution.get("survival_selection"),
            "allow_archive_parents": bool(evolution.get("allow_archive_parents", False)),
        },
    }


def _verify_resume_manifest(config: dict, output_dir: Path) -> None:
    manifest_path = output_dir / "run_manifest.json"
    if not manifest_path.exists():
        raise RuntimeError(f"run.resume=true but run_manifest.json is missing in {output_dir}")
    previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    current_hash = _config_hash(config)
    if previous.get("config_hash") != current_hash:
        raise RuntimeError("run.resume=true but config_hash does not match run_manifest.json")


def _config_hash(config: dict) -> str:
    normalized = deepcopy(config)
    normalized.pop("_repo_root", None)
    normalized.pop("_repo_root_path", None)
    if isinstance(normalized.get("llm"), dict):
        normalized["llm"].pop("api_key", None)
    if isinstance(normalized.get("run"), dict):
        normalized["run"].pop("mode", None)
        normalized["run"].pop("resume", None)
        normalized["run"].pop("allow_dirty_output_dir", None)
    payload = json.dumps(normalized, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
