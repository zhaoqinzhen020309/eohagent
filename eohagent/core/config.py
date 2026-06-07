import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        out = value
        for key, env_value in os.environ.items():
            out = out.replace("${" + key + "}", env_value)
        return out
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(config_path: str | Path) -> dict:
    root = _repo_root()
    default_path = root / "eohagent" / "config" / "default.yaml"
    with default_path.open("r", encoding="utf-8") as f:
        default_config = yaml.safe_load(f) or {}

    path = Path(config_path)
    if not path.is_absolute():
        path = root / path
    with path.open("r", encoding="utf-8") as f:
        user_config = yaml.safe_load(f) or {}

    config = _expand_env(_deep_merge(default_config, user_config))
    config["_repo_root"] = str(root)
    return config


def resolve_path(config: dict, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path(config["_repo_root"]) / path
