from copy import deepcopy
from pathlib import Path

import yaml


def load_config(config_path):
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    parent = config.pop("inherits", None)
    if parent:
        parent_config = load_config(parent)
        return _deep_update(parent_config, config)
    return config


def _deep_update(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result

