import os
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_paths(config: dict) -> dict:
    env = os.environ.get("QUALITRACE_ENV", config.get("environment", "local"))
    if env not in config["paths"]:
        raise ValueError(f"Unknown environment '{env}'. Expected one of: {list(config['paths'].keys())}")
    return config["paths"][env]


def resolve_data_info(config: dict, paths: dict | None = None) -> dict:
    data_cfg = dict(config.get("data", {}))
    resolved_paths = paths or resolve_paths(config)
    data_cfg.setdefault("name", "dataset")
    data_cfg.setdefault("version", "unversioned")
    data_cfg.setdefault("source", resolved_paths.get("data_dir", ""))
    data_cfg["data_dir"] = resolved_paths.get("data_dir", "")
    return data_cfg
