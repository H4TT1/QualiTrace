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
