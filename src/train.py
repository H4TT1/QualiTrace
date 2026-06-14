import argparse

from config_utils import load_config, resolve_paths
from experiments import get_runner


def train(config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    paths = resolve_paths(config)
    model_cfg = config.get("model", {})

    runner = get_runner(model_cfg)
    runner(config, paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the selected QualiTrace training pipeline.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    train(config_path=args.config)
