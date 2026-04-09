from config_utils import load_config, resolve_paths
from experiments import get_runner


def train():
    config = load_config()
    paths = resolve_paths(config)
    model_cfg = config.get("model", {})

    runner = get_runner(model_cfg)
    runner(config, paths)


if __name__ == "__main__":
    train()
