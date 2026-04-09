from typing import Callable, Dict, Tuple


ExperimentRunner = Callable[[dict, dict], None]
_RUNNER_REGISTRY: Dict[Tuple[str, str], ExperimentRunner] = {}


def register_runner(family: str, architecture: str):
    def decorator(runner: ExperimentRunner):
        key = (family, architecture)
        _RUNNER_REGISTRY[key] = runner
        return runner

    return decorator


def get_runner(model_cfg: dict) -> ExperimentRunner:
    family = model_cfg.get("family", "ae")
    architecture = model_cfg.get("architecture", "conv_ae")
    key = (family, architecture)

    if key not in _RUNNER_REGISTRY:
        available = [f"{fam}/{arch}" for fam, arch in sorted(_RUNNER_REGISTRY.keys())]
        raise ValueError(
            f"No experiment runner registered for family='{family}', architecture='{architecture}'. "
            f"Available: {available}"
        )

    return _RUNNER_REGISTRY[key]
