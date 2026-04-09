from typing import Callable, Dict, Tuple


ModelBuilder = Callable[[dict, dict], object]
_MODEL_REGISTRY: Dict[Tuple[str, str], ModelBuilder] = {}


def register_model(family: str, architecture: str):
    def decorator(builder: ModelBuilder):
        key = (family, architecture)
        _MODEL_REGISTRY[key] = builder
        return builder

    return decorator


def build_model(model_cfg: dict, train_cfg: dict):
    family = model_cfg.get("family", "ae")
    architecture = model_cfg.get("architecture", "conv_ae")
    key = (family, architecture)

    if key not in _MODEL_REGISTRY:
        available = [f"{fam}/{arch}" for fam, arch in sorted(_MODEL_REGISTRY.keys())]
        raise ValueError(
            f"No model registered for family='{family}', architecture='{architecture}'. "
            f"Available: {available}"
        )

    return _MODEL_REGISTRY[key](model_cfg, train_cfg)
