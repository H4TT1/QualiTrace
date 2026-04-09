# Backward-compatible shim. New model implementations live in src/models/.
from models.ae import AnomalyAE, MODEL_VARIANTS

__all__ = ["AnomalyAE", "MODEL_VARIANTS"]
