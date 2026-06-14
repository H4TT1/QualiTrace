# Backward-compatible shim. New model implementations live in src/models/.
from models.patchcore import PatchCoreModel

__all__ = ["PatchCoreModel"]
