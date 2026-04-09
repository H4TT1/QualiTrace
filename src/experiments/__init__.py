from .registry import get_runner, register_runner

# Trigger registration for built-in runners.
from . import ae  # noqa: F401

# Optional runners can fail to import if extra deps are missing.
try:
    from . import vlm_clip  # noqa: F401
except ImportError:
    pass

__all__ = ["get_runner", "register_runner"]
