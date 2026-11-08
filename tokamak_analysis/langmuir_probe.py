"""Backward-compatible import facade for relocated physical solvers."""
from importlib import import_module as _import_module
from pathlib import Path as _Path
from tis.physics.legacy_map import LEGACY_MODULE_TARGETS as _TARGETS

_target = _TARGETS.get(_Path(__file__).stem)
if _target is None:
    raise ImportError(f"No relocated implementation registered for {_Path(__file__).stem}")
_module = _import_module(_target)
for _key in dir(_module):
    if not _key.startswith("__"):
        globals()[_key] = getattr(_module, _key)
__all__ = getattr(_module, "__all__", [key for key in globals() if not key.startswith("_")])
