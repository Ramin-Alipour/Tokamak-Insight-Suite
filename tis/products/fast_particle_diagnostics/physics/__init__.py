"""Product-owned solvers with shared-domain dependency fallback."""
from pathlib import Path

_SHARED_PHYSICS = Path(__file__).resolve().parents[3] / "physics" / "shared"
if _SHARED_PHYSICS.is_dir():
    __path__.append(str(_SHARED_PHYSICS))
