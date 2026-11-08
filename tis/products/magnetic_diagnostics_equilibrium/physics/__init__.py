"""Product-owned solver namespace with legacy shared-dependency fallback.

Registered solver modules live in this product package. Until shared domain
primitives are extracted into a neutral package, unresolved sibling imports are
resolved from the compatibility namespace without copying shared implementations.
"""
from pathlib import Path

_LEGACY_DOMAIN = Path(__file__).resolve().parents[4] / "tokamak_analysis"
if _LEGACY_DOMAIN.is_dir():
    __path__.append(str(_LEGACY_DOMAIN))
