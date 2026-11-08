"""Canonical catalogue loader.

The JSON registries are the single source of truth for diagnostic and formula metadata.
Do not maintain a second hand-copied catalogue in Python: it drifts from the UI and API.
"""
from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

_ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_DATA = Path(__file__).resolve().parent / "data"
_DIAGNOSTICS_PATH = _ROOT / "DIAGNOSTIC_REGISTRY.json"
_FORMULAS_PATH = _ROOT / "FORMULA_REGISTRY.json"
if not _DIAGNOSTICS_PATH.is_file():
    _DIAGNOSTICS_PATH = _PACKAGE_DATA / "DIAGNOSTIC_REGISTRY.json"
if not _FORMULAS_PATH.is_file():
    _FORMULAS_PATH = _PACKAGE_DATA / "FORMULA_REGISTRY.json"

def _load_list(path: Path) -> list[dict]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot load canonical catalogue {path.name}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise RuntimeError(f"{path.name} must contain a JSON list of objects")
    return value

@lru_cache(maxsize=1)
def _cached_diagnostics() -> tuple[dict, ...]:
    return tuple(_load_list(_DIAGNOSTICS_PATH))

@lru_cache(maxsize=1)
def _cached_formulas() -> tuple[dict, ...]:
    return tuple(_load_list(_FORMULAS_PATH))

# Public list-compatible copies preserve the existing API while reading canonical JSON.
DIAGNOSTICS = [dict(item) for item in _cached_diagnostics()]
FORMULAS = [dict(item) for item in _cached_formulas()]

def maturity_label(record: Mapping[str, Any]) -> str:
    """Return a conservative, user-facing implementation maturity label.

    These labels describe software scope only; none means experimentally validated.
    """
    contract = record.get("execution_contract") or {}
    level = contract.get("execution_level", "reference_only")
    completeness = contract.get("completeness", "incomplete")
    if level == "reference_only":
        return "Reference only"
    if level == "limited_calculation":
        return "Limited calculation"
    if completeness == "complete_with_declared_scope":
        return "Complete within declared scope — not independently validated"
    if completeness == "complete":
        return "Complete within declared computational scope — not independently validated"
    if level == "analysis_pipeline":
        return "Analysis pipeline — incomplete"
    return "Unclassified — incomplete"


def maturity_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Count maturity labels for provenance and report summaries."""
    from collections import Counter

    return dict(sorted(Counter(maturity_label(record) for record in records).items()))


def diagnostic_families() -> list[str]:
    return sorted({str(item["family"]) for item in DIAGNOSTICS if item.get("family")})
