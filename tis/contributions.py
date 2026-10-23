"""Create portable, provenance-aware reports for community-contributed analysis cases.

The bundle intentionally excludes raw experimental data by default. Researchers can
review the generated JSON and attach their data separately only when they choose.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Mapping


def _json_safe(value: Any) -> Any:
    """Convert common numerical results to strict JSON-compatible values."""
    if hasattr(value, "tolist") and callable(value.tolist):
        return _json_safe(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        # JSON NaN/Infinity are non-standard and break many scientific tools.
        import math
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def build_contribution_bundle(
    *,
    analysis_id: int,
    analysis_name: str,
    software_version: str,
    parameters: Mapping[str, Any],
    result: Any,
    dataset_sha256: str | None = None,
    dataset_name: str | None = None,
    tokamak: str | None = None,
    shot_id: str | int | None = None,
    diagnostic: str | None = None,
    data_units: Mapping[str, str] | None = None,
    reference_result: Any = None,
    reference_description: str | None = None,
    researcher_notes: str | None = None,
) -> dict[str, Any]:
    """Build an exportable contribution record without embedding raw data."""
    if isinstance(analysis_id, bool) or not isinstance(analysis_id, int) or analysis_id < 1:
        raise ValueError("analysis_id must be a positive integer.")
    if not isinstance(analysis_name, str) or not analysis_name.strip():
        raise ValueError("analysis_name must be a nonempty string.")
    if not isinstance(software_version, str) or not software_version.strip():
        raise ValueError("software_version must be a nonempty string.")
    if not isinstance(parameters, Mapping):
        raise ValueError("parameters must be a mapping.")
    if not isinstance(result, Mapping):
        raise ValueError("result must be a mapping.")
    if input_provenance is not None and not isinstance(input_provenance, Mapping):
        raise ValueError("input_provenance must be a mapping when supplied.")
    if data_units is not None and not isinstance(data_units, Mapping):
        raise ValueError("data_units must be a mapping when supplied.")
    if dataset_sha256 is not None and (
        not isinstance(dataset_sha256, str)
        or len(dataset_sha256) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in dataset_sha256)
    ):
        raise ValueError("dataset_sha256 must be a 64-character hexadecimal SHA-256 digest.")
    created = datetime.now(timezone.utc).isoformat()
    bundle = {
        "schema": "tokamak-insight-contribution/1.0",
        "created_at_utc": created,
        "software": {"name": "Tokamak Insight Suite", "version": software_version},
        "analysis": {"id": analysis_id, "name": analysis_name},
        "dataset": {
            "name": dataset_name,
            "sha256": dataset_sha256,
            "raw_data_embedded": False,
        },
        "experiment": {
            "tokamak": tokamak,
            "shot_id": str(shot_id) if shot_id is not None else None,
            "diagnostic": diagnostic,
            "units": _json_safe(data_units or {}),
        },
        "parameters": _json_safe(parameters),
        "software_result": _json_safe(result),
        "reference": {
            "result": _json_safe(reference_result),
            "description": reference_description,
        } if reference_result is not None or reference_description else None,
        "researcher_notes": researcher_notes,
        "review_status": "community-submitted-not-independently-verified",
        "privacy_notice": (
            "This file does not include raw data by default. Inspect it before sharing; "
            "remove confidential metadata and obtain any required institutional approval."
        ),
    }
    # Stable digest makes the exact exported record identifiable after serialization.
    canonical = json.dumps(bundle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    bundle["bundle_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return bundle


def bundle_json(bundle: Mapping[str, Any]) -> str:
    """Serialize a contribution bundle as readable UTF-8 JSON."""
    return json.dumps(_json_safe(bundle), ensure_ascii=False, indent=2, allow_nan=False)
