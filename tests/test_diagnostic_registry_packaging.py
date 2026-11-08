import json
from pathlib import Path


def test_diagnostic_registry_package_copy_matches_canonical_registry():
    root = Path(__file__).resolve().parents[1]
    canonical = json.loads((root / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    packaged = json.loads((root / "tis" / "data" / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    assert packaged == canonical
