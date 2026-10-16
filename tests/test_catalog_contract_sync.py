import json
from pathlib import Path
from tis.catalog import DIAGNOSTICS, FORMULAS

ROOT = Path(__file__).resolve().parents[1]

def test_ui_catalogue_is_loaded_from_canonical_diagnostic_registry():
    rows = json.loads((ROOT / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    assert len(DIAGNOSTICS) == 80
    assert DIAGNOSTICS == rows
    assert all("execution_contract" in row for row in DIAGNOSTICS)

def test_formula_catalogue_is_loaded_from_canonical_formula_registry():
    rows = json.loads((ROOT / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    assert len(FORMULAS) == 37
    assert FORMULAS == rows
    assert all(row["execution_contract"]["implementation_link_status"] == "not_mapped" for row in FORMULAS)

def test_package_and_runtime_versions_are_consistent():
    import re
    from tis import __version__
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'(?m)^version = "([^"]+)"
, metadata).group(1) == __version__
