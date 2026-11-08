import json
from pathlib import Path


def test_general_relativistic_3d_evdf_formula_is_registered_and_packaged():
    root = Path(__file__).resolve().parents[1]
    formula = json.loads((root / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    packaged = json.loads((root / "tis" / "data" / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    assert formula == packaged
    matches = [
        item for item in formula
        if item.get("execution_contract", {}).get("formula_key")
        == "thomson_general_relativistic_3d_evdf_kernel"
    ]
    assert len(matches) == 1
    contract = matches[0]["execution_contract"]
    assert contract["implementation_link_status"] == "mapped"
    assert "build_relativistic_3d_thomson_kernels" in contract["function_links"]
    assert "reconstruct_relativistic_optical_thomson_3d_evdf" in contract["function_links"]
    assert any("Klein-Nishina" in limitation for limitation in contract["limitations"])
