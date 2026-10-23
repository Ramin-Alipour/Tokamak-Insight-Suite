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
    from importlib import import_module
    mapped = [row for row in FORMULAS if row["execution_contract"]["implementation_link_status"] == "mapped"]
    assert len(mapped) == 37
    assert sum(row["execution_contract"]["implementation_link_status"] == "not_mapped" for row in FORMULAS) == 0
    for row in mapped:
        contract = row["execution_contract"]
        module_name, function_name = contract["function_link"].rsplit(".", 1)
        assert callable(getattr(import_module(module_name), function_name, None))
        assert contract["input_specs"] and contract["output_specs"] and contract["test_links"]
        assert contract["uncertainty_model"] != "not_specified"

def test_package_and_runtime_versions_are_consistent():
    from tis import __version__
    metadata = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_line = next(line for line in metadata.splitlines() if line.startswith("version = "))
    assert version_line.split("=", 1)[1].strip().strip('"') == __version__


def test_all_analysis_entry_points_declare_conservative_contracts():
    from tokamak_analysis.analysis_registry import ANALYSES
    assert len(ANALYSES) == 80
    for entry in ANALYSES:
        contract = entry["execution_contract"]
        assert contract["execution_level"] in {"limited_calculation", "analysis_pipeline"}
        assert contract["completeness"] == "incomplete"
        assert contract["raw_data_to_physics"] is (entry["name"].casefold() == "thomson scattering")
        primary_link = f'{entry["module"]}.{entry["function"]}'
        assert contract["function_links"][0] == primary_link
        if entry["name"].casefold() == "thomson scattering":
            assert len(contract["function_links"]) == 5
        else:
            assert contract["function_links"] == [primary_link]


def test_thomson_registry_links_resolve_to_real_callables():
    import importlib

    thomson = next(row for row in DIAGNOSTICS if row["diagnostic"].casefold() == "thomson scattering")
    links = thomson["implementation_functions"]
    assert len(links) == 5
    for link in links:
        module_name, function_name = link.rsplit(".", 1)
        module = importlib.import_module(module_name)
        assert callable(getattr(module, function_name, None)), link


def test_diagnostic_registry_does_not_claim_all_80_analyses_are_complete():
    from collections import Counter

    counts = Counter(row["implementation_status"] for row in DIAGNOSTICS)
    assert counts["Registry reference only"] == 59
    assert counts["Implemented numerical workflow (see limits)"] == 20
    assert sum("overall analysis remains incomplete" in status for status in counts) == 1
    assert all(row["execution_contract"]["completeness"] == "incomplete" for row in DIAGNOSTICS)


def test_packaged_registry_data_matches_repository_registries():
    for name in ("DIAGNOSTIC_REGISTRY.json", "FORMULA_REGISTRY.json"):
        repository_copy = json.loads((ROOT / name).read_text(encoding="utf-8"))
        packaged_copy = json.loads((ROOT / "tis" / "data" / name).read_text(encoding="utf-8"))
        assert packaged_copy == repository_copy
