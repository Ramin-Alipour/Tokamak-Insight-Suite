import json
from pathlib import Path
from tis.catalog import DIAGNOSTICS, FORMULAS, maturity_label

ROOT = Path(__file__).resolve().parents[1]

def test_ui_catalogue_is_loaded_from_canonical_diagnostic_registry():
    rows = json.loads((ROOT / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    assert len(DIAGNOSTICS) == 80
    assert DIAGNOSTICS == rows
    assert all("execution_contract" in row for row in DIAGNOSTICS)

def test_formula_catalogue_is_loaded_from_canonical_formula_registry():
    rows = json.loads((ROOT / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    assert len(FORMULAS) == len(rows)
    assert FORMULAS == rows
    from importlib import import_module
    mapped = [row for row in FORMULAS if row["execution_contract"]["implementation_link_status"] == "mapped"]
    assert len(mapped) == len(rows)
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


def test_analysis_titles_numbers_and_order_match_canonical_80_diagnostic_catalog():
    from tokamak_analysis.analysis_registry import ANALYSES

    expected = [row["diagnostic"] for row in DIAGNOSTICS]
    assert [row["id"] for row in ANALYSES] == list(range(1, 81))
    assert [row["name"] for row in ANALYSES] == expected
    assert [row["diagnostic"] for row in DIAGNOSTICS] == expected


def test_all_analysis_entry_points_declare_conservative_contracts():
    import importlib
    from tokamak_analysis.analysis_registry import ANALYSES
    assert len(ANALYSES) == 80
    for entry in ANALYSES:
        contract = entry["execution_contract"]
        assert contract["execution_level"] in {"limited_calculation", "analysis_pipeline"}
        assert contract["completeness"] in {"incomplete", "complete", "complete_with_declared_scope"}
        if contract["completeness"] != "incomplete":
            assert contract["limitations"], "Scoped completion needs limitations: " + entry["name"]
        if contract["raw_data_to_physics"]:
            assert contract["execution_level"] == "analysis_pipeline"
            assert contract["limitations"], entry["name"]
        primary_link = f'tokamak_analysis.{entry["module"]}.{entry["function"]}'
        assert contract["function_links"][0] == primary_link
        for link in contract["function_links"]:
            module_name, function_name = link.rsplit(".", 1)
            assert callable(getattr(importlib.import_module(module_name), function_name, None)), link
        if entry["name"].casefold() == "thomson scattering":
            assert len(contract["function_links"]) == 3


def test_thomson_registry_links_resolve_to_real_callables():
    import importlib

    thomson = next(row for row in DIAGNOSTICS if row["diagnostic"].casefold() == "thomson scattering")
    links = thomson["implementation_functions"]
    assert len(links) == 2
    for link in links:
        module_name, function_name = link.rsplit(".", 1)
        module = importlib.import_module(module_name)
        assert callable(getattr(module, function_name, None)), link


def test_diagnostic_registry_reports_actual_execution_levels_and_conservative_scope():
    from collections import Counter

    levels = Counter(row["execution_contract"]["execution_level"] for row in DIAGNOSTICS)
    completeness = Counter(row["execution_contract"]["completeness"] for row in DIAGNOSTICS)
    assert len(DIAGNOSTICS) == 80
    assert levels == {"reference_only": 49, "limited_calculation": 16, "analysis_pipeline": 15}
    assert completeness == {"incomplete": 64, "complete": 7, "complete_with_declared_scope": 9}
    assert all(row["implementation_status"] for row in DIAGNOSTICS)
    # "Complete" describes the declared computational scope, not empirical
    # validation of a particular machine or calibration.
    for row in DIAGNOSTICS:
        contract = row["execution_contract"]
        if contract["completeness"] != "incomplete":
            assert contract["limitations"], row["diagnostic"]


def test_packaged_registry_data_matches_repository_registries():
    for name in ("DIAGNOSTIC_REGISTRY.json", "FORMULA_REGISTRY.json"):
        repository_copy = json.loads((ROOT / name).read_text(encoding="utf-8"))
        packaged_copy = json.loads((ROOT / "tis" / "data" / name).read_text(encoding="utf-8"))
        assert packaged_copy == repository_copy


def test_cxrs_registry_links_resolve_and_contract_is_scoped():
    import importlib
    rows = json.loads((ROOT / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    row = next(item for item in rows if item["diagnostic"] == "Charge-exchange recombination spectroscopy (CXRS)")
    assert row["execution_contract"]["completeness"] == "complete_with_declared_scope"
    assert row["execution_contract"]["input_specs"]
    assert row["execution_contract"]["output_specs"]
    assert row["execution_contract"]["uncertainty_model"] != "not_specified"
    assert row["implementation_functions"] == ["tokamak_analysis.cxrs.analyze_cxrs_line"]
    module = importlib.import_module("tokamak_analysis.cxrs")
    assert callable(module.analyze_cxrs_line)


def test_toroidal_field_coil_analysis_contract_and_callable():
    import importlib
    rows = json.loads((ROOT / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    row = next(item for item in rows if item["diagnostic"] == "Toroidal-field coils / magnetic probes")
    contract = row["execution_contract"]
    assert contract["completeness"] == "complete"
    assert contract["input_specs"] and contract["output_specs"]
    assert contract["uncertainty_model"] != "not_specified"
    assert row["implementation_functions"] == [
        "tokamak_analysis.toroidal_field_coils.analyze_toroidal_field_coil",
        "tokamak_analysis.toroidal_field_coils.integrate_pickup_coil_voltage",
    ]
    module = importlib.import_module("tokamak_analysis.toroidal_field_coils")
    assert callable(module.analyze_toroidal_field_coil)

def test_maturity_labels_distinguish_reference_calculation_and_unvalidated_pipeline():
    reference = next(row for row in DIAGNOSTICS if row["execution_contract"]["execution_level"] == "reference_only")
    limited = next(row for row in DIAGNOSTICS if row["execution_contract"]["execution_level"] == "limited_calculation")
    thomson = next(row for row in DIAGNOSTICS if row["diagnostic"].casefold() == "thomson scattering")
    scoped = next(row for row in DIAGNOSTICS if row["execution_contract"]["completeness"] == "complete_with_declared_scope")
    assert maturity_label(reference) == "Reference only"
    assert maturity_label(limited) == "Limited calculation"
    assert "Complete within declared scope" in maturity_label(thomson)
    assert "not independently validated" in maturity_label(scoped)


def test_thomson_spatial_reconstruction_is_in_analysis_contract():
    from tokamak_analysis.analysis_registry import ANALYSES
    import importlib

    entry = ANALYSES[0]
    contract = entry["execution_contract"]
    assert entry["name"] == "Thomson scattering"
    assert contract["completeness"] == "complete_with_declared_scope"
    reconstruction = next(
        mode for mode in contract["workflow_modes"]
        if mode["name"] == "spatial_profile_reconstruction"
    )
    assert reconstruction["function"] == "tokamak_analysis.thomson_scattering.reconstruct_thomson_profile"
    assert "No extrapolation" in reconstruction["scope"]
    module = importlib.import_module("tokamak_analysis.thomson_scattering")
    assert callable(module.reconstruct_thomson_profile)
