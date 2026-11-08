from __future__ import annotations

import pytest

from tis.products.catalog import (
    get_product,
    list_products,
    list_product_analyses,
    product_for_analysis,
    validate_product_catalog,
)
from tis.products.api import execute_product_analysis


def test_product_catalog_reconciles_all_families_and_registered_workflows():
    status = validate_product_catalog()
    assert status == {
        "product_count": 20,
        "product_ids_ordered": True,
        "analysis_entry_point_count": 86,
        "analysis_ids_unique": True,
        "every_analysis_has_one_product_owner": True,
        "product_slugs_unique": True,
        "product_api_modules_unique": True,
        "instrumentation_support_is_cross_cutting": True,
    }
    assert len(list_products()) == 20


def test_product_lookup_accepts_stable_slug_and_returns_defensive_copy():
    by_id = get_product(1)
    by_slug = get_product("thomson-scattering")
    assert by_id["id"] == by_slug["id"] == 1
    assert by_id["entry_ids"] == [1, 30, 46, 81, 82, 83, 84, 85, 86, 87]
    by_id["entry_ids"].clear()
    assert len(get_product(1)["entry_ids"]) == 10


def test_every_registered_analysis_has_exactly_one_product_owner():
    for product in list_products():
        for entry_id in product["entry_ids"]:
            assert product_for_analysis(entry_id)["id"] == product["id"]


def test_product_analysis_listing_uses_canonical_registry_contracts():
    entries = list_product_analyses(1)
    assert [entry["id"] for entry in entries] == get_product(1)["entry_ids"]
    assert entries[0]["module"] == "thomson_workflow"
    assert entries[0]["function"] == "analyze_thomson_spectra"


def test_execution_api_forwards_arguments_without_changing_measurements(monkeypatch):
    import tis.products.api as api

    captured = {}
    sentinel = object()

    def solver(wavelength_nm, spectra, *, calibration_tag):
        captured["wavelength_nm"] = wavelength_nm
        captured["spectra"] = spectra
        captured["calibration_tag"] = calibration_tag
        return sentinel

    monkeypatch.setattr(api, "resolve_analysis", lambda analysis_id: solver)
    wavelength = [526.0, 527.0, 528.0]
    spectra = [[1.0, 2.0, 1.0]]
    result = execute_product_analysis(
        "thomson-scattering", 1, wavelength, spectra,
        calibration_tag="operator-calibrated",
    )
    assert result is sentinel
    assert captured == {
        "wavelength_nm": wavelength,
        "spectra": spectra,
        "calibration_tag": "operator-calibrated",
    }


def test_execution_api_rejects_cross_product_analysis_ids_before_solver(monkeypatch):
    import tis.products.api as api

    def forbidden(_analysis_id):
        raise AssertionError("solver must not be resolved for an ownership violation")

    monkeypatch.setattr(api, "resolve_analysis", forbidden)
    with pytest.raises(ValueError, match="not owned by product"):
        execute_product_analysis("thomson-scattering", 2)


def test_product_lookup_rejects_unknown_and_boolean_identifiers():
    with pytest.raises(ValueError, match="product_id"):
        get_product(True)
    with pytest.raises(KeyError, match="Unknown specialist product"):
        get_product("not-a-product")

def test_dependency_inventory_separates_solver_closures_from_shared_platform_imports():
    from tis.products.dependencies import build_dependency_inventory

    inventory = build_dependency_inventory()
    assert inventory["product_count"] == 20
    products = {row["product_id"]: row for row in inventory["products"]}
    thomson = products["thomson-scattering"]
    langmuir = products["langmuir-electrostatic-probes"]
    assert "tokamak_analysis.thomson_workflow" in thomson["entry_point_modules"]
    assert "tokamak_analysis.thomson_inference" in thomson["transitive_domain_modules"]
    assert "tokamak_analysis.langmuir_probe" in langmuir["entry_point_modules"]
    assert "tokamak_analysis.langmuir_probe" in langmuir["product_local_domain_modules"]
    assert set(inventory["shared_domain_modules"]) <= {
        module for row in products.values()
        for module in row["transitive_domain_modules"]
    }
    assert all("tokamak_analysis." in module for module in inventory["shared_domain_modules"])



def test_all_twenty_products_have_importable_public_package_boundaries():
    from tis.products import load_product_api

    for product in list_products():
        api = load_product_api(product["product_id"])
        assert api.PRODUCT_ID == product["product_id"]
        assert callable(getattr(api, "run_analysis", None)) or callable(
            getattr(api, "run_probe_analysis", None)
        ) or product["id"] == 1
        assert product["api_module"].startswith("tis.products.")
        for entry in product["entry_points"]:
            if product["id"] >= 3:
                assert callable(getattr(api, entry["function"]))


def test_product_packages_do_not_import_another_products_private_modules():
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "tis" / "products"
    for path in root.glob("*/**/*.py"):
        if path.parent == root:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not (
                    name.startswith("tis.products.")
                    and name.split(".")[2] not in {"api", "catalog", "dependencies"}
                    and name.split(".")[2] != path.parent.name
                ), f"{path} imports another product's private module: {name}"
