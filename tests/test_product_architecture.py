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
        assert callable(getattr(api, "run_analysis", None))
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



def test_named_product_workflow_wrappers_dispatch_to_their_own_registered_ids(monkeypatch):
    from importlib import import_module
    from tis.products import load_product_api

    for product in list_products():
        api = load_product_api(product["product_id"])
        api_impl = import_module(product["api_module"] + ".api")
        monkeypatch.setattr(
            api_impl,
            "execute_product_analysis",
            lambda product_id, analysis_id, *args, **kwargs: (
                product_id, analysis_id, args, kwargs
            ),
        )
        if product["id"] == 1:
            calls = [
                ("analyze_spectra", 1),
                ("analyze_cts", 30),
                ("reconstruct_profiles", 46),
                ("analyze_spectral_distribution", 81),
                ("fit_gyrotropic", 82),
                ("fit_temperature_tensor", 83),
                ("analyze_landau_fdt_convergence", 84),
                ("reconstruct_relativistic_evdf", 85),
                ("compare_temperature_tensor_models", 86),
                ("infer_path_averaged_density", 87),
            ]
        elif product["id"] == 2:
            calls = [
                ("analyze_langmuir_iv", 2),
                ("analyze_probe_iv_compatibility", 16),
                ("analyze_ball_pen", 17),
                ("analyze_gundestrup", 18),
                ("analyze_multipin", 19),
                ("analyze_potential_arrays", 20),
            ]
        else:
            calls = [(entry["function"], entry["id"]) for entry in product["entry_points"]]
        for function_name, expected_id in calls:
            function = getattr(api, function_name)
            if product["id"] == 2 and function_name in {"analyze_langmuir_iv", "analyze_probe_iv_compatibility"}:
                result = function("voltage", "current")
            elif product["id"] == 2 and function_name in {"analyze_ball_pen"}:
                result = function([0.0], [1.0])
            elif product["id"] == 2 and function_name == "analyze_gundestrup":
                result = function([0.0, 1.0, 2.0], [1.0, 2.0, 1.0])
            elif product["id"] == 2 and function_name == "analyze_multipin":
                result = function([0.0, 1.0], [[0.0, 1.0], [1.0, 2.0]], [0.0, 1.0])
            elif product["id"] == 2 and function_name == "analyze_potential_arrays":
                result = function([2.0], [1.0])
            else:
                result = function("sentinel")
            assert result[0] == product["product_id"]
            assert result[1] == expected_id
