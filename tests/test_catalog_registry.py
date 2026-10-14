import json
from importlib.resources import files

from tis.catalog import DIAGNOSTICS, FORMULAS


def test_diagnostic_catalog_has_unique_names_and_stable_ids():
    names = [item["diagnostic"].strip().casefold() for item in DIAGNOSTICS]
    ids = [item["id"] for item in DIAGNOSTICS]
    assert len(names) == len(set(names))
    assert len(ids) == len(set(ids))
    assert all(item["implementation_status"] for item in DIAGNOSTICS)


def test_formula_catalog_distinguishes_inference_from_validation():
    allowed = {
        "derived_quantity",
        "model_dependent_estimate",
        "inverse_problem_or_model_inference",
        "reference_only",
    }
    assert len(FORMULAS) > 0
    assert all(item["inference_class"] in allowed for item in FORMULAS)
    assert all(item["status"] == item["inference_class"] for item in FORMULAS)
    assert all(item["validation_status"] for item in FORMULAS)
    assert any(item["inference_class"] == "inverse_problem_or_model_inference"
               for item in FORMULAS)


def test_bundled_registries_are_valid_and_duplicate_free():
    diag_path = files("tis.data").joinpath("DIAGNOSTIC_REGISTRY.json")
    formula_path = files("tis.data").joinpath("FORMULA_REGISTRY.json")
    diagnostics = json.loads(diag_path.read_text(encoding="utf-8"))
    formulas = json.loads(formula_path.read_text(encoding="utf-8"))
    assert len({x["diagnostic"].strip().casefold() for x in diagnostics}) == len(diagnostics)
    assert len({x["id"] for x in diagnostics}) == len(diagnostics)
    assert len({x["id"] for x in formulas}) == len(formulas)
    assert all("inference_class" in x and "validation_status" in x for x in formulas)
    # Root registries, bundled package resources and in-app catalogs must not drift.
    from tis.catalog import DIAGNOSTICS, FORMULAS
    assert {x["id"] for x in diagnostics} == {x["id"] for x in DIAGNOSTICS}
    assert {x["id"] for x in formulas} == {x["id"] for x in FORMULAS}
    assert {x["diagnostic"] for x in diagnostics} == {x["diagnostic"] for x in DIAGNOSTICS}
    assert {x["name"] for x in formulas} == {x["name"] for x in FORMULAS}


def test_in_app_catalog_ids_match_bundled_registry_ids():
    diagnostics = json.loads(files("tis.data").joinpath("DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    formulas = json.loads(files("tis.data").joinpath("FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    assert {x["id"] for x in diagnostics} == {x["id"] for x in DIAGNOSTICS}
    assert {x["id"] for x in formulas} == {x["id"] for x in FORMULAS}
