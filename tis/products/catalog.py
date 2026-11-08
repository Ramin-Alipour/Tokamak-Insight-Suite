"""Canonical product catalog built from the existing 20-family registry."""
from __future__ import annotations

from copy import deepcopy

from tokamak_analysis.analysis_registry import (
    ANALYSIS_FAMILIES,
    ANALYSES,
    get_analysis,
)

_PRODUCT_SLUGS = {
    1: "thomson-scattering",
    2: "langmuir-electrostatic-probes",
    3: "electron-cyclotron-emission",
    4: "doppler-charge-exchange-spectroscopy",
    5: "molecular-spectroscopy",
    6: "magnetic-diagnostics-equilibrium",
    7: "emission-spectroscopy",
    8: "interferometry-reflectometry",
    9: "bolometry-radiation-tomography",
    10: "imaging-ir-thermography",
    11: "neutron-diagnostics",
    12: "gamma-ray-spectroscopy",
    13: "fast-particle-diagnostics",
    14: "gas-pressure-fueling",
    15: "fluctuation-mhd-transient-analysis",
    16: "transport-confinement-plasma-parameters",
    17: "heating-power-analysis",
    18: "plasma-facing-components-thermal-analysis",
    19: "x-ray-diagnostics",
    20: "instrumentation-measurement-support",
}

_PRODUCT_API_MODULES = {
    1: "tis.products.thomson",
    2: "tis.products.langmuir",
    3: "tis.products.electron_cyclotron_emission",
    4: "tis.products.doppler_charge_exchange_spectroscopy",
    5: "tis.products.molecular_spectroscopy",
    6: "tis.products.magnetic_diagnostics_equilibrium",
    7: "tis.products.emission_spectroscopy",
    8: "tis.products.interferometry_reflectometry",
    9: "tis.products.bolometry_radiation_tomography",
    10: "tis.products.imaging_ir_thermography",
    11: "tis.products.neutron_diagnostics",
    12: "tis.products.gamma_ray_spectroscopy",
    13: "tis.products.fast_particle_diagnostics",
    14: "tis.products.gas_pressure_fueling",
    15: "tis.products.fluctuation_mhd_transient_analysis",
    16: "tis.products.transport_confinement_plasma_parameters",
    17: "tis.products.heating_power_analysis",
    18: "tis.products.plasma_facing_components_thermal_analysis",
    19: "tis.products.x_ray_diagnostics",
    20: "tis.products.instrumentation_measurement_support",
}


def _build_product(family: dict) -> dict:
    """Create a product descriptor without duplicating physics metadata."""
    product = deepcopy(family)
    product["product_id"] = _PRODUCT_SLUGS[family["id"]]
    product["product_version"] = "0.1.0"
    product["product_status"] = "modular_monolith"
    product["api_module"] = _PRODUCT_API_MODULES[family["id"]]
    product["entry_points"] = [get_analysis(entry_id) for entry_id in family["entry_ids"]]
    product["execution_entry_point_count"] = len(product["entry_points"])
    return product


def list_products() -> list[dict]:
    """Return all specialist products in canonical navigation order."""
    return [_build_product(family) for family in ANALYSIS_FAMILIES]


def get_product(product_id: int | str) -> dict:
    """Return one product by primary family ID or stable product slug."""
    if isinstance(product_id, bool):
        raise ValueError("product_id must be a family integer or product slug")
    if isinstance(product_id, int):
        family = next((item for item in ANALYSIS_FAMILIES if item["id"] == product_id), None)
    elif isinstance(product_id, str):
        family = next(
            (item for item in ANALYSIS_FAMILIES
             if _PRODUCT_SLUGS[item["id"]] == product_id),
            None,
        )
    else:
        raise ValueError("product_id must be a family integer or product slug")
    if family is None:
        raise KeyError(f"Unknown specialist product: {product_id}")
    return _build_product(family)


def list_product_analyses(product_id: int | str) -> list[dict]:
    """Return the registered callable entry points owned by one product."""
    return [deepcopy(item) for item in get_product(product_id)["entry_points"]]


def product_for_analysis(analysis_id: int) -> dict:
    """Resolve the unique owning product for a registered analysis entry point."""
    if isinstance(analysis_id, bool) or not isinstance(analysis_id, int):
        raise ValueError("analysis_id must be an integer")
    matches = [
        family for family in ANALYSIS_FAMILIES
        if analysis_id in family.get("entry_ids", [])
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Analysis ID {analysis_id} must belong to exactly one product; found {len(matches)}"
        )
    return get_product(matches[0]["id"])


def validate_product_catalog() -> dict:
    """Return a deterministic reconciliation of products and executable entries."""
    family_ids = [item["id"] for item in ANALYSIS_FAMILIES]
    mapped_ids = [
        entry_id for family in ANALYSIS_FAMILIES
        for entry_id in family.get("entry_ids", [])
    ]
    registry_ids = [entry["id"] for entry in ANALYSES]
    return {
        "product_count": len(ANALYSIS_FAMILIES),
        "product_ids_ordered": family_ids == list(range(1, 21)),
        "analysis_entry_point_count": len(ANALYSES),
        "analysis_ids_unique": len(registry_ids) == len(set(registry_ids)),
        "every_analysis_has_one_product_owner": (
            sorted(mapped_ids) == sorted(registry_ids)
            and len(mapped_ids) == len(set(mapped_ids))
        ),
        "product_slugs_unique": len(set(_PRODUCT_SLUGS.values())) == 20,
        "product_api_modules_unique": len(set(_PRODUCT_API_MODULES.values())) == 20,
        "instrumentation_support_is_cross_cutting": (
            ANALYSIS_FAMILIES[19].get("category") == "instrumentation_support"
        ),
    }
