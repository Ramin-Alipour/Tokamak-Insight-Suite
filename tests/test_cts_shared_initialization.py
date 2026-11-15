from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from tis.physics.shared.cts_initialization import (
    estimate_cts_optimizer_initial_parameters,
)


def _case():
    probe_frequency = 140.0e9
    frequency = np.linspace(-7.0e9, 7.0e9, 521)
    geometry = {"scattering_angle_rad": np.deg2rad(110.0)}
    species = [{"name": "D+", "charge_state": 1, "mass_amu": 2.014}]
    fractions = [1.0]
    spectrum = (
        0.03
        + np.exp(-0.5 * ((frequency - 1.7e8) / 1.1e9) ** 2)
        + 0.22 * np.exp(-0.5 * ((frequency - 1.7e8) / 1.7e8) ** 2)
    )
    return probe_frequency, frequency, geometry, species, fractions, spectrum


def test_shared_cts_seed_matches_legacy_physical_initializer():
    from tokamak_analysis.cts_analysis import (
        estimate_cts_optimizer_initial_parameters as legacy_estimator,
    )

    f0, frequency, geometry, species, fractions, spectrum = _case()
    actual = estimate_cts_optimizer_initial_parameters(
        frequency,
        spectrum,
        probe_frequency_Hz=f0,
        scattering_geometry=geometry,
        ion_species=species,
        ion_fractions=fractions,
    )
    legacy = legacy_estimator(
        frequency,
        spectrum,
        probe_frequency_Hz=f0,
        scattering_geometry=geometry,
        ion_species=species,
        ion_fractions=fractions,
    )
    for key in (
        "electron_density_m3",
        "electron_temperature_eV",
        "ion_temperature_eV",
        "spectral_centroid_Hz",
        "spectral_rms_width_Hz",
        "spectral_core_rms_width_Hz",
    ):
        assert actual[key] == pytest.approx(legacy[key], rel=2e-13, abs=0.0)
    assert actual["role"] == "optimizer_initial_estimates_only"
    assert actual["scattering_wave_number_rad_m"] > 0.0
    assert actual["effective_ion_mass_amu"] == pytest.approx(2.014)


def test_product_cts_seed_wrapper_uses_the_shared_initializer():
    from tis.products.thomson.physics.cts_analysis import (
        estimate_cts_optimizer_initial_parameters as product_estimator,
    )

    f0, frequency, geometry, species, fractions, spectrum = _case()
    shared = estimate_cts_optimizer_initial_parameters(
        frequency,
        spectrum,
        probe_frequency_Hz=f0,
        scattering_geometry=geometry,
        ion_species=species,
        ion_fractions=fractions,
    )
    product = product_estimator(
        frequency,
        spectrum,
        probe_frequency_Hz=f0,
        scattering_geometry=geometry,
        ion_species=species,
        ion_fractions=fractions,
    )
    assert product == shared


def test_shared_cts_seed_rejects_unresolved_frequency_dependent_geometry():
    f0, frequency, _, species, fractions, spectrum = _case()
    geometry = {
        "frequency_resolved_geometry": [
            {"scattering_angle_rad": np.deg2rad(angle)}
            for angle in (65.0, 110.0, 145.0)
        ]
    }
    with pytest.raises(ValueError, match="one fixed scattering vector"):
        estimate_cts_optimizer_initial_parameters(
            frequency,
            spectrum,
            probe_frequency_Hz=f0,
            scattering_geometry=geometry,
            ion_species=species,
            ion_fractions=fractions,
        )


def test_shared_cts_seed_rejects_spectrum_without_positive_contrast():
    f0, frequency, geometry, species, fractions, _ = _case()
    with pytest.raises(ValueError, match="no positive contrast"):
        estimate_cts_optimizer_initial_parameters(
            frequency,
            np.ones_like(frequency),
            probe_frequency_Hz=f0,
            scattering_geometry=geometry,
            ion_species=species,
            ion_fractions=fractions,
        )


def test_full_em_solver_no_longer_imports_legacy_cts_initialization():
    root = Path(__file__).resolve().parents[1]
    source = (root / "tis" / "physics" / "shared" / "cts_full_em.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert "cts_initialization" in imports
    assert "tokamak_analysis.cts_analysis" not in imports
