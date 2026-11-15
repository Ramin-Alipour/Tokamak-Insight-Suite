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


def test_shared_cts_seed_matches_the_physical_seed_equations():
    f0, frequency, geometry, species, fractions, spectrum = _case()
    result = estimate_cts_optimizer_initial_parameters(
        frequency,
        spectrum,
        probe_frequency_Hz=f0,
        scattering_geometry=geometry,
        ion_species=species,
        ion_fractions=fractions,
    )

    baseline = np.quantile(spectrum, 0.05)
    weights = np.maximum(spectrum - baseline, 0.0)
    total = np.sum(weights)
    centroid = np.sum(weights * frequency) / total
    offsets = frequency - centroid
    variance = np.sum(weights * offsets**2) / total
    angle = geometry["scattering_angle_rad"]
    q = 4.0 * np.pi * f0 / 299792458.0 * np.sin(angle / 2.0)
    electron_velocity_rms = 2.0 * np.pi * np.sqrt(variance) / q
    expected_te = 9.1093837015e-31 * electron_velocity_rms**2 / 1.602176634e-19

    order = np.argsort(np.abs(offsets))
    cumulative = np.cumsum(weights[order]) / total
    selected = order[cumulative <= 0.68]
    if selected.size < 3:
        selected = order[: min(max(3, int(np.ceil(0.25 * frequency.size))), frequency.size)]
    core_weights = weights[selected]
    core_variance = np.sum(core_weights * offsets[selected]**2) / np.sum(core_weights)
    ion_velocity_rms = 2.0 * np.pi * np.sqrt(max(core_variance, variance * 1e-6)) / q
    expected_ti = 2.014 * 1.66053906660e-27 * ion_velocity_rms**2 / 1.602176634e-19
    expected_ne_seed = 8.8541878128e-12 * expected_te * q**2 / 1.602176634e-19

    assert result["electron_temperature_eV"] == pytest.approx(expected_te, rel=2e-13)
    assert result["ion_temperature_eV"] == pytest.approx(expected_ti, rel=2e-13)
    assert result["electron_density_m3"] == pytest.approx(expected_ne_seed, rel=2e-13)
    assert result["spectral_centroid_Hz"] == pytest.approx(centroid, rel=2e-13)
    assert result["spectral_rms_width_Hz"] == pytest.approx(np.sqrt(variance), rel=2e-13)
    assert result["role"] == "optimizer_initial_estimates_only"
    assert result["scattering_wave_number_rad_m"] == pytest.approx(q, rel=2e-13)
    assert result["effective_ion_mass_amu"] == pytest.approx(2.014)


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
