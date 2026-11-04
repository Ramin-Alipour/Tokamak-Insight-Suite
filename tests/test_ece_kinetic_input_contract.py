"""Strict physical-input contract tests for kinetic ECE entry points."""
import numpy as np
import pytest

from tokamak_analysis.ece_kinetic import (
    kinetic_ece_absorption_coefficient,
    relativistic_kinetic_ece_absorption_coefficient,
)


def _classical(**overrides):
    values = dict(
        frequency_Hz=100e9,
        wavevector_rad_m=np.array([10.0, 0.0, 0.0]),
        magnetic_field_T=np.array([0.0, 0.0, 2.0]),
        electron_density_m3=1e19,
        electron_temperature_eV=1e3,
        polarization=np.array([1.0, 1.0j, 0.0]),
        quadrature_order=16,
    )
    values.update(overrides)
    return values


def _relativistic(**overrides):
    values = dict(
        frequency_Hz=100e9,
        wavevector_rad_m=np.array([10.0, 0.0, 0.0]),
        magnetic_field_T=np.array([0.0, 0.0, 2.0]),
        electron_density_m3=1e19,
        electron_distribution={
            "kind": "relativistic_maxwell_juttner",
            "temperature_eV": 1e3,
        },
        polarization=np.array([1.0, 1.0j, 0.0]),
        quadrature_order=4,
        gyrophase_points=16,
    )
    values.update(overrides)
    return values


@pytest.mark.parametrize("field,value", [
    ("frequency_Hz", True),
    ("frequency_Hz", "100e9"),
    ("electron_density_m3", False),
    ("electron_density_m3", 1e19 + 0j),
    ("electron_temperature_eV", "1000"),
    ("damping_rate_rad_s", True),
])
def test_classical_ece_rejects_coerced_physical_scalars(field, value):
    with pytest.raises(ValueError):
        kinetic_ece_absorption_coefficient(**_classical(**{field: value}))


@pytest.mark.parametrize("field,value", [
    ("frequency_Hz", True),
    ("frequency_Hz", "100e9"),
    ("electron_density_m3", False),
    ("electron_density_m3", 1e19 + 0j),
    ("resonance_width_rad_s", True),
])
def test_relativistic_ece_rejects_coerced_physical_scalars(field, value):
    with pytest.raises(ValueError):
        relativistic_kinetic_ece_absorption_coefficient(
            **_relativistic(**{field: value})
        )


@pytest.mark.parametrize("field", ["wavevector_rad_m", "magnetic_field_T"])
@pytest.mark.parametrize("value", [
    [True, 0.0, 0.0],
    ["10", 0.0, 0.0],
    [10.0 + 0j, 0.0, 0.0],
])
def test_ece_rejects_non_real_or_coerced_geometry_vectors(field, value):
    with pytest.raises(ValueError):
        kinetic_ece_absorption_coefficient(**_classical(**{field: value}))
    with pytest.raises(ValueError):
        relativistic_kinetic_ece_absorption_coefficient(
            **_relativistic(**{field: value})
        )
