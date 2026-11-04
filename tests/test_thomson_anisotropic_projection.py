import numpy as np
import pytest

from tokamak_analysis.thomson_kinetic_models import (
    project_bi_maxwellian_electron_vdf,
    infer_gyrotropic_bi_maxwellian_temperatures,
)


def test_bi_maxwellian_projected_temperature_has_correct_parallel_and_perpendicular_limits():
    velocity = np.linspace(-1.2e8, 1.2e8, 40001)
    parallel = project_bi_maxwellian_electron_vdf(
        velocity, temperature_parallel_eV=800.0,
        temperature_perpendicular_eV=200.0,
        magnetic_field_unit=[1.0, 0.0, 0.0],
        scattering_vector_unit=[1.0, 0.0, 0.0],
    )
    perpendicular = project_bi_maxwellian_electron_vdf(
        velocity, temperature_parallel_eV=800.0,
        temperature_perpendicular_eV=200.0,
        magnetic_field_unit=[1.0, 0.0, 0.0],
        scattering_vector_unit=[0.0, 1.0, 0.0],
    )
    assert parallel["projected_temperature_eV"] == pytest.approx(800.0)
    assert perpendicular["projected_temperature_eV"] == pytest.approx(200.0)
    assert parallel["projected_velocity_standard_deviation_m_s"] > perpendicular["projected_velocity_standard_deviation_m_s"]


def test_bi_maxwellian_projection_reports_single_angle_nonidentifiability_and_normalizes():
    velocity = np.linspace(-8.0e7, 8.0e7, 40001)
    result = project_bi_maxwellian_electron_vdf(
        velocity, temperature_parallel_eV=500.0,
        temperature_perpendicular_eV=100.0,
        magnetic_field_unit=[0.0, 0.0, 1.0],
        scattering_vector_unit=[2**-0.5, 0.0, 2**-0.5],
        flow_projection_m_s=1.2e5,
    )
    assert np.trapz(result["projected_vdf_s_m"], velocity) == pytest.approx(result["velocity_grid_probability_coverage"], rel=2e-6)
    assert result["velocity_grid_probability_coverage"] > 0.999999
    assert result["projected_temperature_eV"] == pytest.approx(300.0)
    assert result["projected_velocity_mean_m_s"] == pytest.approx(1.2e5)
    assert result["separate_parallel_perpendicular_temperatures_identifiable"] is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature_parallel_eV": 0.0},
        {"temperature_perpendicular_eV": -1.0},
    ],
)
def test_bi_maxwellian_projection_rejects_nonpositive_temperatures(kwargs):
    values = dict(
        temperature_parallel_eV=100.0,
        temperature_perpendicular_eV=100.0,
        magnetic_field_unit=[1.0, 0.0, 0.0],
        scattering_vector_unit=[0.0, 1.0, 0.0],
    )
    values.update(kwargs)
    with pytest.raises(ValueError):
        project_bi_maxwellian_electron_vdf(np.linspace(-1e7, 1e7, 101), **values)


def test_bi_maxwellian_projection_rejects_nonunit_geometry():
    with pytest.raises(ValueError, match="unit norm"):
        project_bi_maxwellian_electron_vdf(
            np.linspace(-1e7, 1e7, 101),
            temperature_parallel_eV=100.0,
            temperature_perpendicular_eV=100.0,
            magnetic_field_unit=[2.0, 0.0, 0.0],
            scattering_vector_unit=[0.0, 1.0, 0.0],
        )


def test_multi_pitch_anisotropy_inversion_propagates_pitch_geometry_covariance():
    common = dict(
        projected_temperature_eV=[200.0, 125.0, 100.0],
        magnetic_pitch_cosine=[1.0, 0.5, 0.0],
        projected_temperature_covariance_eV2=np.diag([4.0, 4.0, 4.0]),
    )
    measurement_only = infer_gyrotropic_bi_maxwellian_temperatures(**common)
    with_geometry = infer_gyrotropic_bi_maxwellian_temperatures(
        **common,
        magnetic_pitch_cosine_covariance=np.diag([1.0e-4, 1.0e-4, 1.0e-4]),
    )
    assert with_geometry["reportable_as_inferred_parameter"] is True
    assert with_geometry["geometry_uncertainty_included"] is True
    assert with_geometry["geometry_covariance_method"] == "iterative_first_order_errors_in_variables_GLS"
    assert with_geometry["electron_temperature_parallel_eV"] == pytest.approx(200.0, rel=2e-3)
    assert with_geometry["electron_temperature_perpendicular_eV"] == pytest.approx(100.0, rel=2e-3)
    assert np.trace(with_geometry["temperature_covariance_eV2"]) > np.trace(
        measurement_only["temperature_covariance_eV2"]
    )


def test_multi_pitch_anisotropy_rejects_non_psd_pitch_covariance():
    with pytest.raises(ValueError, match="positive semidefinite"):
        infer_gyrotropic_bi_maxwellian_temperatures(
            [200.0, 150.0, 100.0], [1.0, 0.5, 0.0],
            projected_temperature_covariance_eV2=np.diag([4.0, 4.0, 4.0]),
            magnetic_pitch_cosine_covariance=np.diag([1.0e-4, -1.0e-4, 1.0e-4]),
        )
