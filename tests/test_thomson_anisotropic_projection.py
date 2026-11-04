import numpy as np
import pytest

from tokamak_analysis.thomson_kinetic_models import project_bi_maxwellian_electron_vdf


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
