import numpy as np
import pytest

from tokamak_analysis.radiation_tomography import abel_project_axisymmetric_emissivity


@pytest.mark.parametrize("impact", [0.0, 0.5, 0.9])
def test_abel_projection_matches_analytic_constant_emissivity(impact):
    radius = np.linspace(0.0, 1.0, 101)
    emissivity = np.full_like(radius, 3.0)

    result = abel_project_axisymmetric_emissivity(
        radius, emissivity, np.array([impact])
    )

    expected = 2.0 * 3.0 * np.sqrt(1.0 - impact**2)
    assert result["line_integrated_emissivity_W_m2"][0] == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )


def test_abel_projection_handles_chords_between_radial_grid_points():
    radius = np.linspace(0.0, 1.0, 101)
    emissivity = np.full_like(radius, 2.5)
    impact = 0.537

    result = abel_project_axisymmetric_emissivity(
        radius, emissivity, np.array([impact])
    )

    expected = 2.0 * 2.5 * np.sqrt(1.0 - impact**2)
    assert result["line_integrated_emissivity_W_m2"][0] == pytest.approx(
        expected, rel=1e-12, abs=1e-12
    )


def test_abel_projection_rejects_out_of_domain_impact_parameter():
    with pytest.raises(ValueError, match="impact parameters"):
        abel_project_axisymmetric_emissivity(
            np.linspace(0.0, 1.0, 5), np.ones(5), np.array([1.0])
        )
