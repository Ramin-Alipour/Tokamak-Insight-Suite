import numpy as np
import pytest
from tis.line_shape import instrument_line_shape_workflow


def test_gaussian_reference_line_width_and_resolving_power():
    x = np.linspace(654.0, 658.0, 401)
    center, sigma = 656.3, 0.12
    y = 3.0 + 10.0 * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    result = instrument_line_shape_workflow(
        x, y, reference_wavelength_nm=center, model="Gaussian", minimum_r_squared=0.99
    )
    expected_fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
    assert result["observed_fwhm_nm"] == pytest.approx(expected_fwhm, rel=0.01)
    assert result["nominal_resolving_power"] == pytest.approx(center / expected_fwhm, rel=0.01)
    assert result["fit_quality_status"] == "pass"
    assert result["standard_error_observed_fwhm_nm"] >= 0


def test_line_shape_does_not_claim_automatic_deconvolution():
    x = np.linspace(-2, 2, 301)
    y = 2 + 4 / (1 + ((x - 0.3) / 0.2) ** 2)
    result = instrument_line_shape_workflow(x, y, reference_wavelength_nm=0.3, model="Lorentzian")
    assert result["observed_fwhm_nm"] == pytest.approx(0.4, rel=0.02)
    assert "does not" in result["interpretation"]
    assert result["fit_quality_status"] == "not_assessed"


@pytest.mark.parametrize("kwargs", [
    {"reference_wavelength_nm": 0},
    {"reference_wavelength_nm": 656, "minimum_r_squared": 1.1},
    {"reference_wavelength_nm": 656, "fit_min_nm": 3, "fit_max_nm": 2},
])
def test_line_shape_rejects_invalid_options(kwargs):
    x = np.linspace(654, 658, 51)
    y = 1 + np.exp(-((x - 656) / 0.2) ** 2)
    with pytest.raises(ValueError):
        instrument_line_shape_workflow(x, y, **kwargs)
