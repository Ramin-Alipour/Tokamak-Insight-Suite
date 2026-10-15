import numpy as np
import pytest
from tis.workflows import wavelength_calibration_workflow


def test_linear_calibration_recovers_reference_axis():
    pixels = np.array([100, 250, 400, 550, 700], dtype=float)
    wavelengths = 400.0 + 0.025 * pixels
    result = wavelength_calibration_workflow(pixels, wavelengths, degree=1, max_rms_residual_nm=1e-10)
    assert result["rms_residual_nm"] < 1e-10
    assert result["acceptance_status"] == "pass"
    assert result["monotonic_over_calibration_span"]
    np.testing.assert_allclose(result["fitted_wavelength_nm"], wavelengths, atol=1e-10)


def test_residual_metrics_and_no_implicit_threshold():
    result = wavelength_calibration_workflow([10, 20, 30, 40], [500.0, 500.3, 500.61, 500.9])
    assert result["acceptance_status"] == "not_assessed"
    assert len(result["residuals_nm"]) == 4
    assert result["mean_absolute_residual_nm"] >= 0


def test_explicit_residual_threshold_can_fail():
    result = wavelength_calibration_workflow([0, 1, 2, 3], [400.0, 401.0, 402.2, 403.0], max_rms_residual_nm=1e-5)
    assert result["acceptance_status"] == "fail"


@pytest.mark.parametrize("pixels,wavelengths,kwargs", [
    ([0, 1], [400, 401], {"degree": 1}),
    ([0, 1, 1, 3], [400, 401, 402, 403], {"degree": 1}),
    ([0, 1, 2, 3], [400, 401, 401, 403], {"degree": 1}),
    ([0, 1, np.nan, 3], [400, 401, 402, 403], {"degree": 1}),
    ([0, 1, 2, 3], [400, 401, 402, 403], {"degree": 0}),
    ([0, 1, 2, 3], [400, 401, 402, 403], {"degree": 1, "max_rms_residual_nm": -1}),
])
def test_invalid_calibration_inputs_rejected(pixels, wavelengths, kwargs):
    with pytest.raises(ValueError):
        wavelength_calibration_workflow(pixels, wavelengths, **kwargs)


def test_nonmonotonic_polynomial_mapping_flagged():
    result = wavelength_calibration_workflow([0, 1, 2, 3, 4], [4, 1, 0, 1, 4], degree=2)
    assert not result["monotonic_over_calibration_span"]
