import numpy as np
import pytest

import tokamak_analysis.cxrs as cxrs_module
from tokamak_analysis.cxrs import analyze_cxrs_line


def synthetic():
    x = np.linspace(529.8, 530.2, 401)
    center = 530.0 * (1 + 2.5e4 / 299792458.0)
    sig_intrinsic = 0.025
    sig_inst = 0.008
    sig_obs = np.hypot(sig_intrinsic, sig_inst)
    y = 120 * np.exp(-0.5 * ((x-center)/sig_obs)**2) + 4 + 0.2*(x-center)
    return x, y, center, sig_intrinsic


def test_recovers_cxrs_center_velocity_and_temperature():
    x, y, center, sig_intrinsic = synthetic()
    result = analyze_cxrs_line(x, y, rest_wavelength_nm=530.0, ion_mass_amu=12,
                               instrument_sigma_nm=0.008)
    assert result.wavelength_center_nm == pytest.approx(center, abs=2e-5)
    assert result.line_of_sight_velocity_m_s == pytest.approx(2.5e4, abs=15)
    expected_ti = 12 * 1.66053906660e-27 * 299792458.0**2 * (sig_intrinsic/530.0)**2 / 1.602176634e-19
    assert result.ion_temperature_eV == pytest.approx(expected_ti, rel=0.02)
    assert len(result.residual) == len(x)
    assert result.fit_r_squared > 0.999


def test_weighted_fit_reports_reduced_chi_square():
    x, y, *_ = synthetic()
    result = analyze_cxrs_line(x, y, rest_wavelength_nm=530, ion_mass_amu=12,
                               instrument_sigma_nm=0.008, signal_uncertainty=np.ones_like(y))
    assert result.reduced_chi_square is not None
    assert result.degrees_of_freedom == len(x) - 5


@pytest.mark.parametrize("kwargs", [
    {"rest_wavelength_nm": 0, "ion_mass_amu": 12, "instrument_sigma_nm": .01},
    {"rest_wavelength_nm": 530, "ion_mass_amu": 0, "instrument_sigma_nm": .01},
    {"rest_wavelength_nm": 530, "ion_mass_amu": 12, "instrument_sigma_nm": 0},
])
def test_rejects_nonpositive_physical_parameters(kwargs):
    x, y, *_ = synthetic()
    with pytest.raises(ValueError):
        analyze_cxrs_line(x, y, **kwargs)


def test_rejects_bad_axis_and_uncertainty():
    x, y, *_ = synthetic()
    x[10] = x[9]
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze_cxrs_line(x, y, rest_wavelength_nm=530, ion_mass_amu=12, instrument_sigma_nm=.008)
    x, y, *_ = synthetic()
    with pytest.raises(ValueError, match="signal_uncertainty"):
        analyze_cxrs_line(x, y, rest_wavelength_nm=530, ion_mass_amu=12,
                          instrument_sigma_nm=.008, signal_uncertainty=np.zeros_like(y))


def test_instrument_width_uncertainty_is_propagated_to_temperature():
    x, y, *_ = synthetic()
    base = analyze_cxrs_line(x, y, rest_wavelength_nm=530, ion_mass_amu=12,
                             instrument_sigma_nm=0.008)
    with_instrument_uncertainty = analyze_cxrs_line(
        x, y, rest_wavelength_nm=530, ion_mass_amu=12,
        instrument_sigma_nm=0.008, instrument_sigma_uncertainty_nm=0.001)
    assert with_instrument_uncertainty.instrument_sigma_uncertainty_nm == pytest.approx(0.001)
    assert with_instrument_uncertainty.ion_temperature_uncertainty_eV > base.ion_temperature_uncertainty_eV


def test_rejects_invalid_instrument_width_uncertainty():
    x, y, *_ = synthetic()
    for uncertainty in (-0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="instrument_sigma_uncertainty_nm"):
            analyze_cxrs_line(x, y, rest_wavelength_nm=530, ion_mass_amu=12,
                              instrument_sigma_nm=0.008,
                              instrument_sigma_uncertainty_nm=uncertainty)


def test_rejects_unresolved_instrument_width():
    x, y, *_ = synthetic()
    with pytest.raises(ValueError, match="unresolved"):
        analyze_cxrs_line(x, y, rest_wavelength_nm=530, ion_mass_amu=12, instrument_sigma_nm=.2)


def test_rejects_rest_wavelength_outside_measured_window():
    x, y, *_ = synthetic()
    with pytest.raises(ValueError, match="within the measured wavelength range"):
        analyze_cxrs_line(x, y, rest_wavelength_nm=600.0, ion_mass_amu=12,
                          instrument_sigma_nm=0.008)



def test_viewing_angle_is_preserved_as_metadata_without_velocity_deprojection():
    x, y, center, _ = synthetic()
    angle = np.pi / 3
    result = analyze_cxrs_line(
        x, y, rest_wavelength_nm=530.0, ion_mass_amu=12,
        instrument_sigma_nm=0.008, viewing_angle_rad=angle,
    )
    assert result.viewing_angle_rad == pytest.approx(angle)
    # The API reports line-of-sight velocity; angle metadata must not silently
    # transform it into a toroidal/poloidal component.
    assert result.line_of_sight_velocity_m_s == pytest.approx(
        299792458.0 * (center - 530.0) / 530.0, abs=15
    )


def test_rejects_optimizer_result_with_unidentifiable_covariance(monkeypatch):
    x, y, *_ = synthetic()

    def fake_curve_fit(*args, **kwargs):
        return np.array([120.0, 530.0, 0.026, 4.0, 0.2]), np.full((5, 5), np.nan)

    monkeypatch.setattr(cxrs_module, "curve_fit", fake_curve_fit)
    with pytest.raises(ValueError, match="covariance is singular or non-finite"):
        analyze_cxrs_line(
            x, y, rest_wavelength_nm=530.0, ion_mass_amu=12,
            instrument_sigma_nm=0.008,
        )

def test_rejects_finite_but_rank_deficient_covariance(monkeypatch):
    x, y, *_ = synthetic()

    def fake_curve_fit(*args, **kwargs):
        return np.array([120.0, 530.0, 0.026, 4.0, 0.2]), np.zeros((5, 5))

    monkeypatch.setattr(cxrs_module, "curve_fit", fake_curve_fit)
    with pytest.raises(ValueError, match="covariance is rank-deficient"):
        analyze_cxrs_line(
            x, y, rest_wavelength_nm=530.0, ion_mass_amu=12,
            instrument_sigma_nm=0.008,
        )


def test_rejects_under_resolved_line_profile():
    # The intrinsic width is not meaningfully sampled on this coarse axis.
    x = np.linspace(529.8, 530.2, 15)
    center = 530.0 * (1 + 2.5e4 / 299792458.0)
    sigma_observed = np.hypot(0.025, 0.008)
    y = 120 * np.exp(-0.5 * ((x-center)/sigma_observed)**2) + 4
    with pytest.raises(ValueError, match="under-resolved by wavelength sampling"):
        analyze_cxrs_line(
            x, y, rest_wavelength_nm=530.0, ion_mass_amu=12,
            instrument_sigma_nm=0.008,
        )


def test_rest_wavelength_uncertainty_is_propagated_to_temperature_and_velocity():
    x, y, *_ = synthetic()
    base = analyze_cxrs_line(
        x, y, rest_wavelength_nm=530.0, ion_mass_amu=12,
        instrument_sigma_nm=0.008,
    )
    with_rest_uncertainty = analyze_cxrs_line(
        x, y, rest_wavelength_nm=530.0, ion_mass_amu=12,
        instrument_sigma_nm=0.008, rest_wavelength_uncertainty_nm=1e-5,
    )
    assert with_rest_uncertainty.rest_wavelength_uncertainty_nm == pytest.approx(1e-5)
    assert with_rest_uncertainty.ion_temperature_uncertainty_eV > base.ion_temperature_uncertainty_eV
    assert with_rest_uncertainty.velocity_uncertainty_m_s > base.velocity_uncertainty_m_s


def test_rejects_invalid_rest_wavelength_uncertainty():
    x, y, *_ = synthetic()
    for uncertainty in (-1e-5, np.nan, np.inf):
        with pytest.raises(ValueError, match="rest_wavelength_uncertainty_nm"):
            analyze_cxrs_line(
                x, y, rest_wavelength_nm=530.0, ion_mass_amu=12,
                instrument_sigma_nm=0.008,
                rest_wavelength_uncertainty_nm=uncertainty,
            )
