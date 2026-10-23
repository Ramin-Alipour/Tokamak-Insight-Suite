import numpy as np
import pytest

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


def test_rejects_unresolved_instrument_width():
    x, y, *_ = synthetic()
    with pytest.raises(ValueError, match="unresolved"):
        analyze_cxrs_line(x, y, rest_wavelength_nm=530, ion_mass_amu=12, instrument_sigma_nm=.2)
