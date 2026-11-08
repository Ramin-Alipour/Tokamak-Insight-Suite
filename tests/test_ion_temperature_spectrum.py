from __future__ import annotations

import numpy as np
import pytest

from tis.doppler_temperature import doppler_fwhm_nm_for_temperature
from tokamak_analysis.ion_temperature import ion_temperature_from_doppler_spectrum


def _synthetic_spectrum(ti_eV=35.0, seed=41):
    rng = np.random.default_rng(seed)
    rest = 464.74
    mass = 12.0
    instrument = 0.039
    thermal = doppler_fwhm_nm_for_temperature(rest, mass, ti_eV)
    observed = np.sqrt(thermal**2 + instrument**2)
    x = np.linspace(rest - 0.16, rest + 0.16, 161)
    sigma = observed / 2.354820045
    y = 12.0 + 0.4 * (x - rest) + 100.0 * np.exp(-0.5 * ((x - (rest + 0.006)) / sigma)**2)
    y += rng.normal(0.0, 0.15, x.size)
    return x, y, rest, mass, instrument


def test_spectrum_workflow_recovers_temperature_with_instrument_correction():
    x, y, rest, mass, instrument = _synthetic_spectrum()
    result = ion_temperature_from_doppler_spectrum(
        x, y, rest_wavelength_nm=rest, ion_mass_amu=mass,
        instrumental_fwhm_nm=instrument,
        instrumental_fwhm_uncertainty_nm=0.0005,
    )
    assert result["ion_temperature_eV"] == pytest.approx(35.0, rel=0.12)
    assert result["thermal_doppler_fwhm_nm"] > 0
    assert result["observed_fwhm_nm"] > instrument
    assert result["fit_r_squared"] > 0.99
    assert result["fitted_intensity"].shape == y.shape
    assert result["fit_residuals"].shape == y.shape
    assert result["ion_temperature_uncertainty_eV"] is not None


def test_spectrum_workflow_rejects_invalid_axis_and_spectrum():
    x, y, rest, mass, instrument = _synthetic_spectrum()
    with pytest.raises(ValueError, match="increasing wavelengths"):
        ion_temperature_from_doppler_spectrum(
            x[::-1], y, rest_wavelength_nm=rest, ion_mass_amu=mass,
            instrumental_fwhm_nm=instrument,
        )
    with pytest.raises(ValueError, match="instrumental_fwhm_nm"):
        ion_temperature_from_doppler_spectrum(
            x, y, rest_wavelength_nm=rest, ion_mass_amu=mass,
            instrumental_fwhm_nm=float("nan"),
        )


def test_spectrum_workflow_rejects_unresolved_instrument_width():
    x, y, rest, mass, _ = _synthetic_spectrum(ti_eV=1.0)
    with pytest.raises(ValueError, match="unresolved"):
        ion_temperature_from_doppler_spectrum(
            x, y, rest_wavelength_nm=rest, ion_mass_amu=mass,
            instrumental_fwhm_nm=0.2,
        )


def test_spectrum_workflow_validates_intensity_uncertainty():
    x, y, rest, mass, instrument = _synthetic_spectrum()
    with pytest.raises(ValueError, match="intensity_uncertainty"):
        ion_temperature_from_doppler_spectrum(
            x, y, rest_wavelength_nm=rest, ion_mass_amu=mass,
            instrumental_fwhm_nm=instrument,
            intensity_uncertainty=np.ones(x.size - 1),
        )


def test_spectrum_workflow_rejects_rest_wavelength_outside_window():
    x, y, rest, mass, instrument = _synthetic_spectrum()
    with pytest.raises(ValueError, match="within the measured wavelength window"):
        ion_temperature_from_doppler_spectrum(
            x, y, rest_wavelength_nm=x[-1] + 0.01, ion_mass_amu=mass,
            instrumental_fwhm_nm=instrument,
        )


def test_spectrum_workflow_rejects_peak_at_window_edge():
    x, y, rest, mass, instrument = _synthetic_spectrum()
    y = y.copy()
    y[0] = y.max() * 2
    with pytest.raises(ValueError, match="peak is at a wavelength-window boundary"):
        ion_temperature_from_doppler_spectrum(
            x, y, rest_wavelength_nm=rest, ion_mass_amu=mass,
            instrumental_fwhm_nm=instrument,
        )


def test_spectrum_workflow_rejects_flat_spectrum():
    x, _, rest, mass, instrument = _synthetic_spectrum()
    y = np.full_like(x, 4.0)
    with pytest.raises(ValueError, match="no positive emission-line contrast"):
        ion_temperature_from_doppler_spectrum(
            x, y, rest_wavelength_nm=rest, ion_mass_amu=mass,
            instrumental_fwhm_nm=instrument,
        )


def test_spectrum_workflow_reports_residual_and_weighted_fit_diagnostics():
    x, y, rest, mass, instrument = _synthetic_spectrum()
    uncertainty = np.full(y.shape, 0.15)
    result = ion_temperature_from_doppler_spectrum(
        x, y, rest_wavelength_nm=rest, ion_mass_amu=mass,
        instrumental_fwhm_nm=instrument,
        instrumental_fwhm_uncertainty_nm=0.0005,
        intensity_uncertainty=uncertainty,
    )
    assert result["fit_residual_sum_squares"] == pytest.approx(
        np.sum(result["fit_residuals"] ** 2)
    )
    assert result["fit_residual_rms"] == pytest.approx(
        np.sqrt(np.mean(result["fit_residuals"] ** 2))
    )
    assert result["fit_degrees_of_freedom"] == len(x) - 5
    assert result["fit_reduced_chi_square"] is not None
    assert np.isfinite(result["fit_reduced_chi_square"])
    unweighted = ion_temperature_from_doppler_spectrum(
        x, y, rest_wavelength_nm=rest, ion_mass_amu=mass,
        instrumental_fwhm_nm=instrument,
    )
    assert unweighted["fit_reduced_chi_square"] is None


def test_fit_uncertainty_is_reported_when_instrument_width_is_treated_as_exact():
    x, y, rest, mass, instrument = _synthetic_spectrum()
    result = ion_temperature_from_doppler_spectrum(
        x, y, rest_wavelength_nm=rest, ion_mass_amu=mass,
        instrumental_fwhm_nm=instrument,
    )
    assert result["observed_fwhm_uncertainty_nm"] is not None
    assert result["ion_temperature_uncertainty_eV"] is not None
    assert np.isfinite(result["ion_temperature_uncertainty_eV"])
    assert result["ion_temperature_uncertainty_eV"] > 0
    assert any("treated as exact" in item for item in result["assumptions"])
