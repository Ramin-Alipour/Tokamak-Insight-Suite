import numpy as np
import pytest

from tokamak_analysis.molecular_spectroscopy import (
    analyze_molecular_spectrum,
    fit_custom_molecular_model,
    fit_molecular_boltzmann_temperature,
)


def test_molecular_spectrum_detects_features_and_integrates():
    x = np.linspace(400.0, 410.0, 1001)
    y = 2.0 + 10.0 * np.exp(-0.5 * ((x - 405.0) / 0.08) ** 2)
    out = analyze_molecular_spectrum(
        x, y, reference_bandheads=[(405.02, "band A")],
        band_ranges_nm=[(404.5, 405.5)],
    )
    assert out["peak_count"] == 1
    assert abs(out["peaks"][0]["wavelength_nm"] - 405.0) < 0.02
    assert out["integrated_signal_intensity_nm"] > 0
    assert out["reference_matches"][0]["reference_label"] == "band A"
    assert out["band_integrals"][0]["integrated_signal_intensity_nm"] > 0


def test_molecular_spectrum_rejects_bad_inputs():
    with pytest.raises(ValueError):
        analyze_molecular_spectrum([400, 402, 401, 403, 404], [1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        analyze_molecular_spectrum(np.arange(5) + 400, np.ones(5))


def test_custom_molecular_model_fit():
    x = np.linspace(400, 410, 100)
    y = 3.0 * np.exp(-0.5 * ((x - 405.0) / 0.3) ** 2) + 1.0
    def model(w, amplitude, center, sigma, baseline):
        return amplitude * np.exp(-0.5 * ((w - center) / sigma) ** 2) + baseline
    out = fit_custom_molecular_model(
        x, y, model, parameter_names=("amplitude", "center", "sigma", "baseline"),
        initial_values=(2.5, 404.8, 0.4, 0.8),
    )
    assert out["parameters"]["center"] == pytest.approx(405.0, abs=1e-3)
    assert out["parameters"]["amplitude"] == pytest.approx(3.0, rel=1e-3)


def test_custom_molecular_model_fit_reports_weighted_diagnostics():
    x = np.linspace(400.0, 410.0, 101)
    sigma_y = np.full_like(x, 0.05)
    def model(w, amplitude, center, width, baseline):
        return amplitude * np.exp(-0.5 * ((w - center) / width) ** 2) + baseline
    y = model(x, 4.0, 405.2, 0.35, 1.5)
    out = fit_custom_molecular_model(
        x, y, model,
        parameter_names=("amplitude", "center", "width", "baseline"),
        initial_values=(3.5, 405.0, 0.4, 1.0),
        intensity_uncertainty=sigma_y,
    )
    assert out["degrees_of_freedom"] == x.size - 4
    assert out["chi_square"] < 1e-8
    assert out["reduced_chi_square"] < 1e-8
    assert np.asarray(out["covariance"]).shape == (4, 4)
    assert len(out["residual"]) == x.size


def test_custom_molecular_model_rejects_bad_uncertainty_and_parameter_names():
    x = np.linspace(400.0, 410.0, 20)
    y = np.ones_like(x)
    def model(w, amplitude):
        return np.full_like(w, amplitude)
    with pytest.raises(ValueError):
        fit_custom_molecular_model(
            x, y, model, parameter_names=("amplitude",),
            initial_values=(1.0,), intensity_uncertainty=np.zeros_like(y),
        )
    with pytest.raises(ValueError):
        fit_custom_molecular_model(
            x, y, model, parameter_names=("amplitude", "amplitude"),
            initial_values=(1.0, 2.0),
        )


def test_molecular_spectrum_rejects_duplicate_wavelength_and_invalid_ranges():
    x = np.array([400.0, 401.0, 402.0, 402.0, 404.0])
    y = np.array([0.0, 1.0, 3.0, 2.0, 0.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze_molecular_spectrum(x, y)
    with pytest.raises(ValueError, match="inside the measured"):
        analyze_molecular_spectrum(
            np.arange(5, dtype=float) + 400.0, y,
            band_ranges_nm=[(399.0, 402.0)],
        )


def test_custom_model_rejects_invalid_bounds_and_out_of_bounds_initial_values():
    x = np.linspace(400.0, 410.0, 20)
    y = np.ones_like(x)
    def model(w, amplitude):
        return np.full_like(w, amplitude)
    common = dict(parameter_names=("amplitude",), initial_values=(1.0,))
    with pytest.raises(ValueError, match="lower bound"):
        fit_custom_molecular_model(x, y, model, bounds=(2.0, 1.0), **common)
    with pytest.raises(ValueError, match="within the supplied"):
        fit_custom_molecular_model(x, y, model, bounds=(2.0, 3.0), **common)


def test_custom_model_rejects_nonfinite_or_wrong_shape_output():
    x = np.linspace(400.0, 410.0, 20)
    y = np.ones_like(x)
    def wrong_shape(w, amplitude):
        return np.array([amplitude])
    with pytest.raises(ValueError, match="Custom molecular model fitting failed"):
        fit_custom_molecular_model(
            x, y, wrong_shape, parameter_names=("amplitude",), initial_values=(1.0,)
        )
    def nonfinite(w, amplitude):
        return np.full_like(w, np.nan)
    with pytest.raises(ValueError, match="Custom molecular model fitting failed"):
        fit_custom_molecular_model(
            x, y, nonfinite, parameter_names=("amplitude",), initial_values=(1.0,)
        )



def test_custom_model_requires_more_samples_than_parameters():
    x = np.linspace(400.0, 410.0, 5)
    y = np.ones_like(x)

    def model(w, a, b, c, d, e):
        return a * w + b * w**2 + c * w**3 + d * w**4 + e

    with pytest.raises(ValueError, match="more samples than fitted parameters"):
        fit_custom_molecular_model(
            x, y, model,
            parameter_names=("a", "b", "c", "d", "e"),
            initial_values=(0.0, 0.0, 0.0, 0.0, 1.0),
        )


def test_custom_model_wraps_optimizer_nonconvergence():
    x = np.linspace(400.0, 410.0, 20)
    y = np.ones_like(x)

    def invalid_domain_model(w, a):
        if a > 0.0:
            raise RuntimeError("synthetic optimizer failure")
        return np.full_like(w, a)

    with pytest.raises(ValueError, match="Custom molecular model fitting failed"):
        fit_custom_molecular_model(
            x, y, invalid_domain_model,
            parameter_names=("a",), initial_values=(-1.0,),
            bounds=(-2.0, 2.0),
        )



def test_molecular_boltzmann_temperature_recovers_supplied_excitation_temperature():
    energy = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    wavelength = np.array([400.0, 405.0, 410.0, 415.0, 420.0, 425.0])
    einstein_a = np.array([1.0e7, 1.2e7, 1.4e7, 1.6e7, 1.8e7, 2.0e7])
    degeneracy = np.array([1, 3, 5, 3, 5, 7], dtype=float)
    expected_temperature_eV = 0.8
    intensity = 1.0e-12 * einstein_a * degeneracy / wavelength * np.exp(-energy / expected_temperature_eV)

    result = fit_molecular_boltzmann_temperature(
        energy, intensity, einstein_a, degeneracy, wavelength,
        intensity_uncertainty=0.02 * intensity,
    )

    assert result["excitation_temperature_eV"] == pytest.approx(expected_temperature_eV, rel=1e-10)
    assert result["excitation_temperature_K"] == pytest.approx(expected_temperature_eV * 11604.518121550082)
    assert result["line_count"] == energy.size
    assert result["degrees_of_freedom"] == energy.size - 2
    assert result["r_squared"] == pytest.approx(1.0)
    assert result["excitation_temperature_uncertainty_eV"] > 0
    assert "not automatically electron or ion temperature" in result["caveats"][0]


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"integrated_intensity": np.array([1.0, 1.0, 0.0, 1.0])}, "must be positive"),
        ({"einstein_A_s": np.array([1.0, 1.0, np.nan, 1.0])}, "finite"),
        ({"degeneracy": np.array([1.0, 0.0, 1.0, 1.0])}, "must be positive"),
    ],
)
def test_molecular_boltzmann_temperature_rejects_invalid_transition_data(updates, message):
    values = {
        "upper_energy_eV": np.array([0.5, 1.0, 1.5, 2.0]),
        "integrated_intensity": np.array([4.0, 3.0, 2.0, 1.0]),
        "einstein_A_s": np.ones(4),
        "degeneracy": np.ones(4),
        "wavelength_nm": np.full(4, 400.0),
    }
    values.update(updates)
    with pytest.raises(ValueError, match=message):
        fit_molecular_boltzmann_temperature(**values)


def test_molecular_boltzmann_temperature_rejects_nonnegative_slope_and_bad_uncertainty():
    energy = np.array([0.5, 1.0, 1.5, 2.0])
    common = {
        "upper_energy_eV": energy,
        "integrated_intensity": np.exp(energy),
        "einstein_A_s": np.ones(4),
        "degeneracy": np.ones(4),
        "wavelength_nm": np.full(4, 400.0),
    }
    with pytest.raises(ValueError, match="slope must be negative"):
        fit_molecular_boltzmann_temperature(**common)
    with pytest.raises(ValueError, match="intensity_uncertainty"):
        fit_molecular_boltzmann_temperature(
            **{**common, "integrated_intensity": np.exp(-energy)},
            intensity_uncertainty=np.zeros(4),
        )
