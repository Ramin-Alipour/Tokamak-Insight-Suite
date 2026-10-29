import numpy as np
import pytest

from tokamak_analysis.langmuir_advanced import (
    analyze_langmuir_probe_series,
    estimate_multispecies_ion_density,
    fit_langmuir_electron_models,
    langmuir_fit_window_sensitivity,
)


def test_single_maxwellian_advanced_fit_recovers_temperature():
    v = np.linspace(-5.0, 5.0, 81)
    i = -2e-3 + 3e-4 * np.exp((v + 5.0) / 2.5)
    result = fit_langmuir_electron_models(v, i, (-4.0, 2.0))
    assert result["electron_temperature_components_eV"][0] == pytest.approx(2.5, rel=0.03)
    assert result["physical_distribution_confirmed"] is False


def test_two_temperature_fit_reports_two_components_and_warning():
    v = np.linspace(-6.0, 4.0, 121)
    i = -1e-3 + 1e-4 * np.exp((v + 6.0) / 1.5) + 2e-4 * np.exp((v + 6.0) / 4.0)
    result = fit_langmuir_electron_models(v, i, (-5.5, 2.0), model="two_temperature")
    assert len(result["electron_temperature_components_eV"]) == 2
    assert result["component_identifiability_warning"]
    assert result["physical_distribution_confirmed"] is False


def test_series_analysis_handles_time_and_channel_dimensions():
    v = np.linspace(-5.0, 5.0, 21)
    currents = np.stack([
        np.stack([np.sinh(v / 3.0) * 1e-3, np.sinh((v - 0.2) / 3.5) * 1e-3]),
        np.stack([np.sinh(v / 3.2) * 1e-3, np.sinh((v - 0.3) / 3.6) * 1e-3]),
    ])
    result = analyze_langmuir_probe_series(v, currents, time_labels=[0.0, 1.0], channel_names=["a", "b"])
    assert result["time_count"] == 2
    assert result["channel_count"] == 2
    assert result["time_slices"][1]["time_label"] == 1.0
    assert result["temporal_trend_analysis_performed"] is False


def test_multispecies_density_requires_fractions_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        estimate_multispecies_ion_density(
            -1e-3, 5.0, 1e-4,
            [{"mass_kg": 3.3e-27, "fraction": 0.7, "sheath_coefficient": 0.61},
             {"mass_kg": 6.6e-27, "fraction": 0.2, "sheath_coefficient": 0.61}],
        )


def test_multispecies_density_returns_model_labelled_estimate():
    result = estimate_multispecies_ion_density(
        -1e-3, 5.0, 1e-4,
        [{"mass_kg": 3.3e-27, "fraction": 0.7, "sheath_coefficient": 0.61},
         {"mass_kg": 6.6e-27, "fraction": 0.3, "sheath_coefficient": 0.61}],
    )
    assert result["ion_density_estimate_m-3"] > 0
    assert result["physical_regime_confirmed"] is False


def test_fit_window_sensitivity_reports_spread_separately_from_uncertainty():
    v = np.linspace(-10.0, 10.0, 101)
    i = -2e-3 + 2e-4 * np.exp((v + 10.0) / 3.0)
    result = langmuir_fit_window_sensitivity(v, i, (-8.0, 0.0))
    assert result["valid_temperature_fit_count"] >= 2
    assert result["temperature_window_sensitivity_span_eV"] is not None
    assert "not a statistical confidence interval" in result["interpretation"]
