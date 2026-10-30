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
            [{"mass_kg": 3.3e-27, "fraction": 0.7, "charge_state": 1, "sheath_coefficient": 0.61},
             {"mass_kg": 6.6e-27, "fraction": 0.2, "charge_state": 1, "sheath_coefficient": 0.61}],
        )


def test_multispecies_density_returns_model_labelled_estimate():
    result = estimate_multispecies_ion_density(
        -1e-3, 5.0, 1e-4,
        [{"mass_kg": 3.3e-27, "fraction": 0.7, "charge_state": 1, "sheath_coefficient": 0.61},
         {"mass_kg": 6.6e-27, "fraction": 0.3, "charge_state": 1, "sheath_coefficient": 0.61}],
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


def test_stretched_exponential_model_is_explicitly_empirical():
    from tokamak_analysis.langmuir_advanced import fit_langmuir_stretched_exponential
    v = np.linspace(-5.0, 5.0, 81)
    i = -2e-3 + 3e-4 * np.exp(np.maximum((v + 5.0) / 2.5, 0.0) ** 1.2)
    result = fit_langmuir_stretched_exponential(v, i, (-4.0, 3.0))
    assert result["model"] == "stretched_exponential_empirical_shape"
    assert result["kinetic_distribution_identified"] is False


def test_electron_density_and_particle_flux_helpers_are_positive():
    from tokamak_analysis.langmuir_advanced import (
        estimate_electron_density_from_saturation,
        estimate_ion_particle_flux,
    )
    ne = estimate_electron_density_from_saturation(1e-3, 5.0, 1e-4)
    flux = estimate_ion_particle_flux(1e18, 5.0, 3.34e-27, sheath_factor=1.0)
    assert ne["electron_density_estimate_m-3"] > 0
    assert ne["physical_regime_confirmed"] is False
    assert flux["ion_particle_flux_m-2_s-1"] > 0
    assert flux["uncertainty_propagation_performed"] is False


def test_smoothed_plasma_potential_candidates_keep_raw_and_smoothed_methods_separate():
    from tokamak_analysis.langmuir_advanced import estimate_smoothed_plasma_potential_candidates
    v = np.linspace(-5.0, 5.0, 51)
    i = np.sinh(v / 2.0) * 1e-3 + 1e-5 * np.sin(17.0 * v)
    result = estimate_smoothed_plasma_potential_candidates(v, i, window_length=7, polynomial_order=2)
    assert "raw_max_dIdV_candidate_V" in result
    assert "smoothed_max_dIdV_candidate_V" in result
    assert result["physical_plasma_potential_confirmed"] is False


def test_advanced_helpers_are_exported_from_public_langmuir_module():
    from tokamak_analysis.langmuir_probe import (
        estimate_electron_density_from_saturation,
        estimate_ion_particle_flux,
        estimate_smoothed_plasma_potential_candidates,
        fit_langmuir_stretched_exponential,
        langmuir_fit_window_sensitivity,
    )

    assert callable(estimate_electron_density_from_saturation)
    assert callable(estimate_ion_particle_flux)
    assert callable(estimate_smoothed_plasma_potential_candidates)
    assert callable(fit_langmuir_stretched_exponential)
    assert callable(langmuir_fit_window_sensitivity)


def test_electron_density_is_separate_model_helper_not_canonical_sweep_output():
    from tokamak_analysis.langmuir_probe import (
        analyze_langmuir_probe,
        estimate_electron_density_from_saturation,
    )

    v = np.linspace(-5.0, 5.0, 41)
    i = np.sinh(v / 3.0) * 1e-3
    canonical = analyze_langmuir_probe(v, i)
    estimate = estimate_electron_density_from_saturation(1e-3, 5.0, 1e-4)

    assert "electron_density_estimate_m-3" not in canonical
    assert estimate["electron_density_estimate_m-3"] > 0
    assert estimate["physical_regime_confirmed"] is False
    assert estimate["uncertainty_propagation_performed"] is False

def test_advanced_electron_fits_normalize_both_current_conventions():
    v = np.linspace(-5.0, 5.0, 81)
    i = -2e-3 + 3e-4 * np.exp((v + 5.0) / 2.5)
    positive = fit_langmuir_electron_models(v, i, (-4.0, 2.0), current_convention="electron_positive")
    negative = fit_langmuir_electron_models(v, -i, (-4.0, 2.0), current_convention="ion_positive")
    assert negative["electron_temperature_components_eV"] == pytest.approx(positive["electron_temperature_components_eV"], rel=1e-8)
    assert negative["current_convention_normalized"] == "electron_positive"
    assert negative["current_convention_input"] == "ion_positive"


def test_two_temperature_fit_exposes_component_uncertainty_and_identifiability_diagnostics():
    v = np.linspace(-6.0, 4.0, 121)
    i = -1e-3 + 1e-4 * np.exp((v + 6.0) / 1.5) + 2e-4 * np.exp((v + 6.0) / 4.0)
    result = fit_langmuir_electron_models(v, i, (-5.5, 2.0), model="two_temperature")
    assert len(result["electron_temperature_components_eV"]) == 2
    assert "parameter_correlation_matrix" in result
    assert "covariance_condition_number" in result
    assert "temperature_component_standard_uncertainties_eV" in result
    if result["covariance_valid"]:
        assert len(result["temperature_component_standard_uncertainties_eV"]) == 2
    else:
        assert result["temperature_component_standard_uncertainties_eV"] is None
        assert result["temperature_component_uncertainty_status"] == "unavailable_invalid_or_singular_fit_covariance"


def test_stretched_exponential_and_smoothed_derivatives_share_current_polarity_contract():
    from tokamak_analysis.langmuir_advanced import (
        estimate_smoothed_plasma_potential_candidates,
        fit_langmuir_stretched_exponential,
    )
    v = np.linspace(-5.0, 5.0, 81)
    i = -2e-3 + 3e-4 * np.exp(np.maximum((v + 5.0) / 2.5, 0.0) ** 1.2)
    stretched_a = fit_langmuir_stretched_exponential(v, i, (-4.0, 3.0), current_convention="electron_positive")
    stretched_b = fit_langmuir_stretched_exponential(v, -i, (-4.0, 3.0), current_convention="ion_positive")
    assert stretched_a["shape_exponent"] == pytest.approx(stretched_b["shape_exponent"], rel=1e-8)
    smooth_a = estimate_smoothed_plasma_potential_candidates(v, i, current_convention="electron_positive")
    smooth_b = estimate_smoothed_plasma_potential_candidates(v, -i, current_convention="ion_positive")
    assert smooth_a["raw_max_dIdV_candidate_V"] == pytest.approx(smooth_b["raw_max_dIdV_candidate_V"])
    assert smooth_a["smoothed_max_dIdV_candidate_V"] == pytest.approx(smooth_b["smoothed_max_dIdV_candidate_V"])


def test_multispecies_fraction_basis_and_current_polarity_are_explicit():
    species = [
        {"mass_kg": 3.3e-27, "fraction": 0.7, "charge_state": 1, "sheath_coefficient": 0.61},
        {"mass_kg": 6.6e-27, "fraction": 0.3, "charge_state": 1, "sheath_coefficient": 0.61},
    ]
    electron_positive = estimate_multispecies_ion_density(-1e-3, 5.0, 1e-4, species)
    ion_positive = estimate_multispecies_ion_density(1e-3, 5.0, 1e-4, species, current_convention="ion_positive")
    assert ion_positive["ion_density_estimate_m-3"] == pytest.approx(electron_positive["ion_density_estimate_m-3"])
    assert ion_positive["ion_species_fraction_basis"] == "number_density_fraction"
    with pytest.raises(ValueError, match="number_density"):
        estimate_multispecies_ion_density(-1e-3, 5.0, 1e-4, species, fraction_basis="mass")


def test_electron_density_helper_normalizes_current_polarity():
    from tokamak_analysis.langmuir_advanced import estimate_electron_density_from_saturation
    a = estimate_electron_density_from_saturation(1e-3, 5.0, 1e-4, current_convention="electron_positive")
    b = estimate_electron_density_from_saturation(-1e-3, 5.0, 1e-4, current_convention="ion_positive")
    assert a["electron_density_estimate_m-3"] == pytest.approx(b["electron_density_estimate_m-3"])
    assert b["current_convention_normalized"] == "electron_positive"


def test_public_wrapper_forwards_current_uncertainty_and_current_convention():
    from tokamak_analysis.langmuir_probe import analyze_langmuir_probe
    v = np.linspace(-1.0, 1.0, 9)
    i = 1e-3 * v
    result = analyze_langmuir_probe(v, -i, current_convention="ion_positive", current_uncertainty_A=1e-5)
    assert result["current_convention_input"] == "ion_positive"
    assert result["current_convention"] == "electron_positive"
    assert result["floating_potential_standard_uncertainty_V"] is not None

def test_smoothed_derivative_helper_reports_second_derivative_zero_crossings_separately():
    from tokamak_analysis.langmuir_advanced import estimate_smoothed_plasma_potential_candidates
    v = np.linspace(-2.0, 2.0, 81)
    i = v**3 + 0.01 * v
    result = estimate_smoothed_plasma_potential_candidates(v, i, window_length=9, polynomial_order=3)
    assert "raw_d2IdV2_zero_crossing_candidates_V" in result
    assert "smoothed_d2IdV2_zero_crossing_candidates_V" in result
    assert result["second_derivative_inflection_method"].startswith("zero_crossings_of_d2IdV2")
    assert "raw_max_d2IdV2_extremum_V" in result



def test_advanced_fits_aggregate_duplicate_voltages_before_curve_fit():
    v = np.linspace(-5.0, 5.0, 81)
    i = -2e-3 + 3e-4 * np.exp((v + 5.0) / 2.5)
    v_dup = np.r_[v, v[20], v[20], np.nan]
    i_dup = np.r_[i, i[20] + 5e-4, i[20] - 5e-4, 0.0]
    baseline = fit_langmuir_electron_models(v, i, (-4.0, 2.0))
    aggregated = fit_langmuir_electron_models(v_dup, i_dup, (-4.0, 2.0))
    assert aggregated["fit_points"] == baseline["fit_points"]
    assert aggregated["electron_temperature_components_eV"] == pytest.approx(baseline["electron_temperature_components_eV"], rel=1e-8)
    assert aggregated["input_quality"]["duplicate_voltage_pair_count_aggregated"] == 2
    assert aggregated["input_quality"]["discarded_nonfinite_pair_count"] == 1
    with pytest.raises(ValueError, match="eight distinct"):
        fit_langmuir_electron_models(np.repeat(v[:7], 2), np.repeat(i[:7], 2), (-5.0, -4.0))


def test_stretched_fit_aggregates_duplicate_voltages_and_reports_parameter_uncertainty():
    from tokamak_analysis.langmuir_advanced import fit_langmuir_stretched_exponential
    v = np.linspace(-5.0, 5.0, 81)
    i = -2e-3 + 3e-4 * np.exp(-np.maximum((3.0 - v) / 2.5, 0.0) ** 1.2)
    result = fit_langmuir_stretched_exponential(np.r_[v, v[10]], np.r_[i, i[10]], (-4.0, 3.0))
    assert result["input_quality"]["duplicate_voltage_pair_count_aggregated"] == 1
    assert result["preprocessing_contract"].endswith("duplicate_voltage_current_mean")
    assert result["parameter_uncertainty_status"] in {"conditional_covariance_standard_errors_available", "unavailable_invalid_or_singular_fit_covariance"}
    if result["covariance_valid"]:
        assert result["shape_exponent_standard_uncertainty"] is not None
    else:
        assert result["shape_exponent_standard_uncertainty"] is None


def test_local_polynomial_smoothing_uses_nonuniform_voltage_coordinates():
    from tokamak_analysis.langmuir_advanced import estimate_smoothed_plasma_potential_candidates
    v = np.sort(np.r_[-4.0 + 3.0*np.linspace(0.0,1.0,15)**1.4, -0.8 + 4.8*np.linspace(0.0,1.0,37)**1.7])
    v = np.unique(v)
    i = 2.0 + 3.0*v + 0.5*v**2
    result = estimate_smoothed_plasma_potential_candidates(v, i, window_length=7, polynomial_order=2)
    assert result["smoothing_method"] == "local_polynomial_least_squares_on_actual_voltage_coordinates"
    assert result["voltage_grid_uniformity"].startswith("not_required")
    assert np.asarray(result["smoothed_current_A"]) == pytest.approx(i, rel=1e-8, abs=1e-8)


def test_advanced_helper_uncertainties_are_conditional_and_never_zero_filled():
    from tokamak_analysis.langmuir_advanced import estimate_electron_density_from_saturation, estimate_ion_particle_flux
    ne0 = estimate_electron_density_from_saturation(1e-3, 5.0, 1e-4)
    assert ne0["electron_density_standard_uncertainty_m-3"] is None
    assert ne0["uncertainty_propagation_performed"] is False
    ne1 = estimate_electron_density_from_saturation(1e-3, 5.0, 1e-4,
        current_standard_uncertainty_A=1e-5, temperature_relative_standard_uncertainty=0.02,
        area_relative_standard_uncertainty=0.01)
    assert ne1["electron_density_standard_uncertainty_m-3"] > 0
    flux0 = estimate_ion_particle_flux(1e18, 5.0, 3.34e-27, sheath_factor=1.0)
    assert flux0["ion_particle_flux_standard_uncertainty_m-2_s-1"] is None
    flux1 = estimate_ion_particle_flux(1e18, 5.0, 3.34e-27, sheath_factor=1.0,
        density_relative_standard_uncertainty=0.01, temperature_relative_standard_uncertainty=0.02,
        mass_relative_standard_uncertainty=0.01, sheath_factor_relative_standard_uncertainty=0.03)
    assert flux1["ion_particle_flux_standard_uncertainty_m-2_s-1"] > 0


def test_multispecies_density_requires_charge_state_and_applies_charge_dependence():
    z1 = [{"mass_kg": 3.3e-27, "fraction": 1.0, "charge_state": 1, "sheath_coefficient": 0.61}]
    z2 = [{"mass_kg": 3.3e-27, "fraction": 1.0, "charge_state": 2, "sheath_coefficient": 0.61}]
    n1 = estimate_multispecies_ion_density(-1e-3, 5.0, 1e-4, z1)["ion_density_estimate_m-3"]
    n2 = estimate_multispecies_ion_density(-1e-3, 5.0, 1e-4, z2)["ion_density_estimate_m-3"]
    assert n2 == pytest.approx(n1 / (2.0**1.5))
    assert "not_propagated" in estimate_multispecies_ion_density(-1e-3,5.0,1e-4,z1)["uncertainty_status"]


def test_single_species_core_density_includes_charge_state():
    from tis.probe_analysis import langmuir_iv_analysis
    v = np.linspace(-10.0, 10.0, 201)
    i = -1e-3 + 2e-4*np.exp((v+10.0)/3.0)
    common = dict(electron_fit_range=(-8.0,-1.0), ion_fit_range=(-10.0,-9.0),
                  probe_area_m2=1e-4, ion_mass_kg=3.34e-27,
                  ion_saturation_coefficient=0.61, sheath_model="test model",
                  fit_quality_min_r2=0.0, fit_quality_max_nrmse=10.0,
                  ion_flatness_max_relative_change=10.0)
    a = langmuir_iv_analysis(v,i,ion_charge_state=1,**common)
    b = langmuir_iv_analysis(v,i,ion_charge_state=2,**common)
    assert b["ion_density_estimate_m-3"] == pytest.approx(a["ion_density_estimate_m-3"]/(2.0**1.5))
    assert b["ion_density_model"]["ion_charge_state"] == 2


def test_advanced_model_run_button_requires_a_valid_electron_window():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "advanced_window_valid=(efit is not None and advanced_window_points>=advanced_min_points)" in source
    assert "disabled=bool(needs_electron_window and not advanced_window_valid)" in source


def test_advanced_ui_model_choice_is_defined_before_fit_window_validation():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    selector = source.index("model_choice=st.selectbox('Electron model'")
    validation = source.index("advanced_min_points=")
    assert selector < validation
    assert "advanced_window_points>=advanced_min_points" in source
