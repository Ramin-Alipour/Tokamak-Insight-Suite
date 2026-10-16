import numpy as np
import pytest

from tokamak_analysis.safety_factor import estimate_safety_factor
from tokamak_analysis.neutron_flux import analyze_neutron_flux
from tokamak_analysis.spectral_line_ratio import analyze_line_ratio
from tokamak_analysis.wall_recycling import recycling_coefficient
from tokamak_analysis.measurement_uncertainty import propagate_product_uncertainty
from tokamak_analysis.channel_synchronization import synchronize_channels
from tokamak_analysis.plasma_geometry import geometry_from_boundary


def test_cylindrical_safety_factor_uses_minor_over_major_radius():
    result = estimate_safety_factor(2.0, 0.5, 1.0, 0.1)
    assert result["q_cylindrical_estimate"] == pytest.approx(2.5)


def test_neutron_flux_treats_input_as_binned_counts_by_default():
    result = analyze_neutron_flux([0.0, 1.0, 2.0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(result["count_rate_s-1"], [10.0, 20.0, 30.0])
    assert result["total_counts"] == 60.0


def test_neutron_flux_supports_cumulative_counters_explicitly():
    result = analyze_neutron_flux([0.0, 1.0, 2.0], [0.0, 10.0, 30.0], counts_are_cumulative=True)
    assert result["total_counts"] == 30.0
    np.testing.assert_allclose(result["count_rate_s-1"], [10.0, 15.0, 20.0])


def test_line_ratio_uncertainty_is_finite_for_zero_numerator():
    result = analyze_line_ratio(0.0, 2.0, uncertainty_numerator=0.1, uncertainty_denominator=0.2)
    assert result["line_ratio"] == 0.0
    assert result["line_ratio_uncertainty"] == pytest.approx(0.05)


def test_recycling_uncertainty_is_finite_for_zero_returned_flux():
    result = recycling_coefficient(2.0, 0.0, uncertainties=(0.1, 0.2))
    assert result["recycling_coefficient"] == 0.0
    assert result["uncertainty"] == pytest.approx(0.1)


def test_product_uncertainty_handles_zero_factor():
    result = propagate_product_uncertainty([0.0, 3.0], [2.0, 0.1])
    assert result["value"] == 0.0
    assert result["standard_uncertainty"] == pytest.approx(6.0)
    assert np.isnan(result["relative_uncertainty"])


def test_synchronization_rejects_extrapolation():
    with pytest.raises(ValueError, match="inside"):
        synchronize_channels([[0.0, 1.0, 2.0], [0.5, 1.0, 1.5]],
                              [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
                              target_time=[0.0, 1.0])


def test_geometry_rejects_boundary_crossing_symmetry_axis():
    with pytest.raises(ValueError, match="symmetry axis"):
        geometry_from_boundary([-0.1, 0.5, 0.5, -0.1, -0.1],
                               [0.0, 0.0, 1.0, 1.0, 0.0])


def test_fft_one_sided_amplitude_does_not_double_nyquist_bin():
    from tokamak_analysis.fft_analysis import analyze_fft

    time = np.arange(8, dtype=float) / 8.0
    signal = (-1.0) ** np.arange(8)
    result = analyze_fft(time, signal, detrend=False, window="rectangular")
    assert result["amplitude"][-1] == pytest.approx(1.0)


def test_fft_frequency_resolution_uses_n_times_sample_interval():
    from tokamak_analysis.fft_analysis import frequency_resolution_Hz

    time = np.arange(8, dtype=float) * 0.25
    result = frequency_resolution_Hz(time)
    assert result["frequency_resolution_Hz"] == pytest.approx(0.5)


def test_confinement_uncertainty_handles_zero_stored_energy():
    from tokamak_analysis.confinement import energy_confinement_time

    result = energy_confinement_time(0.0, 10.0, uncertainty_energy_J=2.0, uncertainty_power_W=1.0)
    assert result["energy_confinement_time_s"] == 0.0
    assert result["uncertainty_s"] == pytest.approx(0.2)


def test_fft_rejects_degenerate_two_sample_hann_window():
    from tokamak_analysis.fft_analysis import analyze_fft

    with pytest.raises(ValueError, match="at least 3"):
        analyze_fft([0.0, 1.0], [0.0, 1.0], window="hann")


def test_radial_and_poloidal_registry_entries_use_distinct_directions():
    from tokamak_analysis.analysis_registry import resolve_analysis

    radial = resolve_analysis(61)([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    poloidal = resolve_analysis(62)([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert radial["direction"] == "radial"
    assert poloidal["direction"] == "poloidal"


def test_electron_and_ion_heating_registry_entries_are_species_specific():
    from tokamak_analysis.analysis_registry import resolve_analysis

    electron = resolve_analysis(66)([0.0, 1.0], [10.0, 10.0])
    ion = resolve_analysis(67)([0.0, 1.0], [10.0, 10.0])
    assert electron["species"] == "electron"
    assert ion["species"] == "ion"


def test_divertor_particle_flux_entry_does_not_require_heat_flux():
    from tokamak_analysis.analysis_registry import resolve_analysis

    result = resolve_analysis(74)([0.0, 1.0, 2.0], [1.0, 2.0, 1.0])
    assert result["peak_particle_flux_m2_s"] == 2.0
    assert result["integrated_particle_flux_m_s-1"] == pytest.approx(3.0)
