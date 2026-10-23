import numpy as np
import pytest
from tokamak_analysis.mirnov_analysis import analyze_mirnov_array, analyze_mirnov_signal, magnetic_perturbation_from_mirnov_voltage


def test_mirnov_signal_psd_and_bdot():
    t = np.arange(4096) / 20000.0
    v = 0.4*np.sin(2*np.pi*1250*t) + 0.1*np.sin(2*np.pi*2500*t)
    out = analyze_mirnov_signal(t, v, coil_area_m2=1e-4, turns=10, nperseg=2048)
    assert out["dominant_peaks_Hz"][0] == pytest.approx(1250, abs=20)
    np.testing.assert_allclose(out["approx_Bdot_T_s"], -v/1e-3)
    assert out["power_spectral_density_V2_per_Hz"].shape == out["frequency_Hz"].shape
    assert out["sampling_frequency_Hz"] == pytest.approx(20000)


@pytest.mark.parametrize("turns", [0, -1, 1.5, np.nan, np.inf, True])
def test_invalid_turn_count(turns):
    t = np.arange(16)*1e-4
    with pytest.raises(ValueError, match="turns"):
        analyze_mirnov_signal(t, np.ones_like(t), coil_area_m2=1e-4, turns=turns)


def test_geometry_must_be_supplied_together():
    t = np.arange(16)*1e-4
    with pytest.raises(ValueError, match="together"):
        analyze_mirnov_signal(t, np.ones_like(t), turns=2)


@pytest.mark.parametrize("t,v", [
    ([0,1,2], [1,2]), ([0,1,1], [1,2,3]), ([0,1,2], [1,np.nan,3]),
    ([[0,1],[2,3]], [[1,2],[3,4]])
])
def test_invalid_signal_inputs(t, v):
    with pytest.raises(ValueError):
        analyze_mirnov_signal(t, v)


def test_nonuniform_sampling_rejected():
    t = np.array([0,1,2,3,4,5,6,8])*1e-4
    with pytest.raises(ValueError, match="Uniform"):
        analyze_mirnov_signal(t, np.ones_like(t))


def test_invalid_spectral_options_rejected():
    t = np.arange(32)*1e-4
    y = np.sin(2*np.pi*1000*t)
    with pytest.raises(ValueError, match="nperseg"):
        analyze_mirnov_signal(t, y, nperseg=33)
    with pytest.raises(ValueError, match="frequency range"):
        analyze_mirnov_signal(t, y, frequency_range_Hz=(1, 1e9))
    with pytest.raises(ValueError, match="prominence"):
        analyze_mirnov_signal(t, y, peak_prominence_fraction=2)


@pytest.mark.parametrize("detrend", ["bad", "", None])
def test_invalid_detrend_rejected(detrend):
    t = np.arange(16)*1e-4
    with pytest.raises(ValueError, match="detrend"):
        analyze_mirnov_signal(t, np.ones_like(t), detrend=detrend)


@pytest.mark.parametrize("turns", [0, -1, 1.5, np.nan, np.inf, True])
def test_integration_rejects_invalid_turn_count(turns):
    with pytest.raises(ValueError, match="turns"):
        magnetic_perturbation_from_mirnov_voltage(np.arange(4)*1e-4, np.ones(4), turns, 1e-4)


def test_integration_recovers_known_linear_perturbation():
    t = np.linspace(0, 0.01, 101)
    n, area, slope = 20, 2e-4, 0.5
    v = np.full_like(t, -n*area*slope)
    out = magnetic_perturbation_from_mirnov_voltage(t, v, n, area)
    np.testing.assert_allclose(out["dB_dt_T_s"], slope)
    np.testing.assert_allclose(out["delta_B_T"], slope*t)


def test_integration_baseline_removal():
    t = np.linspace(0, 1, 11)
    out = magnetic_perturbation_from_mirnov_voltage(t, np.full_like(t, 3.0), 1, 1.0, baseline_voltage_V=3.0, polarity_sign=1)
    np.testing.assert_allclose(out["delta_B_T"], 0.0)


def test_array_candidate_mode_number():
    fs = 10000.0
    t = np.arange(2048)/fs
    angles = np.array([0.0, 0.7, 1.5, 2.4, 3.2])
    signals = np.array([np.sin(2*np.pi*500*t + 2*a) for a in angles])
    out = analyze_mirnov_array(t, signals, angles, nperseg=512)
    assert out["dominant_frequency_Hz"] == pytest.approx(500, abs=20)
    assert out["estimated_toroidal_mode_number"] == 2
    assert out["phase_fit_rms_rad"] < 0.1


def test_array_requires_three_channels():
    t = np.arange(64)*1e-4
    with pytest.raises(ValueError, match="three"):
        analyze_mirnov_array(t, np.ones((2, len(t))), [0, 1])
