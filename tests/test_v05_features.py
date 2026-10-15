"""Regression tests for v0.5 additions and fixes. Every expected value is analytic or known-by-construction."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tis import plasma, report
from tis.catalog import DIAGNOSTICS, FORMULAS
from tis.core import (grid_from_table, integrate_time_series, langmuir_iv_analysis, load_table,
                      mode_number_from_channels, mode_number_from_phase)
from tis.events import current_quench_analysis, detect_elms, detect_sawteeth, interval_statistics
from tis.signals import (hampel_despike, longest_contiguous_segment, prepare_time_series_robust,
                         resample_uniform, sampling_audit)

ROOT = Path(__file__).resolve().parents[1]


# ---------------- import fixes ----------------
def test_load_table_decodes_windows_1252_and_records_encoding():
    raw = "t,Température\n0,1\n1,2\n2,3\n".encode("cp1252")
    frame = load_table(raw, "x.csv")
    assert "Température" in frame.columns
    assert frame.attrs["encoding"] in {"cp1252", "latin-1"}


def test_load_table_strips_utf8_bom():
    frame = load_table("\ufefftime,sig\n0,1\n1,2\n2,3\n".encode("utf-8"), "x.csv")
    assert list(frame.columns) == ["time", "sig"]
    assert frame.attrs["encoding"] == "utf-8-sig"


def test_load_table_honours_explicit_encoding_failure():
    with pytest.raises(ValueError, match="decode"):
        load_table("t,é\n0,1\n1,2\n".encode("latin-1"), "x.csv", encoding="utf-8")


# ---------------- sign conventions ----------------
def _wave_phases(n_true, theta):
    # cross-spectral phase relative to sensor 0 for a wave cos(wt - n*theta)
    return -n_true * (theta - theta[0])


def test_mode_number_wave_convention_gives_positive_n_for_forward_wave():
    theta = np.arange(8) * 2 * np.pi / 8
    ph = np.angle(np.exp(1j * _wave_phases(2, theta)))
    assert mode_number_from_phase(theta, ph, convention="wave_cos_wt_minus_n_theta")["mode_number_fit"] == 2
    # default keeps the legacy raw-phase convention (opposite sign) and says so
    legacy = mode_number_from_phase(theta, ph)
    assert legacy["mode_number_fit"] == -2 and legacy["sign_convention"] == "phase_equals_n_theta"


def test_mode_number_rejects_unknown_convention():
    with pytest.raises(ValueError, match="convention"):
        mode_number_from_phase([0, 1, 2, 3], [0, 1, 2, 3], convention="bogus")


@pytest.mark.parametrize("n_true", [1, 2, 3, -2])
def test_mode_number_from_channels_recovers_signed_mode(n_true):
    rng = np.random.default_rng(10 + n_true)
    fs, N = 1e5, 8192
    t = np.arange(N) / fs
    theta = np.arange(8) * 2 * np.pi / 8
    sig = np.array([np.cos(2 * np.pi * 8e3 * t - n_true * a) + 0.2 * rng.standard_normal(N) for a in theta])
    r = mode_number_from_channels(theta, sig, fs, convention="wave_cos_wt_minus_n_theta")
    assert r["mode_number_fit"] == n_true


# ---------------- Langmuir ----------------
def _probe(te=15.0, isat=-1e-3, vf=-20.0, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    v = np.linspace(-80, 30, 600)
    vp = 10.0
    i = np.where(v < vp, isat * (1 - np.exp((v - vf) / te)), isat * (1 - np.exp((vp - vf) / te)))
    return v, i + noise * rng.standard_normal(len(v))


def test_langmuir_reports_te_with_standard_error_and_recovers_temperature():
    v, i = _probe(noise=2e-5)
    r = langmuir_iv_analysis(v, i, (-18, 5), (-80, -50))
    assert r["electron_temperature_eV"] == pytest.approx(15.0, rel=0.08)
    assert r["electron_temperature_standard_error_eV"] > 0
    assert abs(r["electron_temperature_eV"] - 15.0) < 5 * r["electron_temperature_standard_error_eV"] + 0.5


def test_langmuir_ion_positive_convention_matches_electron_positive():
    v, i = _probe(noise=1e-5)
    a = langmuir_iv_analysis(v, i, (-18, 5), (-80, -50))
    b = langmuir_iv_analysis(v, -i, (-18, 5), (-80, -50), current_convention="ion_positive")
    assert b["electron_temperature_eV"] == pytest.approx(a["electron_temperature_eV"])
    with pytest.raises(ValueError, match="current_convention"):
        langmuir_iv_analysis(v, i, current_convention="x")


# ---------------- integration drift ----------------
def test_integrate_time_series_removes_linear_drift_exactly():
    t = np.linspace(0, 1, 10001)
    drift = 0.02 + 0.05 * t                  # pure instrument drift, true signal = 0
    r = integrate_time_series(t, drift, baseline_start_s=0.0, baseline_end_s=0.1, drift_start_s=0.9, drift_end_s=1.0)
    assert np.max(np.abs(r["integrated_signal"])) < 1e-9
    assert r["linear_drift_rate_calibrated_units_per_s"] == pytest.approx(0.05)
    # constant-offset-only correction leaves a quadratic residual
    c = integrate_time_series(t, drift, baseline_start_s=0.0, baseline_end_s=0.1)
    assert np.max(np.abs(c["integrated_signal"])) > 1e-3


def test_integrate_time_series_drift_argument_validation():
    t = np.linspace(0, 1, 101)
    with pytest.raises(ValueError, match="both drift"):
        integrate_time_series(t, t, baseline_start_s=0, baseline_end_s=.1, drift_start_s=.9)
    with pytest.raises(ValueError, match="baseline window"):
        integrate_time_series(t, t, drift_start_s=.9, drift_end_s=1.0)
    with pytest.raises(ValueError, match="after the baseline"):
        integrate_time_series(t, t, baseline_start_s=.5, baseline_end_s=.6, drift_start_s=.1, drift_end_s=.2)


# ---------------- signals ----------------
def test_sampling_audit_locates_gap():
    t = np.arange(1000) * 1e-3
    t = t[(t < 0.4) | (t > 0.45)]
    a = sampling_audit(t)
    assert a["n_gaps"] == 1 and a["largest_gap_s"] == pytest.approx(0.051, abs=2e-3)


def test_robust_policy_handles_a_dropped_sample_that_strict_mode_rejects():
    t = np.arange(10000) / 1e4
    y = np.sin(2 * np.pi * 50 * t)
    t2, y2 = np.delete(t, 5000), np.delete(y, 5000)
    with pytest.raises(ValueError, match="gap"):
        prepare_time_series_robust(t2, y2, None, "error")
    _, _, fs, rep = prepare_time_series_robust(t2, y2, None, "interpolate")
    assert rep["modified"] and fs == pytest.approx(1e4, rel=1e-3)
    _, yy, _, rep2 = prepare_time_series_robust(t2, y2, None, "longest_segment")
    assert rep2["modified"] and len(yy) in (4999, 5000)


def test_robust_policy_is_identity_for_clean_data():
    t = np.arange(2000) / 1e3
    t1, y1, fs, rep = prepare_time_series_robust(t, np.sin(t), None, "interpolate")
    assert not rep["modified"] and len(t1) == 2000


def test_resample_does_not_invent_data_inside_long_gaps():
    t = np.r_[np.arange(0, 1, 1e-3), np.arange(2, 3, 1e-3)]
    r = resample_uniform(t, np.sin(t))
    assert r["n_nan_in_long_gaps"] > 900
    seg = longest_contiguous_segment(t, np.sin(t))
    assert seg["n_segments"] == 2 and seg["n_samples_kept"] == 1000


def test_hampel_removes_isolated_spikes_but_keeps_signal():
    t = np.linspace(0, 1, 2000)
    y = np.sin(2 * np.pi * 5 * t)
    z = y.copy(); z[[300, 900, 1500]] += 25.0
    r = hampel_despike(z, 11, 6.0)
    assert r["n_replaced"] == 3 and np.max(np.abs(r["values"] - y)) < 0.05
    clean = hampel_despike(y, 11, 6.0)
    assert clean["n_replaced"] == 0


# ---------------- plasma scalings (ITER design reference) ----------------
def test_q95_iter_reference():
    assert plasma.q95_iter_formula(15, 5.3, 6.2, 2.0, 1.7, 0.33) == pytest.approx(3.0, abs=0.1)


def test_ipb98y2_iter_reference_and_exponents():
    tau = plasma.ipb98y2_confinement_time(15, 5.3, 87, 10.1, 2.5, 6.2, 2.0, 1.7)["tau_E_IPB98y2_s"]
    assert tau == pytest.approx(3.7, rel=0.06)          # ITER design value 3.7 s
    t2 = plasma.ipb98y2_confinement_time(30, 5.3, 87, 10.1, 2.5, 6.2, 2.0, 1.7)["tau_E_IPB98y2_s"]
    assert t2 / tau == pytest.approx(2 ** 0.93, rel=1e-9)  # Ip^0.93 scaling


def test_martin08_reference_and_power_law():
    p = plasma.lh_threshold_martin08(5e19, 5.3, 680.0, 2.5)["P_LH_MW"]
    assert 45 < p < 60
    p2 = plasma.lh_threshold_martin08(1e20, 5.3, 680.0, 2.5)["P_LH_MW"]
    assert p2 / p == pytest.approx(2 ** 0.717, rel=1e-9)


def test_beta_and_greenwald_definitions():
    b = plasma.beta_metrics(1e20, 1e4, 1e4, 15e6, 5.3, 2.0, 1.7)
    p = 1.602176634e-19 * 2e20 * 1e4
    assert b["pressure_Pa"] == pytest.approx(p)
    assert b["beta_t"] == pytest.approx(2 * 1.25663706212e-6 * p / 5.3 ** 2)
    g = plasma.greenwald(6e19, 1e6, 0.5)
    assert g["greenwald_density_1e20_m-3"] == pytest.approx(1 / (math.pi * 0.25))


def test_spitzer_and_collisionality_scaling():
    a = plasma.spitzer_resistivity(1000, 1.0, ln_lambda=17)["eta_perp_ohm_m"]
    b = plasma.spitzer_resistivity(2000, 1.0, ln_lambda=17)["eta_perp_ohm_m"]
    assert a / b == pytest.approx(2 ** 1.5)
    n1 = plasma.collisionality_nu_star(1e20, 1e3, 3, 1.0, 0.3, ln_lambda=15)
    n2 = plasma.collisionality_nu_star(1e20, 2e3, 3, 1.0, 0.3, ln_lambda=15)
    assert n1 / n2 == pytest.approx(4.0)
    with pytest.raises(ValueError, match="Provide"):
        plasma.spitzer_resistivity(1000)


@pytest.mark.parametrize("bad", [0, float("nan")])  # Ip is signed, so -1 is valid
def test_plasma_functions_reject_invalid_inputs(bad):
    with pytest.raises(ValueError):
        plasma.q95_iter_formula(bad, 5, 6, 2, 1.7)


def test_confinement_summary_bundle_is_self_consistent():
    s = plasma.confinement_summary(ip_A=1e6, bt_T=1.0, r_m=0.9, a_m=0.55, kappa=1.8, ne_line_avg_m3=4e19,
                                   p_loss_W=2e6, stored_energy_J=1.2e5)
    assert s["H98y2"] == pytest.approx(s["tau_E_measured_s"] / s["ipb98y2"]["tau_E_IPB98y2_s"])
    assert s["P_loss_over_P_LH"] == pytest.approx(2.0 / s["lh_threshold"]["P_LH_MW"])


# ---------------- events (analytic ground truth) ----------------
def test_detect_elms_finds_all_injected_bursts_without_false_positives():
    rng = np.random.default_rng(3)
    fs = 20000
    t = np.arange(0, 2, 1 / fs)
    y = 1 + 0.3 * np.sin(2 * np.pi * 0.5 * t) + 0.02 * rng.standard_normal(len(t))
    true = np.cumsum(rng.uniform(0.008, 0.012, 200)); true = true[true < 1.95]
    for e in true:
        m = t >= e
        y[m] += rng.uniform(0.5, 1.5) * np.exp(-(t[m] - e) / 0.0008)
    r = detect_elms(t, y)
    assert r["n_elms"] == len(true)
    assert all(np.min(np.abs(r["event_times_s"] - e)) < 1e-3 for e in true)
    assert r["statistics"]["mean_frequency_Hz"] == pytest.approx(1 / np.mean(np.diff(true)), rel=0.03)
    quiet = detect_elms(t, 1 + 0.02 * np.random.default_rng(9).standard_normal(len(t)))
    assert quiet["n_elms"] == 0


def test_detect_sawteeth_period_and_amplitude():
    rng = np.random.default_rng(4)
    fs, T = 20000, 0.06
    t = np.arange(0, 1, 1 / fs)
    y = (t % T) / T + 0.01 * rng.standard_normal(len(t))
    r = detect_sawteeth(t, y)
    assert abs(r["n_crashes"] - 16) <= 1
    assert r["period_statistics"]["mean_interval_s"] == pytest.approx(T, rel=0.01)
    assert np.median(r["crash_amplitudes"]) == pytest.approx(1.0, rel=0.1)


def test_current_quench_time_matches_construction_and_flat_top_is_not_disrupted():
    t = np.arange(0, 0.3, 1e-5)
    ip = np.full_like(t, 200e3)
    after = t > 0.2
    ip[after] = np.maximum(200e3 * (1 - (t[after] - 0.2) / 0.010), 0)
    r = current_quench_analysis(t, ip)
    assert r["disrupted"] and r["current_quench_time_s"] == pytest.approx(0.010, rel=0.01)
    assert not current_quench_analysis(t[t < 0.19], ip[t < 0.19])["disrupted"]
    with pytest.raises(ValueError, match="fraction"):
        current_quench_analysis(t, ip, upper_fraction=0.2, lower_fraction=0.8)


def test_current_quench_runaway_plateau_is_not_called_complete():
    t = np.arange(0, 0.3, 1e-5)
    ip = np.full_like(t, 200e3)
    k = t > 0.2
    ip[k] = np.maximum(200e3 * (1 - (t[k] - 0.2) / 0.005), 60e3)   # stops at a 30% plateau
    r = current_quench_analysis(t, ip)
    assert r["disrupted"] is False and r["runaway_or_incomplete_quench"] is True


def test_interval_statistics_needs_two_events():
    assert interval_statistics([1.0])["n_events"] == 1
    s = interval_statistics([0, 1, 2, 3])
    assert s["mean_frequency_Hz"] == pytest.approx(1.0) and s["coefficient_of_variation"] == pytest.approx(0.0)


# ---------------- grids, reports, registries ----------------
def test_grid_from_table_parses_tis_efit_export_format():
    R = np.array([0.5, 1.0, 1.5]); Z = np.array([-1.0, 0.0, 1.0])
    frame = pd.DataFrame(np.arange(9.0).reshape(3, 3), columns=[f"R={v:.9g}_m" for v in R])
    frame.insert(0, "z_m", Z[::-1])               # unsorted z must be re-ordered consistently
    g = grid_from_table(frame)
    assert np.allclose(g["R_m"], R) and np.allclose(g["Z_m"], Z)
    assert g["grid"][0, 0] == frame.iloc[2, 1]    # z=-1 row is the last input row
    with pytest.raises(ValueError, match="parse"):
        grid_from_table(pd.DataFrame({"z_m": [0, 1, 2], "a": [1, 2, 3], "b": [1, 2, 3], "c": [1, 2, 3]}))


def test_html_report_escapes_markup_and_embeds_sections():
    html = report.build_html_report({"project": "<script>alert(1)</script>", "analysis_history": [{"name": "x<b>", "created_at_utc": "t"}]},
                                    [{"title": "R<i>", "results": {"k": "<img onerror=1>"}, "figures": ["AAAA"]}])
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert "<img onerror=1>" not in html
    assert "data:image/png;base64,AAAA" in html


def test_registry_json_files_match_catalog_module():
    diag = json.loads((ROOT / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    forms = json.loads((ROOT / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    assert diag == json.loads(json.dumps(DIAGNOSTICS, ensure_ascii=False))
    assert forms == json.loads(json.dumps(FORMULAS, ensure_ascii=False))
    names = [d["diagnostic"] for d in diag]
    assert len(names) == len(set(names)), "duplicate diagnostic entries"


def test_every_mapped_function_actually_exists():
    import tis.core, tis.events, tis.plasma, tis.signals, tis.workflows, tis.line_shape, tis.magnetic, tis.gas_flow, tis.fast_bolometry
    modules = (tis.core, tis.events, tis.plasma, tis.signals, tis.workflows, tis.line_shape, tis.magnetic, tis.gas_flow)
    for d in DIAGNOSTICS:
        for fn in d["implementation_functions"]:
            assert any(hasattr(m, fn) for m in modules), f"{d['diagnostic']} maps to missing function {fn}"


@pytest.mark.parametrize("seed", range(6))
def test_sawtooth_detector_is_stable_across_noise_realisations(seed):
    rng = np.random.default_rng(100 + seed)
    fs, T = 20000, 0.06
    t = np.arange(0, 1, 1 / fs)
    y = (t % T) / T + 0.01 * rng.standard_normal(len(t))
    r = detect_sawteeth(t, y)
    assert abs(r["n_crashes"] - 16) <= 1
    assert r["period_statistics"]["median_interval_s"] == pytest.approx(T, rel=0.01)
    assert r["period_statistics"]["min_interval_s"] > 0.9 * T       # no split/duplicate crashes


def test_hampel_does_not_flag_curvature_of_a_clean_noisy_signal():
    rng = np.random.default_rng(5)
    t = np.linspace(0, 1, 4000)
    y = np.sin(2 * np.pi * 5 * t) + 0.05 * rng.standard_normal(len(t))
    z = y.copy(); z[[700, 2100, 3300]] += 4.0
    assert hampel_despike(z, 11, 6.0)["n_replaced"] == 3
    assert hampel_despike(np.sin(2 * np.pi * 5 * t), 11, 6.0)["n_replaced"] == 0


def test_magnetic_flux_workflow_uses_faraday_sign_and_initial_value():
    from tis.magnetic import magnetic_flux_from_voltage
    t = np.linspace(0, 2, 101)
    result = magnetic_flux_from_voltage(t, np.full_like(t, 3.0), initial_flux_Wb=0.4)
    assert result["magnetic_flux_Wb"][-1] == pytest.approx(-5.6)
    assert result["flux_change_Wb"][-1] == pytest.approx(-6.0)


def test_gas_flow_conversion_and_inventory():
    from tis.gas_flow import gas_flow_to_particle_inventory
    t = np.linspace(0, 60, 61)
    result = gas_flow_to_particle_inventory(t, np.full_like(t, 60.0))
    assert result["integrated_moles_mol"][-1] > 0
    assert result["integrated_particles"][-1] == pytest.approx(
        result["integrated_moles_mol"][-1] * 6.02214076e23
    )


def test_effective_pumping_speed():
    from tis.gas_flow import effective_pumping_speed_m3_s
    assert effective_pumping_speed_m3_s(2.0, 4.0) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        effective_pumping_speed_m3_s(1.0, 0.0)


def test_fast_bolometry_timeseries_reports_known_peak():
    from tis.fast_bolometry import fast_bolometry_timeseries
    fs = 2000.0
    t = np.arange(2048) / fs
    result = fast_bolometry_timeseries(t, np.sin(2 * np.pi * 125 * t))
    assert result["sample_rate_Hz"] == pytest.approx(fs)
    assert result["dominant_nonzero_psd_frequency_Hz"] == pytest.approx(125, abs=2)
