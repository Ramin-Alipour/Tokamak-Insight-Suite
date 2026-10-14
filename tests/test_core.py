import numpy as np
import pandas as pd
import pytest
from tis.core import (
    linear_calibration, polynomial_calibration, prepare_time_series, butterworth_filter,
    fft_spectrum, welch_psd, cross_analysis, cross_correlation_lag, fit_spectral_line,
    doppler_temperature_eV, doppler_velocity_m_s, derive_parameters, uncertainty_linear,
    regularized_tomography, spectral_calibration, probe_iv_summary, signal_quality,
    linear_phase_density, load_table, sha256_bytes,
    reflectometry_cutoff_density, electron_cyclotron_resonance_field,
    axisymmetric_radiated_power, equilibrium_grid_derivatives,
    exponential_heat_flux_fit, integrate_time_series, time_frequency_spectrogram, autocorrelation_analysis,
)
from tis.catalog import DIAGNOSTICS, FORMULAS, diagnostic_families


def test_linear_calibration():
    np.testing.assert_allclose(linear_calibration([0,1,2],2,1),[1,3,5])


def test_polynomial_calibration():
    np.testing.assert_allclose(polynomial_calibration([0,1,2],[1,0,1]),[1,2,5])


def test_prepare_time_series_and_fs():
    t,y,fs=prepare_time_series([0,.001,.002,.003,.004,.005,.006,.007],[0,1,0,-1,0,1,0,-1])
    assert fs==pytest.approx(1000)
    assert len(t)==len(y)==8


def test_prepare_rejects_irregular_sampling():
    with pytest.raises(ValueError,match='Irregular'):
        prepare_time_series([0,1,2,4,5,6,7,8],[1,2,3,4,5,6,7,8])


def test_filter_passes_frequency():
    fs=1000;t=np.arange(2000)/fs;x=np.sin(2*np.pi*20*t)+np.sin(2*np.pi*200*t)
    y=butterworth_filter(x,fs,lowpass=60,order=4)
    assert np.std(y-np.sin(2*np.pi*20*t)) < .2


def test_fft_peak():
    fs=1000;t=np.arange(1000)/fs;x=np.sin(2*np.pi*50*t)
    f,a=fft_spectrum(x,fs)
    assert f[np.argmax(a[1:])+1]==pytest.approx(50)


def test_welch_psd_shape():
    f,p=welch_psd(np.random.default_rng(1).normal(size=256),1000)
    assert len(f)==len(p)>2 and np.all(p>=0)


def test_cross_analysis_coherence_bounds():
    rng=np.random.default_rng(2);x=rng.normal(size=512);y=x+.05*rng.normal(size=512)
    z=cross_analysis(x,y,1000)
    assert np.all((z['coherence']>=0)&(z['coherence']<=1))


def test_cross_correlation_lag():
    rng=np.random.default_rng(3);x=rng.normal(size=300);y=np.r_[np.zeros(4),x[:-4]]
    _,_,lag=cross_correlation_lag(x,y,1000)
    assert abs(abs(lag)-.004)<.001


def test_gaussian_spectral_fit():
    x=np.linspace(654,658,401);y=3+10*np.exp(-.5*((x-656.3)/.12)**2)
    fit=fit_spectral_line(x,y,model='Gaussian')
    assert fit['center']==pytest.approx(656.3,abs=1e-3)
    assert fit['sigma']==pytest.approx(.12,rel=1e-2)
    assert fit['r_squared']>.999


def test_lorentzian_spectral_fit():
    x=np.linspace(-2,2,301);y=2+4/(1+((x-.3)/.2)**2)
    fit=fit_spectral_line(x,y,model='Lorentzian')
    assert fit['center']==pytest.approx(.3,abs=1e-3)
    assert fit['half_width_at_half_max']==pytest.approx(.2,rel=.02)


def test_voigt_spectral_fit():
    x=np.linspace(-2,2,301); from scipy.special import voigt_profile
    y=1+5*voigt_profile(x-.2,.15,.1)/voigt_profile(0,.15,.1)
    fit=fit_spectral_line(x,y,model='Voigt')
    assert fit['center']==pytest.approx(.2,abs=.01)
    assert fit['r_squared']>.99


def test_doppler_temperature_positive():
    assert doppler_temperature_eV(.01,500,40)>0


def test_doppler_velocity():
    assert doppler_velocity_m_s(.001,500)==pytest.approx(599.584916)


def test_derive_parameters():
    d=derive_parameters(1e19,100,80,5e4,1,.7,.2,1,1000,5000)
    assert d['electron_pressure_Pa']==pytest.approx(160.2176634)
    assert d['energy_confinement_time_s']==pytest.approx(.2)
    assert d['greenwald_density_1e20_m-3']>0


def test_uncertainty_propagation():
    val,u=uncertainty_linear([1,2],2,1,.1,.01,.2)
    np.testing.assert_allclose(val,[3,5]); assert np.all(u>0)


def test_regularized_tomography():
    A=np.array([[1,0],[0,1],[1,1]],float); y=np.array([2,3,5],float)
    r=regularized_tomography(A,y,1e-8,True)
    np.testing.assert_allclose(r['solution'],[2,3],atol=1e-3)
    assert r['residual_norm']<1e-3


def test_spectral_calibration():
    r=spectral_calibration([0,1,2],[400,401,402],1)
    assert r['rms_residual']<1e-10
    assert r['coefficients_high_to_low'][0]==pytest.approx(1)


def test_probe_summary_does_not_claim_temperature():
    r=probe_iv_summary(np.linspace(-5,5,11),np.tanh(np.linspace(-5,5,11)))
    assert 'warning' in r and 'Te' in r['warning']


def test_signal_quality_missing():
    r=signal_quality([1,np.nan,3])
    assert r['n_finite']==2 and r['n_missing_or_nonfinite']==1


def test_interferometry_phase_density_finite():
    assert np.isfinite(linear_phase_density(.1,140e9))


def test_load_csv():
    raw=b'time,x\n0,1\n1,2\n'
    d=load_table(raw,'demo.csv')
    assert d.shape==(2,2)


def test_hash_deterministic():
    assert sha256_bytes(b'test')==sha256_bytes(b'test')


def test_catalog_coverage():
    assert len(DIAGNOSTICS)>=60
    assert len(FORMULAS)>=30
    assert len(diagnostic_families())>=10
    assert all({'family','diagnostic','quantities','status','method','caveat'} <= set(d) for d in DIAGNOSTICS)


def test_unsafe_invalid_cutoff_rejected():
    with pytest.raises(ValueError): butterworth_filter(np.arange(100),100,lowpass=50)


def test_tomography_rank_deficient_without_regularization_is_stable():
    A=np.array([[1,1],[2,2]],float); y=np.array([2,4],float)
    r=regularized_tomography(A,y,0,False)
    assert np.all(np.isfinite(r['solution']))
    assert r['residual_norm']<1e-8


def test_synthetic_example_has_expected_mhd_peak():
    from pathlib import Path
    path=Path(__file__).parents[1]/'examples'/'sample_mhd_signal.csv'
    df=pd.read_csv(path); t,y,fs=prepare_time_series(df['time_s'],df['mirnov_ch1_V'])
    f,a=fft_spectrum(y,fs)
    peak=f[1+np.argmax(a[1:])]
    assert peak==pytest.approx(850,abs=3)


def test_catalog_has_distinct_measured_derived_inferred_labels():
    statuses=' '.join(d['status'] for d in DIAGNOSTICS)
    assert 'M' in statuses and 'D' in statuses and 'I' in statuses


def test_langmuir_iv_estimates_floating_potential_and_temperature():
    from tis.core import langmuir_iv_analysis
    v=np.linspace(-10,10,301)
    # Synthetic I-V with an exponential electron-retarding region and negative ion saturation.
    i=np.where(v < -7, -0.01, 0.02*np.exp(np.clip(v/5,-20,5))-0.01)
    r=langmuir_iv_analysis(v,i,electron_fit_range=(-5,0),ion_fit_range=(-10,-7))
    assert r['electron_temperature_eV']==pytest.approx(5,rel=.15)
    assert r['floating_potential_V'] is not None
    assert 'interpretation_warning' in r


def test_mode_number_from_phase():
    from tis.core import mode_number_from_phase
    theta=np.linspace(0,2*np.pi,8,endpoint=False)
    phase=3*theta+.2
    r=mode_number_from_phase(theta,phase)
    assert r['mode_number_fit']==pytest.approx(3,abs=1e-10)


def test_fluctuation_flux_covariance():
    from tis.core import fluctuation_flux
    n=np.array([1.,2.,3.,4.]); vr=np.array([-1.,-1.,1.,1.])
    r=fluctuation_flux(n,vr)
    assert r['particle_flux_covariance_m-2_s-1']==pytest.approx(1.0)


def test_heat_flux_exponential_fit():
    from tis.core import exponential_heat_flux_fit
    x=np.linspace(0,.05,80); y=1000*np.exp(-x/.01)+10
    r=exponential_heat_flux_fit(x,y)
    assert r['lambda_q_m']==pytest.approx(.01,rel=.02)


def test_abel_reconstruction_shapes():
    from tis.core import abel_reconstruction
    r=abel_reconstruction([0,.02,.04,.06],[1,.8,.5,.2],[0,.02,.04,.06,.08],lam=.01)
    assert len(r['radius_centers_m'])==4
    assert len(r['emissivity_or_density_profile'])==4
    assert np.all(np.isfinite(r['emissivity_or_density_profile']))


def test_axisymmetric_radiated_power_positive():
    from tis.core import axisymmetric_radiated_power
    R=np.linspace(1,2,10); Z=np.linspace(-.5,.5,8); eps=np.ones((len(Z),len(R)))
    r=axisymmetric_radiated_power(R,Z,eps)
    assert r['total_radiated_power_W']>0


def test_profile_gradient_analysis():
    from tis.core import profile_gradient_analysis
    x=np.linspace(-1,1,101); y=np.exp(-x*x)
    r=profile_gradient_analysis(x,y)
    assert r['peak_value']==pytest.approx(1)
    assert r['FWHM_coordinate_units']>0


def test_triple_probe_temperature():
    from tis.core import triple_probe_temperature_eV
    assert triple_probe_temperature_eV(np.log(2))['electron_temperature_estimate_eV']==pytest.approx(1)


def test_reflectometry_o_mode_cutoff_density():
    from tis.core import reflectometry_cutoff_density
    r=reflectometry_cutoff_density(60e9)
    assert r['cutoff_density_m-3']>0
    with pytest.raises(ValueError): reflectometry_cutoff_density(60e9,'X')


def test_electron_cyclotron_resonance_field():
    from tis.core import electron_cyclotron_resonance_field
    r=electron_cyclotron_resonance_field(110e9,2)
    assert 1.9 < r['resonance_field_T'] < 2.1


def test_corrected_line_ratio():
    from tis.core import corrected_line_ratio
    r=corrected_line_ratio(100,50,2,1)
    assert r['instrument_response_corrected_ratio']==pytest.approx(1)

def test_fft_detrend_false_preserves_dc_component():
    x = np.full(256, 3.0)
    f, amplitude = fft_spectrum(x, fs=1000, detrend=False, window='boxcar')
    assert amplitude[0] == pytest.approx(3.0, abs=1e-12)
    assert np.max(amplitude[1:]) < 1e-12


def test_fft_detrend_true_removes_dc_component():
    x = np.full(256, 3.0)
    f, amplitude = fft_spectrum(x, fs=1000, detrend=True, window='boxcar')
    assert np.max(amplitude) < 1e-12


@pytest.mark.parametrize(
    "kwargs",
    [{"lowpass": 0}, {"lowpass": -1}, {"highpass": 0}, {"highpass": -1},
     {"lowpass": float("nan")}, {"highpass": float("inf")}],
)
def test_filter_rejects_explicit_nonpositive_or_nonfinite_cutoff(kwargs):
    with pytest.raises(ValueError):
        butterworth_filter(np.linspace(0, 1, 100), fs=100, **kwargs)

def test_o_mode_reflectometry_cutoff_matches_known_28_ghz_scale():
    result = reflectometry_cutoff_density(28e9, mode="O")
    assert result["cutoff_density_m-3"] == pytest.approx(9.72e18, rel=0.02)


def test_fundamental_electron_cyclotron_field_at_28_ghz():
    result = electron_cyclotron_resonance_field(28e9, harmonic=1)
    assert result["resonance_field_T"] == pytest.approx(1.001, rel=0.01)


def test_axisymmetric_radiated_power_constant_emissivity():
    R = np.linspace(1.0, 2.0, 101)
    Z = np.linspace(-0.5, 0.5, 81)
    eps = np.full((len(Z), len(R)), 4.0)
    result = axisymmetric_radiated_power(R, Z, eps)
    # Integral of 2*pi*R*eps over R in [1,2] and Z in [-0.5,0.5].
    assert result["total_radiated_power_W"] == pytest.approx(12 * np.pi, rel=2e-4)


def test_equilibrium_grid_derivatives_of_quadratic_flux():
    R = np.linspace(1.0, 2.0, 21)
    Z = np.linspace(-0.5, 0.5, 21)
    psi = Z[:, None] ** 2 + R[None, :] ** 2
    result = equilibrium_grid_derivatives(R, Z, psi, Bphi_T=1.5)
    np.testing.assert_allclose(result["dpsi_dR"][:, 1:-1], np.broadcast_to(2 * R[None, 1:-1], (len(Z), len(R) - 2)), atol=1e-10)
    np.testing.assert_allclose(result["dpsi_dZ"][1:-1, :], np.broadcast_to(2 * Z[1:-1, None], (len(Z) - 2, len(R))), atol=1e-10)
    assert np.all(np.isfinite(result["Btotal_T"]))


def test_exponential_heat_flux_fit_recovers_synthetic_decay_length():
    x = np.linspace(0.0, 0.1, 100)
    y = 5e6 * np.exp(-x / 0.02) + 2e5
    result = exponential_heat_flux_fit(x, y)
    assert result["lambda_q_m"] == pytest.approx(0.02, rel=0.02)
    assert result["q0_W_m-2"] == pytest.approx(5e6, rel=0.02)
    assert result["r_squared"] > 0.999

@pytest.mark.parametrize("degree", [0, -1, 1.5, True])
def test_spectral_calibration_rejects_invalid_degree(degree):
    with pytest.raises(ValueError, match="degree"):
        spectral_calibration([0, 1, 2, 3], [400, 401, 402, 403], degree=degree)


def test_spectral_calibration_rejects_mismatched_arrays():
    with pytest.raises(ValueError, match="equal length"):
        spectral_calibration([0, 1, 2], [400, 401], degree=1)

def test_axisymmetric_radiated_power_rejects_negative_emissivity():
    R = np.linspace(1.0, 2.0, 5)
    Z = np.linspace(-0.5, 0.5, 5)
    eps = np.ones((len(Z), len(R)))
    eps[2, 2] = -0.1
    with pytest.raises(ValueError, match="nonnegative"):
        axisymmetric_radiated_power(R, Z, eps)

def test_integrate_time_series_recovers_linear_integral():
    t = np.linspace(0.0, 1.0, 101)
    y = np.full_like(t, 2.0)
    result = integrate_time_series(t, y, scale=3.0, initial_value=1.0)
    np.testing.assert_allclose(result["integrated_signal"], 1.0 + 6.0 * t, atol=1e-12)


def test_integrate_time_series_baseline_correction_removes_constant_offset():
    t = np.linspace(0.0, 1.0, 101)
    y = np.full_like(t, 4.0)
    result = integrate_time_series(t, y, baseline_start_s=0.0, baseline_end_s=0.2)
    np.testing.assert_allclose(result["integrated_signal"], 0.0, atol=1e-12)
    assert result["baseline_offset_in_calibrated_units"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    "time, values, kwargs",
    [
        ([0, 1, 1], [1, 2, 3], {}),
        ([0, 1], [1, np.nan], {}),
        ([0, 1], [1, 2], {"baseline_start_s": 0.0}),
        ([0, 1], [1, 2], {"baseline_start_s": 1.0, "baseline_end_s": 0.0}),
    ],
)
def test_integrate_time_series_rejects_invalid_inputs(time, values, kwargs):
    with pytest.raises(ValueError):
        integrate_time_series(time, values, **kwargs)

def test_time_frequency_spectrogram_detects_stationary_tone():
    fs = 1000.0
    t = np.arange(4096) / fs
    x = np.sin(2 * np.pi * 125 * t)
    result = time_frequency_spectrogram(x, fs, nperseg=256)
    assert result["psd"].shape == (len(result["frequency_Hz"]), len(result["time_s"]))
    peak = result["frequency_Hz"][np.argmax(np.mean(result["psd"], axis=1))]
    assert peak == pytest.approx(125.0, abs=4.0)
    assert np.all(np.isfinite(result["psd"])) and np.all(result["psd"] >= 0)


@pytest.mark.parametrize(
    "kwargs",
    [{"nperseg": 7}, {"nperseg": 8, "noverlap": 8}, {"nperseg": 8, "noverlap": -1}],
)
def test_time_frequency_spectrogram_rejects_invalid_window_settings(kwargs):
    with pytest.raises(ValueError):
        time_frequency_spectrogram(np.arange(64, dtype=float), fs=1000, **kwargs)

def test_autocorrelation_detects_sinusoidal_period_scale():
    fs = 1000.0
    t = np.arange(4000) / fs
    x = 1e-8 * np.sin(2 * np.pi * 20 * t)
    result = autocorrelation_analysis(x, fs, max_lag_s=0.1)
    assert result["lag_s"][0] == 0.0
    assert result["autocorrelation"][0] == pytest.approx(1.0)
    assert result["e_folding_time_s"] is not None
    assert len(result["lag_s"]) == len(result["autocorrelation"])


def test_autocorrelation_rejects_constant_signal():
    with pytest.raises(ValueError, match="constant"):
        autocorrelation_analysis(np.ones(100), fs=1000)


def test_langmuir_exact_zero_current_uses_sample_without_division_by_zero():
    from tis.core import langmuir_iv_analysis
    v=np.linspace(-4,4, nine:=9)
    i=np.array([-4.,-3.,-2.,-1.,0.,0.,1.,2.,3.])
    r=langmuir_iv_analysis(v,i)
    assert r["floating_potential_V"] == pytest.approx(0.0)


def test_langmuir_requires_area_and_ion_mass_together():
    from tis.core import langmuir_iv_analysis
    v=np.linspace(-4,4,9)
    i=np.linspace(-1,1,9)
    with pytest.raises(ValueError, match="supplied together"):
        langmuir_iv_analysis(v,i,area_m2=1e-6)


@pytest.mark.parametrize("kwargs", [
    {"area_m2": 0.0, "ion_mass_amu": 2.0},
    {"area_m2": 1e-6, "ion_mass_amu": float("nan")},
    {"ti_eV": -1.0},
    {"sheath_factor": 0.0},
    {"gamma_i": -1.0},
    {"electron_fit_range": (float("nan"), 1.0)},
])
def test_langmuir_rejects_nonphysical_model_inputs(kwargs):
    from tis.core import langmuir_iv_analysis
    v=np.linspace(-4,4,9)
    i=np.linspace(-1,1,9)
    with pytest.raises(ValueError):
        langmuir_iv_analysis(v,i,**kwargs)


def test_langmuir_rejects_requested_ion_fit_with_too_few_samples():
    from tis.core import langmuir_iv_analysis
    v=np.linspace(-4,4,9)
    i=np.linspace(-1,1,9)
    with pytest.raises(ValueError, match="at least 3 valid samples"):
        langmuir_iv_analysis(v,i,ion_fit_range=(100,101))


def test_spectral_fit_reports_parameter_uncertainties():
    x=np.linspace(500,501,301)
    y=2.0+10.0*np.exp(-0.5*((x-500.42)/0.07)**2)
    r=fit_spectral_line(x,y,model="Gaussian")
    for key in ("standard_error_center","standard_error_sigma","standard_error_FWHM"):
        assert np.isfinite(r[key])
        assert r[key] >= 0
    assert r["center"] == pytest.approx(500.42, abs=1e-5)


@pytest.mark.parametrize("x,y,xmin,xmax", [
    ([1,1,1,1,1,1,1,1], [1,2,3,4,5,6,7,8], None, None),
    ([1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8], 5.0, 2.0),
    ([1,2,3,4,5,6,7,8], [1,2,3,4,5,6,7,8], float("nan"), None),
])
def test_spectral_fit_rejects_invalid_coordinates_and_limits(x,y,xmin,xmax):
    with pytest.raises(ValueError):
        fit_spectral_line(x,y,xmin=xmin,xmax=xmax,model="Gaussian")


def test_doppler_temperature_uncertainty_propagation():
    from tis.core import doppler_temperature_with_uncertainty_eV
    r=doppler_temperature_with_uncertainty_eV(0.02,500.0,40.0,0.001,0.01,0.1)
    expected=doppler_temperature_eV(0.02,500.0,40.0)
    assert r["temperature_eV"] == pytest.approx(expected)
    assert r["standard_uncertainty_eV"] > 0
    assert r["relative_standard_uncertainty"] == pytest.approx(
        np.sqrt((2*0.001/0.02)**2+(2*0.01/500.0)**2+(0.1/40.0)**2)
    )


def test_doppler_temperature_uncertainty_rejects_negative_uncertainty():
    from tis.core import doppler_temperature_with_uncertainty_eV
    with pytest.raises(ValueError):
        doppler_temperature_with_uncertainty_eV(0.02,500.0,40.0,sigma_sigma_lambda=-0.1)


def test_doppler_temperature_uncertainty_rejects_undefined_zero_width_case():
    from tis.core import doppler_temperature_with_uncertainty_eV
    with pytest.raises(ValueError, match="zero Doppler width"):
        doppler_temperature_with_uncertainty_eV(0.0,500.0,40.0,sigma_sigma_lambda=0.001)
