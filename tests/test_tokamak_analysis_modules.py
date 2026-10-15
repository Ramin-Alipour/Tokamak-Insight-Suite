"""Smoke tests for the modular tokamak diagnostic analysis workflows."""
import importlib
import numpy as np
import pytest
import tokamak_analysis


@pytest.mark.parametrize("module_name", tokamak_analysis.__all__)
def test_all_analysis_modules_import(module_name):
    importlib.import_module(f"tokamak_analysis.{module_name}")


def test_thomson_profile_workflow():
    from tokamak_analysis.thomson_workflow import analyze_thomson_profile
    r=np.array([0.0,0.1,0.2,0.3])
    out=analyze_thomson_profile(r,2e19*(1-r),100*(1-.5*r))
    np.testing.assert_allclose(out["density_gradient_m-4"],-2e19,rtol=1e-12)
    np.testing.assert_allclose(out["temperature_gradient_eV_m-1"],-50,rtol=1e-12)


def test_beta_and_confinement():
    from tokamak_analysis.beta import analyze_beta
    from tokamak_analysis.confinement import energy_confinement_time
    assert analyze_beta(1e3,2.0)["beta"] > 0
    assert energy_confinement_time(1000,500)["energy_confinement_time_s"] == pytest.approx(2.0)


def test_doppler_temperature_and_flow():
    from tokamak_analysis.doppler_temperature import ion_temperature_from_doppler_fwhm
    from tokamak_analysis.doppler_flow import doppler_flow_velocity
    assert ion_temperature_from_doppler_fwhm(.01,500,40)["ion_temperature_eV"] > 0
    assert doppler_flow_velocity(500.001,500)["velocity_m_s"] > 0


def test_plasma_geometry_closed_circle():
    from tokamak_analysis.plasma_geometry import geometry_from_boundary
    th=np.linspace(0,2*np.pi,501)
    r=1.5+.2*np.cos(th); z=.2*np.sin(th)
    out=geometry_from_boundary(r,z)
    assert out["cross_section_area_m2"] == pytest.approx(np.pi*.2**2,rel=2e-3)
    assert out["toroidal_volume_m3"] > 0


def test_interferometry_line_average_density():
    from tokamak_analysis.interferometry import interferometry_density
    out=interferometry_density(1e-3,140e9,path_length_m=1.0)
    assert out["line_average_density_m-3"] > 0


def test_signal_synchronization():
    from tokamak_analysis.channel_synchronization import synchronize_channels
    t1=np.linspace(0,1,11); t2=np.linspace(.1,.9,9)
    out=synchronize_channels([t1,t2],[t1,t2])
    assert out["synchronized_signals"].shape[0] == 2
    assert np.all(np.diff(out["time_s"])>0)


def test_denoising_shape_and_finite():
    from tokamak_analysis.denoising import denoise_signal
    x=np.sin(np.linspace(0,4*np.pi,101))
    out=denoise_signal(x,method="savgol",window_length=9)
    assert out["filtered_signal"].shape==x.shape
    assert np.all(np.isfinite(out["filtered_signal"]))


def test_reaction_rate_and_neutral_gas():
    from tokamak_analysis.fusion_rate import fusion_reaction_rate
    from tokamak_analysis.neutral_gas import neutral_density
    assert fusion_reaction_rate(1e19,1e19,1e-22)["reaction_rate_density_m-3_s-1"] > 0
    assert neutral_density(1e-3,300)["neutral_density_m-3"] > 0


def test_invalid_input_rejected():
    from tokamak_analysis.fft_analysis import analyze_fft
    with pytest.raises(ValueError):
        analyze_fft([0,1,2,4],[1,2,3,4])


def test_analysis_registry_resolves_every_callable():
    from tokamak_analysis.analysis_registry import ANALYSES, resolve_analysis
    assert len(ANALYSES) == 80
    assert [x["id"] for x in ANALYSES] == list(range(1, 81))
    for entry in ANALYSES:
        assert callable(resolve_analysis(entry["id"]))
