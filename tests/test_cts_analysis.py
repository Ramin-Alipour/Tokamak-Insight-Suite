import numpy as np
import pytest
from tokamak_analysis.cts_analysis import analyze_cts_spectrum, cts_spectral_density

def _case():
    f0=140e9; f=np.unique(np.r_[np.linspace(-7e9,-.8e9,60),np.linspace(-.8e9,.8e9,401),np.linspace(.8e9,7e9,60)]); g={"scattering_angle_rad":np.deg2rad(110.)}
    ions=[{"name":"D+","charge_state":1,"mass_amu":2.014}]
    return f0,f,g,ions,[1.],np.eye(f.size)

def test_cts_forward_model_is_finite_positive_and_geometry_sensitive():
    f0,f,g,ions,frac,_=_case()
    a=cts_spectral_density(f,f0,g,electron_density_m3=2e19,electron_temperature_eV=1200,ion_temperature_eV=800,ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0)
    b=cts_spectral_density(f,f0,{"scattering_angle_rad":np.deg2rad(70.)},electron_density_m3=2e19,electron_temperature_eV=1200,ion_temperature_eV=800,ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0)
    assert a.shape==f.shape and np.all(np.isfinite(a)) and np.all(a>=0) and not np.allclose(a,b,rtol=1e-8,atol=0.0)

def test_cts_inversion_recovers_controlled_synthetic_parameters():
    f0,f,g,ions,frac,response=_case(); truth=(2e19,1200.,800.,1.5e4)
    z=cts_spectral_density(f,f0,g,electron_density_m3=truth[0],electron_temperature_eV=truth[1],ion_temperature_eV=truth[2],ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,ion_velocity_projection_m_s=truth[3])
    y=2.4e4*z+3.
    out=analyze_cts_spectrum(f,y,probe_frequency_Hz=f0,scattering_geometry=g,ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,instrument_response=response,
      signal_uncertainty=np.full_like(y,.02*np.max(y)),electron_density_initial_m3=truth[0],electron_temperature_initial_eV=truth[1],
      ion_temperature_initial_eV=truth[2],ion_velocity_initial_m_s=truth[3],global_search=False)
    assert out["electron_density_m3"]==pytest.approx(truth[0],rel=.08)
    assert out["electron_temperature_eV"]==pytest.approx(truth[1],rel=.08)
    assert out["ion_temperature_eV"]==pytest.approx(truth[2],rel=.08)
    assert out["ion_velocity_projection_m_s"]==pytest.approx(truth[3],rel=.08)
    assert out["electron_velocity_projection_m_s"] == 0.0
    assert out["electron_velocity_projection_is_fitted"] is False
    assert out["electron_velocity_projection_input_uncertainty_m_s"] is None

def test_cts_hot_ion_component_forward_model():
    f0,f,g,ions,frac,_=_case()
    z=cts_spectral_density(f,f0,g,electron_density_m3=2e19,electron_temperature_eV=1200,ion_temperature_eV=800,
      ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,fast_ion={"species_index":0,"fraction":.08,"temperature_eV":12000})
    assert np.all(np.isfinite(z)) and np.all(z>0)

@pytest.mark.parametrize("g",[{},{"scattering_angle_rad":0.},{"scattering_angle_rad":4.}])
def test_cts_rejects_invalid_geometry(g):
    f0,f,_,ions,frac,_=_case()
    with pytest.raises(ValueError):
        cts_spectral_density(f,f0,g,electron_density_m3=2e19,electron_temperature_eV=1200,ion_temperature_eV=800,ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0)


def test_cts_tabulated_non_maxwellian_fast_ion_forward_model():
    f0,f,g,ions,frac,_=_case()
    velocity=np.linspace(-1.2e6,1.2e6,81)
    pdf=np.exp(-0.5*(velocity/2.5e5)**2)*(1.0+0.35*np.exp(-0.5*((velocity-5e5)/1.2e5)**2))
    spectrum=cts_spectral_density(
        f,f0,g,electron_density_m3=2e19,electron_temperature_eV=1200,
        ion_temperature_eV=800,ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,
        fast_ion={"species_index":0,"fraction":0.08,"kind":"tabulated_1d",
                  "velocity_m_s":velocity,"probability_density_m_s_inverse":pdf},
    )
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum>=0)


def test_cts_improved_genetic_search_is_deterministic_and_bounded():
    from tokamak_analysis.cts_analysis import _improved_genetic_search

    objective=lambda p: float((p[0]-0.25)**2+(p[1]+0.4)**2)
    bounds=[(-1.0,1.0),(-1.0,1.0)]
    first=_improved_genetic_search(objective,bounds,seed=11,population_size=20,generations=25)
    second=_improved_genetic_search(objective,bounds,seed=11,population_size=20,generations=25)
    np.testing.assert_allclose(first[0],second[0])
    assert first[1] < 0.05
    assert np.all(first[0]>=np.array([-1.0,-1.0]))
    assert np.all(first[0]<=np.array([1.0,1.0]))


def test_cts_correlated_frequency_covariance_is_accepted_and_reported():
    f0,f,g,ions,frac,response=_case()
    truth=(2e19,1200.,800.,1.5e4)
    z=cts_spectral_density(f,f0,g,electron_density_m3=truth[0],electron_temperature_eV=truth[1],
        ion_temperature_eV=truth[2],ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,
        ion_velocity_projection_m_s=truth[3])
    y=2.4e4*z+3.
    covariance=np.eye(f.size)*(0.02*np.max(y))**2
    out=analyze_cts_spectrum(
        f,y,probe_frequency_Hz=f0,scattering_geometry=g,ion_species=ions,
        ion_fractions=frac,electron_velocity_projection_m_s=0.0,instrument_response=response,spectral_covariance=covariance,
        electron_density_initial_m3=truth[0],electron_temperature_initial_eV=truth[1],
        ion_temperature_initial_eV=truth[2],ion_velocity_initial_m_s=truth[3],
        global_search=False,
    )
    assert out["reduced_chi_square"] is not None
    assert out["inversion_method"]=="differential_evolution"


def test_cts_regularized_inversion_requires_explicit_vdf_regularization_target():
    f0,f,g,ions,frac,response=_case()
    truth=(2e19,1200.,800.,1.5e4)
    z=cts_spectral_density(f,f0,g,electron_density_m3=truth[0],electron_temperature_eV=truth[1],
        ion_temperature_eV=truth[2],ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,
        ion_velocity_projection_m_s=truth[3])
    y=2.4e4*z+3.
    with pytest.raises(ValueError,match="requires regularization_strength"):
        analyze_cts_spectrum(f,y,probe_frequency_Hz=f0,scattering_geometry=g,ion_species=ions,
            ion_fractions=frac,electron_velocity_projection_m_s=0.0,instrument_response=response,inversion_method="regularized",
            global_search=False)
    with pytest.raises(ValueError,match="requires fast_ion.kind='infer_tabulated_1d'"):
        analyze_cts_spectrum(f,y,probe_frequency_Hz=f0,scattering_geometry=g,ion_species=ions,
            ion_fractions=frac,electron_velocity_projection_m_s=0.0,instrument_response=response,inversion_method="regularized",
            regularization_strength=0.01,global_search=False,
            electron_density_initial_m3=truth[0],electron_temperature_initial_eV=truth[1],
            ion_temperature_initial_eV=truth[2],ion_velocity_initial_m_s=truth[3])


def test_cts_unknown_fast_ion_kind_is_rejected():
    f0,f,g,ions,frac,response=_case()
    with pytest.raises(ValueError,match="fast_ion.kind must be"):
        analyze_cts_spectrum(
            f,np.ones_like(f),probe_frequency_Hz=f0,scattering_geometry=g,
            ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,
            instrument_response=response,fast_ion={"species_index":0,"kind":"not-a-model"},
            global_search=False,
        )


def test_cts_response_and_external_geometry_uncertainties_are_monte_carlo_propagated():
    f0,f,g,ions,frac,response=_case()
    truth=(2e19,1200.,800.,1.5e4)
    z=cts_spectral_density(f,f0,g,electron_density_m3=truth[0],electron_temperature_eV=truth[1],
        ion_temperature_eV=truth[2],ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,
        ion_velocity_projection_m_s=truth[3])
    y=2.4e4*z+3.
    out=analyze_cts_spectrum(
        f,y,probe_frequency_Hz=f0,scattering_geometry=g,ion_species=ions,
        ion_fractions=frac,electron_velocity_projection_m_s=0.0,instrument_response=response,
        electron_density_initial_m3=truth[0],electron_temperature_initial_eV=truth[1],
        ion_temperature_initial_eV=truth[2],ion_velocity_initial_m_s=truth[3],
        global_search=False,instrument_response_uncertainty=np.zeros_like(response),
        external_parameter_uncertainties={"scattering_angle_rad":1e-4,"electron_velocity_projection_m_s":50.0},
        response_mc_samples=20,
    )
    assert out["response_external_mc_successes"] >= 12
    assert out["response_uncertainty_propagated"] is True
    assert out["external_parameter_uncertainty_propagated"] is True
    assert out["fit_covariance"].shape[0]==4


def test_cts_analysis_can_fit_fraction_of_fixed_tabulated_fast_ion_vdf():
    f0,f,g,ions,frac,response=_case()
    velocity=np.linspace(-1.2e6,1.2e6,81)
    pdf=np.exp(-0.5*(velocity/2.5e5)**2)*(1.0+0.35*np.exp(-0.5*((velocity-5e5)/1.2e5)**2))
    truth={"species_index":0,"fraction":0.08,"kind":"tabulated_1d",
           "velocity_m_s":velocity,"probability_density_m_s_inverse":pdf}
    z=cts_spectral_density(f,f0,g,electron_density_m3=2e19,electron_temperature_eV=1200,
        ion_temperature_eV=800,ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,
        ion_velocity_projection_m_s=1.5e4,fast_ion=truth)
    y=2.4e4*z+3.
    out=analyze_cts_spectrum(
        f,y,probe_frequency_Hz=f0,scattering_geometry=g,ion_species=ions,
        ion_fractions=frac,electron_velocity_projection_m_s=0.0,instrument_response=response,
        electron_density_initial_m3=2e19,electron_temperature_initial_eV=1200,
        ion_temperature_initial_eV=800,ion_velocity_initial_m_s=1.5e4,
        fast_ion={**truth,"fraction_initial":0.08,"fraction_bounds":(0.01,0.2)},
        global_search=False,
    )
    assert out["fast_ion"]["fraction"]==pytest.approx(0.08,rel=0.15)
    assert out["fast_ion_vdf_provenance"].startswith("externally supplied")
    assert "fast_ion_temperature_eV" not in out["parameter_names"]


def test_cts_regularized_tabulated_vdf_inversion_reconstructs_nonnegative_normalized_shape():
    f0,f,g,ions,frac,response=_case()
    velocity=np.linspace(-2.0e6,2.0e6,41)
    initial_pdf=np.exp(-0.5*(velocity/2.8e5)**2)*(1.0+0.2*np.exp(-0.5*((velocity-3e5)/1.5e5)**2))
    truth={"species_index":0,"fraction":0.06,"kind":"tabulated_1d",
           "velocity_m_s":velocity,"probability_density_m_s_inverse":initial_pdf}
    z=cts_spectral_density(f,f0,g,electron_density_m3=2e19,electron_temperature_eV=1200,
        ion_temperature_eV=800,ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,
        ion_velocity_projection_m_s=1.5e4,fast_ion=truth)
    y=2.4e4*z+3.
    out=analyze_cts_spectrum(
        f,y,probe_frequency_Hz=f0,scattering_geometry=g,ion_species=ions,
        ion_fractions=frac,electron_velocity_projection_m_s=0.0,instrument_response=response,
        electron_density_initial_m3=2e19,electron_temperature_initial_eV=1200,
        ion_temperature_initial_eV=800,ion_velocity_initial_m_s=1.5e4,
        fast_ion={"species_index":0,"kind":"infer_tabulated_1d",
                  "velocity_m_s":velocity,"initial_probability_density_m_s_inverse":initial_pdf,
                  "fraction_initial":0.06,"fraction_bounds":(0.01,0.15),"smoothness":0.5},
        inversion_method="regularized",regularization_strength=0.01,global_search=False,
    )
    pdf=out["fast_ion_vdf_probability_density_m_s_inverse"]
    assert np.all(pdf>=0)
    assert np.trapezoid(pdf,velocity)==pytest.approx(1.0,rel=1e-6)
    assert out["fast_ion_vdf_covariance"].shape==(velocity.size,velocity.size)
    assert out["fast_ion_vdf_uncertainty_m_s_inverse"].shape==velocity.shape
    assert "fast_ion_vdf_logit_0" in out["parameter_names"]
    assert out["regularization_scope"] == "fast-ion VDF smoothness only; no plasma-parameter prior"


def test_cts_requires_an_explicit_electron_flow_projection():
    f0, f, g, ions, frac, _ = _case()
    with pytest.raises(ValueError, match="electron_velocity_projection_m_s must be supplied explicitly"):
        cts_spectral_density(
            f, f0, g, electron_density_m3=2e19,
            electron_temperature_eV=1200, ion_temperature_eV=800,
            ion_species=ions, ion_fractions=frac,
        )
    with pytest.raises(ValueError, match="electron_velocity_projection_m_s is required"):
        analyze_cts_spectrum(
            f, np.ones_like(f), probe_frequency_Hz=f0,
            scattering_geometry=g, ion_species=ions, ion_fractions=frac,
            instrument_response=np.eye(f.size),
        )



def test_cts_forward_model_rejects_unknown_fast_ion_kind():
    f0,f,g,ions,frac,_=_case()
    with pytest.raises(ValueError,match="fast_ion.kind must be maxwellian or tabulated_1d"):
        cts_spectral_density(
            f,f0,g,electron_density_m3=2e19,electron_temperature_eV=1200,
            ion_temperature_eV=800,ion_species=ions,ion_fractions=frac,
            electron_velocity_projection_m_s=0.0,
            fast_ion={"species_index":0,"fraction":0.1,"kind":"not-a-model","temperature_eV":10000},
        )


def test_full_em_cts_returns_hermitian_density_current_field_covariance():
    from tokamak_analysis.cts_full_em import full_em_cts_spectral_density

    shifts = np.array([0.8e9, 1.0e9, 1.2e9])
    result = full_em_cts_spectral_density(
        shifts, 140e9,
        {"incident_direction": [1.0, 0.0, 0.0],
         "scattered_direction": [np.cos(1.8), np.sin(1.8), 0.0]},
        electron_density_m3=2e19, electron_temperature_eV=1200.0,
        ion_temperature_eV=800.0,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], magnetic_field_T=[0.0, 0.0, 2.0],
        electron_flow_velocity_m_s=[0.0, 0.0, 0.0],
        ion_flow_velocity_m_s={"D+": [0.0, 0.0, 0.0]},
        incident_polarization=[0.0, 0.0, 1.0],
        scattered_polarization=[0.0, 0.0, 1.0],
        harmonic_order=3, quadrature_order=4,
    )
    covariance = result["fluctuation_covariance_density_current_E_B"]
    assert result["spectral_density"].shape == shifts.shape
    assert np.all(np.isfinite(result["spectral_density"]))
    assert np.all(result["spectral_density"] >= 0.0)
    assert covariance.shape == (shifts.size, 10, 10)
    for matrix in covariance:
        np.testing.assert_allclose(matrix, matrix.conj().T, rtol=1e-8, atol=1e-20)
        assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-12 * max(np.max(np.abs(matrix)), 1.0)


def test_full_em_cts_requires_complete_vector_flow_and_polarization_metadata():
    from tokamak_analysis.cts_full_em import full_em_cts_spectral_density

    with pytest.raises(ValueError, match="electron_flow_velocity_m_s"):
        full_em_cts_spectral_density(
            [1e9], 140e9,
            {"incident_direction": [1, 0, 0], "scattered_direction": [0, 1, 0]},
            electron_density_m3=2e19, electron_temperature_eV=1200,
            ion_temperature_eV=800,
            ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
            ion_fractions=[1.0], magnetic_field_T=[0, 0, 2],
            electron_flow_velocity_m_s=None, ion_flow_velocity_m_s={"D+": [0, 0, 0]},
            incident_polarization=[0, 0, 1], scattered_polarization=[0, 0, 1],
        )


def test_cts_cold_plasma_dielectric_vacuum_limit_and_hermiticity():
    from tokamak_analysis.cts_em_geometry import cold_plasma_dielectric_tensor

    epsilon = cold_plasma_dielectric_tensor(2.0e11, [0.0, 0.0, 2.0], [])
    np.testing.assert_allclose(epsilon, np.eye(3), rtol=0.0, atol=1e-14)
    species = [
        {"density_m3": 2e19, "charge_C": -1.602176634e-19, "mass_kg": 9.1093837015e-31},
        {"density_m3": 2e19, "charge_C": 1.602176634e-19, "mass_kg": 3.3435837724e-27},
    ]
    epsilon = cold_plasma_dielectric_tensor(2.0e11, [0.3, -0.4, 1.2], species)
    np.testing.assert_allclose(epsilon, epsilon.conj().T, rtol=1e-12, atol=1e-12)


def test_cts_cold_plasma_vacuum_modes_and_gaussian_beam_overlap():
    from tokamak_analysis.cts_em_geometry import (
        solve_cold_plasma_modes, gaussian_beam_overlap,
    )

    omega = 2.0 * np.pi * 140e9
    k = np.array([omega / 299792458.0, 0.0, 0.0])
    modes = solve_cold_plasma_modes(omega, k, [0.0, 0.0, 2.0], [])
    np.testing.assert_allclose(modes["dispersion_eigenvalues"][:2], 0.0, atol=1e-12)
    overlap = gaussian_beam_overlap(
        [0, 0, 0], np.eye(3), [0, 0, 0], np.eye(3)
    )
    assert overlap == pytest.approx(1.0 / (8.0 * np.pi ** 1.5), rel=1e-12)
    displaced = gaussian_beam_overlap(
        [0, 0, 0], np.eye(3), [10, 0, 0], np.eye(3)
    )
    assert displaced < overlap


def test_cts_cold_plasma_ray_trace_is_straight_in_vacuum():
    from tokamak_analysis.cts_em_geometry import trace_cold_plasma_ray

    omega = 2.0 * np.pi * 140e9
    k0 = np.array([omega / 299792458.0, 0.0, 0.0])
    result = trace_cold_plasma_ray(
        [0.0, 0.0, 0.0], k0, omega,
        equilibrium=lambda position: {
            "magnetic_field_T": [0.0, 0.0, 2.0], "species": []
        },
        mode_index=0, path_parameter_max=1e-3, sample_count=8,
    )
    assert result["position_m"].shape == (8, 3)
    assert np.all(np.isfinite(result["wavevector_rad_m"]))
    np.testing.assert_allclose(result["wavevector_rad_m"], k0, rtol=1e-7, atol=1e-7)
