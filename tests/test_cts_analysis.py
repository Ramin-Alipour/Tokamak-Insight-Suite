import numpy as np
import pytest
from tokamak_analysis.cts_analysis import analyze_cts_spectrum, cts_spectral_density

def _case():
    f0=140e9; f=np.unique(np.r_[np.linspace(-7e9,-.8e9,60),np.linspace(-.8e9,.8e9,401),np.linspace(.8e9,7e9,60)]); g={"scattering_angle_rad":np.deg2rad(110.)}
    ions=[{"name":"D+","charge_state":1,"mass_amu":2.014}]
    return f0,f,g,ions,[1.],np.eye(f.size)

def test_cts_inversion_requires_explicit_optimizer_initial_estimates():
    f0,f,g,ions,frac,response=_case()
    with pytest.raises(ValueError, match="Explicit optimizer initial estimates are required"):
        analyze_cts_spectrum(
            f,np.ones_like(f),probe_frequency_Hz=f0,scattering_geometry=g,
            ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,
            instrument_response=response,
        )


def test_cts_global_search_requires_explicit_finite_scientific_bounds():
    f0,f,g,ions,frac,response=_case()
    with pytest.raises(ValueError, match="Global search requires explicit finite parameter_bounds"):
        analyze_cts_spectrum(
            f,np.ones_like(f),probe_frequency_Hz=f0,scattering_geometry=g,
            ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,
            instrument_response=response,electron_density_initial_m3=2e19,
            electron_temperature_initial_eV=1200.0,ion_temperature_initial_eV=800.0,
            ion_velocity_initial_m_s=0.0,global_search=True,inversion_method="differential_evolution",
        )


def test_cts_maxwellian_fast_ion_requires_explicit_temperature_initial_estimate():
    f0,f,g,ions,frac,response=_case()
    with pytest.raises(ValueError, match="fast_ion.temperature_initial_eV is required"):
        analyze_cts_spectrum(
            f,np.ones_like(f),probe_frequency_Hz=f0,scattering_geometry=g,
            ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,
            instrument_response=response,electron_density_initial_m3=2e19,
            electron_temperature_initial_eV=1200.0,ion_temperature_initial_eV=800.0,
            ion_velocity_initial_m_s=0.0,
            fast_ion={"kind":"maxwellian","species_index":0,"fraction_initial":0.1},
            global_search=False,
        )


def test_cts_forward_model_is_finite_positive_and_geometry_sensitive():
    f0,f,g,ions,frac,_=_case()
    a=cts_spectral_density(f,f0,g,electron_density_m3=2e19,electron_temperature_eV=1200,ion_temperature_eV=800,ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,ion_velocity_projection_m_s=0.0)
    b=cts_spectral_density(f,f0,{"scattering_angle_rad":np.deg2rad(70.)},electron_density_m3=2e19,electron_temperature_eV=1200,ion_temperature_eV=800,ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,ion_velocity_projection_m_s=0.0)
    assert a.shape==f.shape and np.all(np.isfinite(a)) and np.all(a>=0) and not np.allclose(a,b,rtol=1e-8,atol=0.0)

def test_cts_inversion_recovers_controlled_synthetic_parameters():
    f0,f,g,ions,frac,response=_case(); truth=(2e19,1200.,800.,1.5e4)
    z=cts_spectral_density(f,f0,g,electron_density_m3=truth[0],electron_temperature_eV=truth[1],ion_temperature_eV=truth[2],ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,ion_velocity_projection_m_s=truth[3])
    y=2.4e4*z+3.
    out=analyze_cts_spectrum(f,y,probe_frequency_Hz=f0,scattering_geometry=g,ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,instrument_response=response,
      signal_uncertainty=np.full_like(y,.02*np.max(y)),electron_density_initial_m3=truth[0],electron_temperature_initial_eV=truth[1],
      ion_temperature_initial_eV=truth[2],ion_velocity_initial_m_s=truth[3],global_search=False,inversion_method="local_least_squares")
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
      ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,ion_velocity_projection_m_s=0.0,fast_ion={"kind":"maxwellian","species_index":0,"fraction":.08,"temperature_eV":12000})
    assert np.all(np.isfinite(z)) and np.all(z>0)

@pytest.mark.parametrize("g",[{},{"scattering_angle_rad":0.},{"scattering_angle_rad":4.}])
def test_cts_rejects_invalid_geometry(g):
    f0,f,_,ions,frac,_=_case()
    with pytest.raises(ValueError):
        cts_spectral_density(f,f0,g,electron_density_m3=2e19,electron_temperature_eV=1200,ion_temperature_eV=800,ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,ion_velocity_projection_m_s=0.0)


def test_cts_tabulated_non_maxwellian_fast_ion_forward_model():
    f0,f,g,ions,frac,_=_case()
    velocity=np.linspace(-1.2e6,1.2e6,81)
    pdf=np.exp(-0.5*(velocity/2.5e5)**2)*(1.0+0.35*np.exp(-0.5*((velocity-5e5)/1.2e5)**2))
    spectrum=cts_spectral_density(
        f,f0,g,electron_density_m3=2e19,electron_temperature_eV=1200,
        ion_temperature_eV=800,ion_species=ions,ion_fractions=frac,electron_velocity_projection_m_s=0.0,
        ion_velocity_projection_m_s=0.0,fast_ion={"species_index":0,"fraction":0.08,"kind":"tabulated_1d",
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
        global_search=False,inversion_method="local_least_squares",
    )
    assert out["reduced_chi_square"] is not None
    assert out["inversion_method"]=="local_least_squares"


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
            global_search=False,electron_density_initial_m3=truth[0],electron_temperature_initial_eV=truth[1],
            ion_temperature_initial_eV=truth[2],ion_velocity_initial_m_s=truth[3])
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
            electron_density_initial_m3=2e19,electron_temperature_initial_eV=1200.0,
            ion_temperature_initial_eV=800.0,ion_velocity_initial_m_s=0.0,
            global_search=False,inversion_method="local_least_squares",
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
        global_search=False,inversion_method="local_least_squares",instrument_response_uncertainty=np.zeros_like(response),
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
        global_search=False,inversion_method="local_least_squares",
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


def test_cts_forward_model_requires_explicit_ion_flow_projection():
    f0,f,g,ions,frac,_=_case()
    with pytest.raises(ValueError, match="ion_velocity_projection_m_s must be supplied explicitly"):
        cts_spectral_density(
            f,f0,g,electron_density_m3=2e19,electron_temperature_eV=1200,
            ion_temperature_eV=800,ion_species=ions,ion_fractions=frac,
            electron_velocity_projection_m_s=0.0,
        )


def test_cts_requires_an_explicit_electron_flow_projection():
    f0, f, g, ions, frac, _ = _case()
    with pytest.raises(ValueError, match="electron_velocity_projection_m_s must be supplied explicitly"):
        cts_spectral_density(
            f, f0, g, electron_density_m3=2e19,
            electron_temperature_eV=1200, ion_temperature_eV=800,
            ion_species=ions, ion_fractions=frac, ion_velocity_projection_m_s=0.0,
        )
    with pytest.raises(ValueError, match="electron_velocity_projection_m_s is required"):
        analyze_cts_spectrum(
            f, np.ones_like(f), probe_frequency_Hz=f0,
            scattering_geometry=g, ion_species=ions, ion_fractions=frac,
            instrument_response=np.eye(f.size),
        )



def test_cts_forward_model_rejects_unknown_fast_ion_kind():
    f0,f,g,ions,frac,_=_case()
    with pytest.raises(ValueError,match="requires explicit fast_ion.kind"):
        cts_spectral_density(
            f,f0,g,electron_density_m3=2e19,electron_temperature_eV=1200,
            ion_temperature_eV=800,ion_species=ions,ion_fractions=frac,
            electron_velocity_projection_m_s=0.0,ion_velocity_projection_m_s=0.0,
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
    assert result["bulk_electron_closure_validity_status"] == "not_quantitatively_screened"
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
    np.testing.assert_allclose(result["wavevector_rad_m"], np.broadcast_to(k0, result["wavevector_rad_m"].shape), rtol=1e-7, atol=1e-7)


def test_cts_cold_plasma_vacuum_propagating_mode_is_flux_normalized():
    from tokamak_analysis.cts_em_geometry import solve_cold_plasma_propagating_modes

    omega = 2.0 * np.pi * 140e9
    mode = solve_cold_plasma_propagating_modes(
        omega, [1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [], mode_index=0
    )
    assert mode["refractive_index"] == pytest.approx(1.0, rel=1e-12)
    assert mode["signed_longitudinal_flux_before_normalization"] > 0
    assert np.dot(mode["poynting_vector_unit_flux"], [1.0, 0.0, 0.0]) == pytest.approx(1.0, rel=1e-10)
    assert np.all(np.isfinite(mode["electric_polarization_unit_flux"]))


def test_bindslev_gamma_inverse_and_x_tensor_are_finite_and_consistent():
    from tokamak_analysis.cts_bindslev import bindslev_gamma_tensor, bindslev_x_tensor

    omega = 2.0 * np.pi * 140e9
    b = np.array([0.0, 0.0, 2.0])
    gamma = bindslev_gamma_tensor(omega, b)
    assert gamma["inverse_identity_residual"] < 1e-10
    x = bindslev_x_tensor(omega, [120.0, 0.0, 80.0], b)
    assert x["X"].shape == (3, 3, 3)
    assert np.all(np.isfinite(x["X"]))
    np.testing.assert_allclose(x["rotation_lab_from_B"].T @ x["rotation_lab_from_B"],
                               np.eye(3), atol=1e-12)


def test_bindslev_source_current_operator_and_full_cross_spectrum_contraction():
    from tokamak_analysis.cts_bindslev import (
        bindslev_source_current_operator, bindslev_correlated_transfer_spectrum,
    )
    from tokamak_analysis.cts_em_geometry import cold_plasma_dielectric_tensor

    omega_i = 2.0 * np.pi * 140e9
    omega_s = 2.0 * np.pi * 141e9
    b = np.array([0.0, 0.0, 2.0])
    ne = 2e19
    species = [{"density_m3": ne, "charge_C": -1.602176634e-19,
                "mass_kg": 9.1093837015e-31}]
    chi_i = cold_plasma_dielectric_tensor(omega_i, b, species) - np.eye(3)
    chi_s = cold_plasma_dielectric_tensor(omega_s, b, species) - np.eye(3)
    ei = np.array([0.0, 0.0, 1.0], dtype=complex)
    source = bindslev_source_current_operator(
        omega_incident_rad_s=omega_i, omega_scattered_rad_s=omega_s,
        wavevector_incident_rad_m=[omega_i/299792458.0, 0.0, 0.0],
        wavevector_scattered_rad_m=[0.0, omega_s/299792458.0, 0.0],
        incident_electric_field=ei, incident_susceptibility=chi_i,
        scattered_susceptibility=chi_s, magnetic_field_T=b,
        electron_density_m3=ne,
    )
    operator = source["source_current_operator"]
    assert operator.shape == (3, 10)
    assert np.all(np.isfinite(operator))
    seed = np.arange(1, 11, dtype=float).astype(complex)
    covariance = np.outer(seed, seed.conj())
    result = bindslev_correlated_transfer_spectrum(
        covariance, operator, np.array([0.0, 0.0, 1.0])
    )
    assert result["received_source_current_spectral_power"] >= 0
    assert result["includes_all_cross_correlations"] is True
    assert set(result["transfer_contributions"]) == {
        "density", "particle_flux", "electric_field", "magnetic_field"
    }
    with pytest.raises(ValueError, match="positive semidefinite"):
        bindslev_correlated_transfer_spectrum(
            np.diag([1.0] * 9 + [-1e-3]), operator, np.array([0.0, 0.0, 1.0])
        )
    with pytest.raises(ValueError, match="receiver_polarization must be non-zero"):
        bindslev_correlated_transfer_spectrum(
            covariance, operator, np.zeros(3)
        )


def test_full_em_cts_exposes_bindslev_three_wave_coupling():
    from tokamak_analysis.cts_full_em import full_em_cts_spectral_density

    result = full_em_cts_spectral_density(
        [0.8e9, 1.0e9, 1.2e9], 140e9,
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
    assert result["bindslev_source_current_operator"].shape == (3, 3, 10)
    assert result["bindslev_X_tensor"].shape == (3, 3, 3, 3)
    assert result["bindslev_Gamma_inverse_tensor"].shape == (3, 3, 3, 3, 3)
    assert result["bindslev_dielectric_form_factors"].shape == (3, 10, 10)
    assert len(result["bindslev_coupling_coefficients"]) == 3
    assert "Gamma^{-1}" in result["three_wave_coupling"]
    assert np.all(np.isfinite(result["spectral_density"]))


def test_relativistic_anisotropic_gyrotropic_fast_ion_response_and_shot_noise():
    from tokamak_analysis.cts_relativistic import relativistic_gyrotropic_species_response

    result = relativistic_gyrotropic_species_response(
        2*np.pi*1.0e8, [12.0, 0.0, 18.0], [0.0, 0.0, 2.0],
        density_m3=1.0e16, mass_kg=2.014*1.66053906660e-27,
        charge_C=1.602176634e-19,
        distribution={"kind":"relativistic_bigaussian",
                      "temperature_perpendicular_eV":8.0e5,
                      "temperature_parallel_eV":2.0e5},
        quadrature_order=4, gyrophase_points=16,
        resonance_width_rad_s=2*np.pi*1e5,
    )
    sigma=result["conductivity_S_m"]
    noise=result["bare_current_spectral_covariance"]
    assert sigma.shape==(3,3) and np.all(np.isfinite(sigma))
    assert noise.shape==(3,3) and np.all(np.isfinite(noise))
    np.testing.assert_allclose(noise,noise.conj().T,rtol=1e-10,atol=1e-30)
    assert np.min(np.linalg.eigvalsh(noise)) >= -1e-10*max(np.max(np.abs(noise)),1e-300)
    assert result["relativistic_kinematics"] is True
    assert result["anisotropy"].startswith("p_perpendicular")
    assert result["velocity_max_fraction_c"] > 0


def test_full_em_cts_accepts_relativistic_anisotropic_fast_ion_population():
    from tokamak_analysis.cts_full_em import full_em_cts_spectral_density

    result=full_em_cts_spectral_density(
        [0.8e9,1.0e9],140e9,
        {"incident_direction":[1.0,0.0,0.0],
         "scattered_direction":[np.cos(1.8),np.sin(1.8),0.0]},
        electron_density_m3=2e19,electron_temperature_eV=1200.0,
        ion_temperature_eV=800.0,
        ion_species=[{"name":"D+","charge_state":1,"mass_amu":2.014}],
        ion_fractions=[1.0],magnetic_field_T=[0.0,0.0,2.0],
        electron_flow_velocity_m_s=[0.0,0.0,0.0],
        ion_flow_velocity_m_s={"D+":[0.0,0.0,0.0]},
        incident_polarization=[0.0,0.0,1.0],
        scattered_polarization=[0.0,0.0,1.0],
        harmonic_order=3,quadrature_order=4,
        fast_ion_populations=[{
            "name":"D_fast","charge_state":1,"mass_amu":2.014,
            "density_m3":1e16,
            "distribution":{"kind":"relativistic_bigaussian",
                "temperature_perpendicular_eV":8e5,
                "temperature_parallel_eV":2e5}
        }],
    )
    assert len(result["fast_ion_populations"])==1
    assert len(result["fast_ion_kinetic_diagnostics"])==2
    assert result["fast_ion_kinetic_diagnostics"][0][0]["relativistic_kinematics"] is True
    assert result["spectral_density"].shape==(2,)
    assert np.all(np.isfinite(result["spectral_density"]))


def test_bindslev_dielectric_form_factors_are_hermitian_and_flux_normalized():
    from tokamak_analysis.cts_bindslev import (
        bindslev_source_current_operator, bindslev_dielectric_form_factors,
    )
    from tokamak_analysis.cts_em_geometry import cold_plasma_dielectric_tensor

    wi=2*np.pi*140e9; ws=2*np.pi*141e9; b=np.array([0.0,0.0,2.0]); ne=2e19
    ions=[{"density_m3":ne,"charge_C":-1.602176634e-19,"mass_kg":9.1093837015e-31}]
    chi_i=cold_plasma_dielectric_tensor(wi,b,ions)-np.eye(3)
    chi_s=cold_plasma_dielectric_tensor(ws,b,ions)-np.eye(3)
    ki=np.array([wi/299792458.0,0.0,0.0])
    ks=np.array([0.0,ws/299792458.0,0.0])
    ei=np.array([0.0,0.0,1.0],complex)
    es=np.array([0.0,0.0,1.0],complex)
    op=bindslev_source_current_operator(
        omega_incident_rad_s=wi,omega_scattered_rad_s=ws,
        wavevector_incident_rad_m=ki,wavevector_scattered_rad_m=ks,
        incident_electric_field=ei,incident_susceptibility=chi_i,
        scattered_susceptibility=chi_s,magnetic_field_T=b,electron_density_m3=ne,
    )["dielectric_coupling_operator"]
    result=bindslev_dielectric_form_factors(op,ei,es,ki,ks,wi,ws,ne)
    G=result["G_alpha_beta"]
    assert G.shape==(10,10)
    np.testing.assert_allclose(G,G.conj().T,rtol=1e-10,atol=1e-20)
    assert result["normalized_incident_flux_S_i"]>0
    assert result["normalized_scattered_flux_S_s"]>0


def test_full_em_species_resolved_ion_temperature_inputs():
    from tokamak_analysis.cts_full_em import _species_list

    species = _species_list(
        2.0e19,
        [
            {"name": "D+", "charge_state": 1, "mass_amu": 2.014},
            {"name": "C6+", "charge_state": 6, "mass_amu": 12.0},
        ],
        [0.7, 0.3], 1200.0,
        {"D+": 400.0, "C6+": 900.0},
        [0.0, 0.0, 0.0],
        {"D+": [0.0, 0.0, 0.0], "C6+": [0.0, 0.0, 0.0]},
    )
    assert species[1]["temperature"] == pytest.approx(400.0)
    assert species[2]["temperature"] == pytest.approx(900.0)
    assert species[1]["density"] == pytest.approx(0.7*2.0e19)
    assert species[2]["density"] == pytest.approx(0.3*2.0e19/6.0)
    with pytest.raises(ValueError, match="exactly the ion species names"):
        _species_list(
            2.0e19,
            [{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
            [1.0], 1200.0, {"T+": 400.0},
            [0.0, 0.0, 0.0], {"D+": [0.0, 0.0, 0.0]},
        )


def test_relativistic_gyrotropic_three_wave_vertex_is_finite():
    from tokamak_analysis.cts_relativistic import relativistic_gyrotropic_three_wave_vertex

    wi = 2*np.pi*140e9
    wf = 2*np.pi*1.0e8
    result = relativistic_gyrotropic_three_wave_vertex(
        wi, [wi/299792458.0, 0.0, 0.0], [0.0, 0.0, 1.0],
        wf, [0.0, 12.0, 18.0], [0.0, 0.0, 2.0],
        density_m3=1.0e14, mass_kg=2.014*1.66053906660e-27,
        charge_C=1.602176634e-19,
        distribution={"kind": "relativistic_bigaussian",
                      "temperature_perpendicular_eV": 2.0e5,
                      "temperature_parallel_eV": 1.0e5},
        quadrature_order=4, gyrophase_points=16,
        resonance_width_rad_s=2*np.pi*1.0e6,
    )
    vertex = result["current_vertex_A_m2_per_V_m"]
    assert vertex.shape == (3, 3)
    assert np.all(np.isfinite(vertex))
    assert result["relativistic_kinematics"] is True
    assert result["second_order_vlasov"] is True


def test_frequency_resolved_cts_ray_and_gaussian_overlap_geometry(monkeypatch):
    import tokamak_analysis.cts_em_geometry as geometry

    def fake_trace(position, wavevector, omega, **kwargs):
        base = np.asarray(position, dtype=float)
        k = np.asarray(wavevector, dtype=float)
        return {
            "position_m": np.vstack((base, base + np.array([0.5, 0.0, 0.0]), base + np.array([1.0, 0.0, 0.0]))),
            "wavevector_rad_m": np.vstack((k, k, k)),
        }

    monkeypatch.setattr(geometry, "trace_cold_plasma_ray", fake_trace)
    equilibrium = lambda x: {"magnetic_field_T": np.array([0.0, 0.0, 2.0]), "species": []}
    cov = np.eye(3)*1.0e-4
    shifts = np.array([0.8e9, 1.0e9])
    result = geometry.trace_cts_frequency_resolved_beams(
        shifts, 140e9,
        incident_ray={"initial_position_m": [0.0, 0.0, 0.0],
                      "initial_wavevector_rad_m": [1.0, 0.0, 0.0]},
        receiver_rays=[
            {"initial_position_m": [0.0, 0.0, 0.0],
             "initial_wavevector_rad_m": [0.0, 1.0, 0.0]},
            {"initial_position_m": [0.0, 0.0, 0.0],
             "initial_wavevector_rad_m": [0.0, 1.0, 0.0]},
        ],
        equilibrium=equilibrium,
        incident_beam_covariance_m2=cov,
        receiver_beam_covariances_m2=cov,
    )
    assert len(result["frequency_resolved_geometry"]) == 2
    assert all(len(item["scattering_samples"]) == 3 for item in result["frequency_resolved_geometry"])
    for item in result["frequency_resolved_geometry"]:
        assert sum(sample["sample_weight"] for sample in item["scattering_samples"]) == pytest.approx(1.0)
    np.testing.assert_allclose(result["relative_beam_overlap"], [1.0, 1.0])
    np.testing.assert_allclose(
        result["frequency_resolved_geometry"][0]["magnetic_field_T"], [0.0, 0.0, 2.0]
    )


def test_full_em_cts_relativistic_non_maxwellian_electron_uses_kinetic_vertex():
    from tokamak_analysis.cts_full_em import full_em_cts_spectral_density

    result = full_em_cts_spectral_density(
        [0.8e9, 1.0e9], 140e9,
        {"incident_direction": [1.0, 0.0, 0.0],
         "scattered_direction": [np.cos(1.8), np.sin(1.8), 0.0]},
        electron_density_m3=2e19, electron_temperature_eV=1200.0,
        ion_temperature_eV={"D+": 800.0},
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], magnetic_field_T=[0.0, 0.0, 2.0],
        electron_flow_velocity_m_s=[0.0, 0.0, 0.0],
        ion_flow_velocity_m_s={"D+": [0.0, 0.0, 0.0]},
        incident_polarization=[0.0, 0.0, 1.0],
        scattered_polarization=[0.0, 0.0, 1.0],
        harmonic_order=3, quadrature_order=4,
        electron_distribution={
            "kind": "relativistic_maxwell_juttner",
        },
    )
    assert result["relativistic_fast_ion_three_wave_vertex_used"] is False
    assert result["relativistic_kinetic_three_wave_vertex_used"] is True
    assert len(result["relativistic_three_wave_vertex_diagnostics"]) == 2
    assert result["relativistic_three_wave_vertex_diagnostics"][0][0]["population_role"] == "relativistic Maxwell-Juttner electron"
    assert result["nonthermal_electron_distribution"]["kind"] == "relativistic_maxwell_juttner"
    assert result["nonthermal_electron_distribution"]["temperature_eV"] == pytest.approx(1200.0)
    assert "non-Maxwellian electron" in result["model"]
    assert np.all(np.isfinite(result["spectral_density"]))


def test_full_em_cts_consumes_frequency_resolved_ray_and_overlap_geometry():
    from tokamak_analysis.cts_full_em import full_em_cts_spectral_density

    shifts = np.array([0.8e9, 1.0e9])
    geometry = {
        "frequency_resolved_geometry": [
            {"incident_direction": [1.0, 0.0, 0.0],
             "scattered_direction": [np.cos(1.8), np.sin(1.8), 0.0],
             "magnetic_field_T": [0.0, 0.0, 2.0],
             "beam_overlap_relative": 0.25, "scattering_location_m": [0.0, 0.0, 0.0]},
            {"incident_direction": [1.0, 0.0, 0.0],
             "scattered_direction": [np.cos(1.8), np.sin(1.8), 0.0],
             "magnetic_field_T": [0.0, 0.0, 2.0],
             "beam_overlap_relative": 1.0, "scattering_location_m": [0.0, 0.0, 0.0]},
        ]
    }
    result = full_em_cts_spectral_density(
        shifts, 140e9, geometry,
        electron_density_m3=2e19, electron_temperature_eV=1200.0,
        ion_temperature_eV={"D+": 800.0},
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], magnetic_field_T=[0.0, 0.0, 2.0],
        electron_flow_velocity_m_s=[0.0, 0.0, 0.0],
        ion_flow_velocity_m_s={"D+": [0.0, 0.0, 0.0]},
        incident_polarization=[0.0, 0.0, 1.0],
        scattered_polarization=[0.0, 0.0, 1.0],
        harmonic_order=3, quadrature_order=4,
    )
    assert result["ray_tracing_geometry_used"] is True
    np.testing.assert_allclose(result["relative_gaussian_beam_overlap"], [0.25, 1.0])
    assert result["scattering_locations_m"].shape == (2, 3)
    assert np.all(np.isfinite(result["spectral_density"]))


def test_relativistic_maxwell_juttner_momentum_grid_is_normalized():
    from tokamak_analysis.cts_relativistic import _momentum_grid, ME

    pperp, ppar, f0, dfperp, dfpar, weights = _momentum_grid(
        {"kind": "relativistic_maxwell_juttner", "temperature_eV": 2.0e5},
        ME, 16,
    )
    assert pperp.shape == ppar.shape == f0.shape == weights.shape
    assert np.all(np.isfinite(dfperp)) and np.all(np.isfinite(dfpar))
    assert np.all(f0 >= 0)
    assert np.sum(weights*f0) == pytest.approx(1.0, rel=1e-10)


def test_full_em_cts_integrates_scattering_volume_samples():
    from tokamak_analysis.cts_full_em import full_em_cts_spectral_density

    local_samples = [
        {"incident_direction": [1.0, 0.0, 0.0],
         "scattered_direction": [np.cos(1.8), np.sin(1.8), 0.0],
         "magnetic_field_T": [0.0, 0.0, 2.0],
         "scattering_location_m": [0.0, 0.0, 0.0],
         "sample_weight": 0.25, "beam_overlap_relative": 1.0},
        {"incident_direction": [1.0, 0.0, 0.0],
         "scattered_direction": [np.cos(1.8), np.sin(1.8), 0.0],
         "magnetic_field_T": [0.0, 0.0, 2.1],
         "scattering_location_m": [0.01, 0.0, 0.0],
         "sample_weight": 0.75, "beam_overlap_relative": 1.0},
    ]
    result = full_em_cts_spectral_density(
        [0.9e9], 140e9,
        {"frequency_shift_Hz": [0.9e9],
         "frequency_resolved_geometry": [{
             "incident_direction": [1.0, 0.0, 0.0],
             "scattered_direction": [np.cos(1.8), np.sin(1.8), 0.0],
             "magnetic_field_T": [0.0, 0.0, 2.075],
             "scattering_location_m": [0.0075, 0.0, 0.0],
             "beam_overlap_relative": 0.5,
             "scattering_samples": local_samples,
         }]},
        electron_density_m3=2e19, electron_temperature_eV=1200.0,
        ion_temperature_eV={"D+": 800.0},
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0], magnetic_field_T=[0.0, 0.0, 2.0],
        electron_flow_velocity_m_s=[0.0, 0.0, 0.0],
        ion_flow_velocity_m_s={"D+": [0.0, 0.0, 0.0]},
        incident_polarization=[0.0, 0.0, 1.0],
        scattered_polarization=[0.0, 0.0, 1.0],
        harmonic_order=3, quadrature_order=4,
    )
    assert result["scattering_volume_integration_used"] is True
    np.testing.assert_array_equal(result["scattering_volume_sample_count"], [2])
    np.testing.assert_allclose(result["relative_gaussian_beam_overlap"], [0.5])
    np.testing.assert_allclose(result["scattering_locations_m"], [[0.0075, 0.0, 0.0]])
    assert result["spectral_density"].shape == (1,)
    assert np.all(np.isfinite(result["spectral_density"]))


def test_frequency_resolved_cts_ray_helper_has_no_implicit_propagation_branch():
    from inspect import signature
    from tokamak_analysis.cts_em_geometry import trace_cts_frequency_resolved_beams

    assert signature(trace_cts_frequency_resolved_beams).parameters["mode_index"].default is None


def test_full_em_bindslev_closure_enforces_researcher_supplied_theta_limit():
    from tokamak_analysis.cts_full_em import full_em_cts_spectral_density

    with pytest.raises(ValueError, match="exceeds the researcher-supplied"):
        full_em_cts_spectral_density(
            [0.1e9, 0.2e9], 140e9, {},
            electron_density_m3=2e19, electron_temperature_eV=1200.0,
            ion_temperature_eV=800.0,
            ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
            ion_fractions=[1.0], magnetic_field_T=[0.0, 0.0, 2.0],
            electron_flow_velocity_m_s=[0.0, 0.0, 0.0],
            ion_flow_velocity_m_s={"D+": [0.0, 0.0, 0.0]},
            incident_polarization=[0.0, 0.0, 1.0],
            scattered_polarization=[0.0, 0.0, 1.0],
            bindslev_validity_theta_e_limit=0.001,
        )


def test_cts_full_em_dispatcher_maps_public_spectrum_keyword(monkeypatch):
    from tokamak_analysis import cts_analysis

    captured = {}

    def fake_full_em(**kwargs):
        captured.update(kwargs)
        return {"received": True}

    monkeypatch.setattr(cts_analysis, "analyze_full_em_cts_spectrum", fake_full_em)
    result = cts_analysis.analyze_cts_spectrum_model(
        model="full_electromagnetic",
        frequency_shift_Hz=np.array([-1.0, 1.0]),
        spectrum=np.array([2.0, 3.0]),
    )
    assert result == {"received": True}
    assert "spectrum" not in captured
    np.testing.assert_array_equal(captured["measured_spectrum"], [2.0, 3.0])


def test_cts_full_em_dispatcher_rejects_duplicate_spectrum_keywords(monkeypatch):
    from tokamak_analysis import cts_analysis

    monkeypatch.setattr(
        cts_analysis, "analyze_full_em_cts_spectrum",
        lambda **kwargs: kwargs,
    )
    with pytest.raises(ValueError, match="Supply the calibrated CTS spectrum once"):
        cts_analysis.analyze_cts_spectrum_model(
            model="full_electromagnetic",
            spectrum=[1.0, 2.0],
            measured_spectrum=[1.0, 2.0],
        )
