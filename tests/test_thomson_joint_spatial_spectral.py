import numpy as np
import pytest

from tokamak_analysis.thomson_joint_spatial_spectral import (
    _huber_total_variation_residual,
    reconstruct_joint_thomson_spatial_spectra,
)


def _synthetic_case():
    wavelength = np.linspace(520.0, 544.0, 241)
    radius = np.linspace(0.0, 1.0, 5)
    mixing = np.array([
        [0.80, 0.20, 0.00, 0.00, 0.00],
        [0.15, 0.70, 0.15, 0.00, 0.00],
        [0.00, 0.15, 0.70, 0.15, 0.00],
        [0.00, 0.00, 0.15, 0.70, 0.15],
        [0.00, 0.00, 0.00, 0.20, 0.80],
    ])
    te = np.array([70., 85., 100., 115., 130.])
    flow = np.array([-1.0e4, -0.5e4, 0., 0.5e4, 1.0e4])
    amp = np.array([1., 1.1, 1.2, 1.1, 1.])
    sigma = 2*532*np.sqrt(te/510998.95069)*np.sin(np.pi/4)
    center = 532*(1-2*np.sin(np.pi/4)*flow/299792458.)
    intrinsic = amp[:,None]*np.exp(-0.5*((wavelength[None,:]-center[:,None])/sigma[:,None])**2)
    spectra = mixing @ intrinsic
    spectra += 0.01
    return wavelength, spectra, radius, mixing, te, flow


def test_joint_spatial_spectral_inverse_recovers_synthetic_te_and_flow_profiles():
    x, y, r, h, te, flow = _synthetic_case()
    out = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532., scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
        signal_uncertainty=np.full_like(y, 0.01), regularization_strength=0.0,
    )
    assert out["analysis_status"] == "completed"
    np.testing.assert_allclose(out["electron_temperature_profile_eV"], te, rtol=0.04)
    np.testing.assert_allclose(out["electron_velocity_projection_profile_m_s"], flow, atol=2.5e3)
    assert out["fitted_spectra"].shape == y.shape
    assert out["parameter_covariance"].shape == (3*r.size+2*y.shape[0],)*2
    assert out["profile_parameter_covariance"].shape == (3*r.size, 3*r.size)
    assert out["regularized_parameter_covariance"].shape == out["parameter_covariance"].shape
    assert out["covariance_decomposition"]["reported_parameter_covariance"] == (
        "data_only_GLS_covariance_excluding_regularization_penalty"
    )
    np.testing.assert_allclose(
        out["profile_parameter_covariance"],
        out["parameter_covariance"][:3*r.size, :3*r.size],
    )
    assert out["spatial_mixing_operator"] == "researcher_supplied_fixed_forward_operator"
    assert out["experimental_validation"] == "not_performed; numerical software inference only"
    assert out["electron_temperature_uncertainty_profile_eV"].shape == (r.size,)
    assert out["electron_velocity_projection_uncertainty_profile_m_s"].shape == (r.size,)
    assert np.all(np.isnan(out["electron_density_profile_m3"]))
    assert out["parameter_identifiability"]["status"] == "identified_from_measured_data"


def test_joint_inverse_fits_polynomial_detector_background_jointly():
    x, y, r, h, *_ = _synthetic_case()
    z = 2.0 * (x - x.min()) / np.ptp(x) - 1.0
    baseline = np.vstack([
        0.02 + 0.004*z + 0.003*z*z,
        0.03 - 0.003*z + 0.002*z*z,
        0.01 + 0.002*z - 0.001*z*z,
        0.025 - 0.004*z + 0.0015*z*z,
        0.015 + 0.003*z + 0.0025*z*z,
    ])
    measured = y + baseline
    out = reconstruct_joint_thomson_spatial_spectra(
        x, measured, r, h, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
        signal_uncertainty=np.full_like(measured, 0.01),
        regularization_strength=0.0, background_model="polynomial",
        background_degree=2,
    )
    assert out["background_contract"]["fit_is_joint_with_physical_thomson_forward_model"] is True
    assert out["detector_background_model"] == "polynomial"
    assert out["detector_background_coefficients"].shape == (measured.shape[0], 3)
    assert out["parameter_covariance"].shape == (3*r.size + 3*measured.shape[0],)*2
    assert out["parameter_order"][-1] == "background_channel_4_coefficient_2"
    np.testing.assert_allclose(
        out["electron_temperature_profile_eV"], [70., 85., 100., 115., 130.],
        rtol=0.06,
    )


def test_joint_inverse_accepts_full_channel_major_covariance():
    x, y, r, h, *_ = _synthetic_case()
    covariance = np.eye(y.size) * 1e-4
    out = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532., scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
        joint_spectral_covariance=covariance, regularization_strength=1e-5,
    )
    assert out["noise_model"] == "full_channel_major_joint_spectral_covariance"
    assert np.all(np.isfinite(out["electron_temperature_uncertainty_eV"]))


def test_joint_inverse_rejects_singular_mixing_and_invalid_covariance():
    x, y, r, h, *_ = _synthetic_case()
    bad = h.copy()
    bad[-1] = bad[0]
    with pytest.raises(ValueError, match="rank deficient"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, bad, laser_wavelength_nm=532., scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
        )
    with pytest.raises(ValueError, match="positive definite"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, h, laser_wavelength_nm=532., scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
            joint_spectral_covariance=np.zeros((y.size,y.size)),
        )


def test_regularization_does_not_manufacture_data_identifiability():
    x, y, r, _, *_ = _synthetic_case()
    # One detector channel observes three spatial nodes through one summed
    # measurement. Curvature regularization may select a smooth solution, but
    # it cannot create independent measured constraints for all node values.
    mixing = np.ones((1, r.size), dtype=float)
    out = reconstruct_joint_thomson_spatial_spectra(
        x, y[:1], r, mixing, laser_wavelength_nm=532.,
        scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size), regularization_strength=1e-2,
        signal_uncertainty=np.full((1, x.size), 0.01),
    )
    assert out["optimizer"]["regularization_rows_excluded_from_identifiability"] is True
    assert out["optimizer"]["data_only_jacobian_rank"] < out["optimizer"]["parameter_count"]
    assert out["primary_result_status"] == "not_identifiable_from_measured_data"
    assert out["primary_outputs_reportable_as_identified"] is False
    assert out["electron_temperature_profile_eV"] is None
    assert out["electron_velocity_projection_profile_m_s"] is None
    assert out["parameter_covariance"] is None
    assert out["optimizer_candidate"]["role"] == "numerical_candidate_not_a_measured_result"



def test_response_stack_validates_column_sums_per_channel_and_preserves_operators():
    from tokamak_analysis.thomson_joint_spatial_spectral import _response_stack

    shared = np.array([
        [0.90, 0.10, 0.00],
        [0.10, 0.80, 0.10],
        [0.00, 0.10, 0.90],
    ])
    shared_before = shared.copy()
    stacked_shared = _response_stack(shared, n_channels=2, n_wavelengths=3)
    assert stacked_shared.shape == (2, 3, 3)
    np.testing.assert_array_equal(stacked_shared[0], shared_before)
    np.testing.assert_array_equal(stacked_shared[1], shared_before)
    np.testing.assert_array_equal(shared, shared_before)

    channel_specific = np.stack([
        shared,
        np.array([
            [0.85, 0.15, 0.00],
            [0.15, 0.75, 0.10],
            [0.00, 0.10, 0.90],
        ]),
    ])
    channel_before = channel_specific.copy()
    stacked_specific = _response_stack(channel_specific, n_channels=2, n_wavelengths=3)
    np.testing.assert_array_equal(stacked_specific, channel_before)
    np.testing.assert_array_equal(channel_specific, channel_before)


def test_response_stack_rejects_non_unit_columns_independently_for_each_channel():
    from tokamak_analysis.thomson_joint_spatial_spectral import _response_stack

    response = np.repeat(np.eye(3)[None, :, :], 2, axis=0)
    response[1, 0, 0] = 0.5
    with pytest.raises(ValueError, match="columns must sum to one for every channel"):
        _response_stack(response, n_channels=2, n_wavelengths=3)


@pytest.mark.parametrize("angle", [0.0, np.pi])
def test_joint_inverse_rejects_optical_scattering_angle_endpoints(angle):
    x, y, r, h, *_ = _synthetic_case()
    with pytest.raises(ValueError, match="strictly between 0 and pi"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, h, laser_wavelength_nm=532.0,
            scattering_angle_rad=angle, instrument_response=np.eye(x.size),
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("wavelength_nm", np.array([False, True]), "wavelength_nm must contain real numeric values"),
        ("measured_spectra", np.array([["1", "2"]]), "measured_spectra must contain real numeric values"),
        ("radius_m", np.array([0.0, np.nan]), "radius_m must contain only finite values"),
        ("spatial_mixing_matrix", np.array([[True, False]]), "spatial_mixing_matrix must contain real numeric values"),
    ],
)
def test_joint_inverse_rejects_non_numeric_or_nonfinite_measured_inputs(field, bad_value, message):
    x, y, r, h, *_ = _synthetic_case()
    values = {
        "wavelength_nm": x,
        "measured_spectra": y,
        "radius_m": r,
        "spatial_mixing_matrix": h,
    }
    values[field] = bad_value
    with pytest.raises(ValueError, match=message):
        reconstruct_joint_thomson_spatial_spectra(
            values["wavelength_nm"], values["measured_spectra"],
            values["radius_m"], values["spatial_mixing_matrix"],
            laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"laser_wavelength_nm": True}, "laser_wavelength_nm must be a finite real scalar"),
        ({"scattering_angle_rad": "1.57079632679"}, "scattering_angle_rad must be a finite real scalar"),
        ({"regularization_strength": False}, "regularization_strength must be a finite real scalar"),
        ({"positive_amplitude": "true"}, "positive_amplitude must be boolean"),
        ({"max_nfev": True}, "max_nfev must be a positive integer"),
    ],
)
def test_joint_inverse_rejects_invalid_scalar_and_control_types(kwargs, message):
    x, y, r, h, *_ = _synthetic_case()
    arguments = {
        "laser_wavelength_nm": 532.0,
        "scattering_angle_rad": np.pi / 2,
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match=message):
        reconstruct_joint_thomson_spatial_spectra(x, y, r, h, **arguments)


def test_joint_inverse_requires_explicit_measured_response_operator():
    x, y, r, h, *_ = _synthetic_case()
    with pytest.raises(ValueError, match="A measured instrument_response is required"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, h, laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2,
        )


def test_joint_maxwellian_inverse_rejects_fit_outside_nonrelativistic_domain():
    wavelength = np.linspace(400.0, 560.0, 401)
    radius = np.linspace(0.0, 1.0, 3)
    mixing = np.eye(3)
    te = 100.0
    flow = 0.11 * 299792458.0
    sigma = 2.0 * 532.0 * np.sqrt(te / 510998.95069) * np.sin(np.pi / 4.0)
    center = 532.0 * (1.0 - 2.0 * np.sin(np.pi / 4.0) * flow / 299792458.0)
    spectra = np.repeat(
        np.exp(-0.5 * ((wavelength - center) / sigma) ** 2)[None, :], 3, axis=0
    )
    out = reconstruct_joint_thomson_spatial_spectra(
        wavelength, spectra, radius, mixing,
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(wavelength.size),
        signal_uncertainty=np.full_like(spectra, 1e-3),
    )
    assert out["analysis_status"] == "model_domain_rejected"
    assert out["primary_result_status"] == "model_domain_rejected"
    assert out["primary_outputs_reportable_as_identified"] is False
    assert out["electron_temperature_profile_eV"] is None
    assert np.all(out["validity_screening"]["bulk_flow_beta_profile"] >= 0.1)
    assert out["optimizer_candidate"]["role"] == "numerical_candidate_not_a_measured_result"


def test_bound_aware_jacobian_handles_active_and_fixed_parameter_bounds():
    from tokamak_analysis.thomson_joint_spatial_spectral import _bound_aware_jacobian

    lower = np.array([0.0, -2.0, 3.0])
    upper = np.array([5.0, 2.0, 3.0])
    point = np.array([0.0, 0.5, 3.0])
    evaluated = []

    def model(parameters):
        evaluated.append(parameters.copy())
        x, y, fixed = parameters
        return np.array([x*x + 3.0*y, 2.0*x - y + fixed])

    jacobian = _bound_aware_jacobian(model, point, lower, upper)
    np.testing.assert_allclose(jacobian[:, 0], [0.0, 2.0], atol=2e-5)
    np.testing.assert_allclose(jacobian[:, 1], [3.0, -1.0], rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(jacobian[:, 2], [0.0, 0.0])
    assert all(np.all(sample >= lower) and np.all(sample <= upper) for sample in evaluated)



def test_bound_aware_jacobian_supports_unbounded_physical_parameters():
    from tokamak_analysis.thomson_joint_spatial_spectral import _bound_aware_jacobian

    point = np.array([0.7, -1.2])
    lower = np.array([-np.inf, -np.inf])
    upper = np.array([np.inf, np.inf])
    jacobian = _bound_aware_jacobian(
        lambda p: np.array([p[0] ** 2 + 3.0 * p[1], p[0] - p[1] ** 2]),
        point, lower, upper,
    )
    expected = np.array([[2.0 * point[0], 3.0], [1.0, -2.0 * point[1]]])
    np.testing.assert_allclose(jacobian, expected, rtol=2e-5, atol=2e-6)


def test_structured_thomson_background_models_recover_detector_baseline():
    from tokamak_analysis.thomson_background import fit_structured_background

    x = np.linspace(520.0, 540.0, 101)
    z = 2.0*(x-x.min())/np.ptp(x)-1.0
    y = 0.3 + 0.08*z + 0.015*z*z
    result = fit_structured_background(x, y, model="polynomial", degree=2)
    np.testing.assert_allclose(result["background"], y, atol=1e-11)
    assert result["background_role"] == "detector_space_nuisance_not_plasma_emission"


def test_joint_kappa_inverse_fits_index_and_propagates_its_uncertainty():
    wavelength = np.linspace(520.0, 544.0, 241)
    radius = np.linspace(0.0, 1.0, 5)
    mixing = np.array([
        [0.80, 0.20, 0.00, 0.00, 0.00],
        [0.15, 0.70, 0.15, 0.00, 0.00],
        [0.00, 0.15, 0.70, 0.15, 0.00],
        [0.00, 0.00, 0.15, 0.70, 0.15],
        [0.00, 0.00, 0.00, 0.20, 0.80],
    ])
    te = np.array([70.0, 85.0, 100.0, 115.0, 130.0])
    flow = np.array([-1.0e4, -0.5e4, 0.0, 0.5e4, 1.0e4])
    amplitude = np.array([1.0, 1.1, 1.2, 1.1, 1.0])
    true_kappa = 3.8
    sigma = 2.0*532.0*np.sqrt(te/510998.95069)*np.sin(np.pi/4.0)
    center = 532.0*(1.0-2.0*np.sin(np.pi/4.0)*flow/299792458.0)
    scale = sigma*np.sqrt((2.0*true_kappa-3.0)/true_kappa)
    from scipy.special import gammaln
    normalization = np.exp(
        gammaln(true_kappa)-0.5*np.log(np.pi*true_kappa)
        -np.log(scale)-gammaln(true_kappa-0.5)
    )
    z = (wavelength[None, :]-center[:, None])/scale[:, None]
    intrinsic = amplitude[:, None]*normalization[:, None]*(
        1.0+z*z/true_kappa
    )**(-true_kappa)
    measured = mixing@intrinsic + 0.005

    out = reconstruct_joint_thomson_spatial_spectra(
        wavelength, measured, radius, mixing,
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi/2.0,
        instrument_response=np.eye(wavelength.size),
        signal_uncertainty=np.full_like(measured, 0.002),
        physical_model="kappa", kappa_index=4.5, fit_kappa_index=True,
        regularization_strength=0.0, max_nfev=5000,
    )
    assert out["primary_result_status"] == "identified_from_measured_data"
    assert out["physical_model_selection"]["kappa_index_role"] == (
        "inferred_from_measured_spectra_with_profiled_uncertainty"
    )
    assert out["physical_model_selection"]["kappa_index"] == pytest.approx(
        true_kappa, rel=0.08
    )
    assert np.isfinite(out["kappa_index_uncertainty"])
    assert out["parameter_covariance"].shape == (
        3*radius.size + 1 + 2*measured.shape[0],
    )*2
    assert "kappa_index" in out["parameter_order"]
    assert out["detector_background_coefficients"].shape == (measured.shape[0], 1)
    assert np.all(np.isfinite(out["detector_background_coefficients"]))
    assert out["detector_background_offset"].shape == (measured.shape[0],)


def test_joint_inverse_adds_model_discrepancy_covariance_once_to_measurement_covariance():
    x, y, r, h, *_ = _synthetic_case()
    measurement = np.eye(y.size) * 1.0e-4
    discrepancy = np.eye(y.size) * 3.0e-4
    base = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(x.size),
        joint_spectral_covariance=measurement,
    )
    combined = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(x.size),
        joint_spectral_covariance=measurement,
        model_discrepancy_covariance=discrepancy,
    )
    assert combined["noise_model"] == "measurement_plus_model_discrepancy_channel_major_covariance"
    provenance = combined["spectral_covariance_provenance"]
    assert provenance["measurement_covariance_supplied"] is True
    assert provenance["model_discrepancy_covariance_supplied"] is True
    assert provenance["combined_covariance"] == (
        "measurement_covariance_plus_independent_model_discrepancy_covariance"
    )
    assert provenance["cross_covariance_assumed_zero"] is True
    np.testing.assert_allclose(
        combined["parameter_covariance"],
        4.0 * base["parameter_covariance"],
        rtol=2e-3, atol=1e-12,
    )


def test_joint_inverse_accepts_discrepancy_only_and_rejects_invalid_discrepancy_covariance():
    x, y, r, h, *_ = _synthetic_case()
    discrepancy = np.eye(y.size) * 2.0e-4
    out = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(x.size),
        model_discrepancy_covariance=discrepancy,
    )
    assert out["noise_model"] == "model_discrepancy_only_channel_major_covariance"
    assert out["spectral_covariance_provenance"]["measurement_covariance_supplied"] is False
    assert out["spectral_covariance_provenance"]["model_discrepancy_covariance_supplied"] is True

    invalid = np.eye(y.size)
    invalid[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, h, laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(x.size),
            model_discrepancy_covariance=invalid,
        )

    singular = np.zeros((y.size, y.size))
    with pytest.raises(ValueError, match="combined spectral covariance must be positive definite"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, h, laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(x.size),
            model_discrepancy_covariance=singular,
        )


def test_joint_inverse_reports_regularization_choice_sensitivity_separately():
    x, y, r, h, *_ = _synthetic_case()
    out = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(x.size),
        signal_uncertainty=np.full_like(y, 0.01),
        regularization_strength=1.0e-5,
        regularization_sensitivity_strengths=[1.0e-7, 1.0e-3],
    )
    sensitivity = out["regularization_sensitivity"]
    assert sensitivity["strengths"].tolist() == [1.0e-7, 1.0e-3]
    assert sensitivity["selected_strength"] == pytest.approx(1.0e-5)
    assert sensitivity["profile_candidates"].shape == (2, 3*r.size)
    assert sensitivity["profile_standard_deviation"].shape == (3*r.size,)
    assert "not measurement uncertainty" in sensitivity["interpretation"]
    assert len(sensitivity["candidate_diagnostics"]) == 2


@pytest.mark.parametrize(
    "strengths",
    [[1.0e-4], [1.0e-3, 1.0e-4], [0.0, 1.0e-3]],
)
def test_joint_inverse_rejects_invalid_regularization_sensitivity_grid(strengths):
    x, y, r, h, *_ = _synthetic_case()
    with pytest.raises(ValueError, match="strictly increasing positive finite values"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, h, laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(x.size),
            regularization_sensitivity_strengths=strengths,
        )


def test_joint_inverse_rejects_nonfinite_regularization_sensitivity_strengths():
    x, y, r, h, *_ = _synthetic_case()
    with pytest.raises(ValueError, match="regularization_sensitivity_strengths must contain only finite values"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, h, laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(x.size),
            regularization_sensitivity_strengths=[1.0e-3, np.inf],
        )



def test_joint_profile_workflow_propagates_spatial_mixing_covariance_by_refits():
    from tokamak_analysis.thomson_workflow import reconstruct_thomson_profile_from_spectra

    x, y, r, h, *_ = _synthetic_case()
    covariance = np.diag(np.where(h.reshape(-1) > 0.0, 1.0e-8, 0.0))
    result = reconstruct_thomson_profile_from_spectra(
        x, y, r,
        model="non_collective_maxwellian",
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(x.size),
        signal_uncertainty=np.full_like(y, 0.01),
        joint_spatial_spectral=True,
        spatial_mixing_matrix=h,
        spatial_mixing_covariance=covariance,
        spatial_mixing_mc_samples=20,
        spatial_mixing_mc_seed=17,
        spatial_regularization_strength=0.0,
    )
    uncertainty = result["spatial_mixing_uncertainty"]
    assert uncertainty["supplied"] is True
    assert uncertainty["method"] == "nonlinear_forward_refit_monte_carlo"
    assert uncertainty["requested_samples"] == 20
    assert uncertainty["successful_samples"] >= 20
    assert uncertainty["propagated"] is True
    assert result["parameter_covariance"].shape == (len(result["parameter_order"]),) * 2
    assert result["profile_parameter_covariance"].shape == (3 * r.size, 3 * r.size)
    assert result["covariance_decomposition"]["spatial_mixing_covariance"] == (
        "nonlinear_refit_external_operator_contribution"
    )


def test_joint_profile_workflow_rejects_invalid_spatial_mixing_covariance():
    from tokamak_analysis.thomson_workflow import reconstruct_thomson_profile_from_spectra

    x, y, r, h, *_ = _synthetic_case()
    invalid = np.eye(h.size)
    invalid[0, 1] = 0.1
    with pytest.raises(ValueError, match="must be symmetric"):
        reconstruct_thomson_profile_from_spectra(
            x, y, r, model="non_collective_maxwellian",
            laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2.0,
            instrument_response=np.eye(x.size), signal_uncertainty=np.full_like(y, 0.01),
            joint_spatial_spectral=True, spatial_mixing_matrix=h,
            spatial_mixing_covariance=invalid,
        )



def test_joint_compton_inverse_integrates_measured_laser_profile_in_physical_kernel(monkeypatch):
    import tokamak_analysis.thomson_kinetic_models as kinetic

    x, y, radius, mixing, *_ = _synthetic_case()
    incident_wavelengths = np.array([531.8, 532.0, 532.2])
    incident_density = np.array([0.0, 1.0, 0.0])
    evaluated_laser_wavelengths = []

    def fake_mj_spectrum(
        wavelength_nm, laser_wavelength_nm, scattering_angle_rad,
        electron_temperature_eV, **kwargs,
    ):
        evaluated_laser_wavelengths.append(float(laser_wavelength_nm))
        wavelength = np.asarray(wavelength_nm, dtype=float)
        centre = float(laser_wavelength_nm) * (
            1.0 - 2.0 * np.sin(scattering_angle_rad / 2.0)
            * kwargs.get("flow_projection_m_s", 0.0) / 299792458.0
        )
        return {
            "spectral_density_per_nm": np.exp(
                -0.5 * ((wavelength - centre) / 1.5) ** 2
            )
        }

    monkeypatch.setattr(kinetic, "relativistic_maxwell_juttner_spectrum", fake_mj_spectrum)
    out = reconstruct_joint_thomson_spatial_spectra(
        x, y, radius, mixing,
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(x.size),
        signal_uncertainty=np.full_like(y, 0.01),
        physical_model="relativistic_maxwell_juttner",
        incident_polarization=[0.0, 0.0, 1.0],
        include_compton_recoil=True,
        laser_spectrum_wavelength_nm=incident_wavelengths,
        laser_spectrum_density_per_nm=incident_density,
        radial_order=12, angular_order=8, max_nfev=1,
    )
    assert set(incident_wavelengths).issubset(set(evaluated_laser_wavelengths))
    assert out["laser_spectral_profile"]["model"] == (
        "measured_incident_laser_profile_integrated_inside_Compton_forward_model"
    )
    assert out["spectral_bin_integration"] == (
        "8_point_Gauss_Legendre_finite_volume_bin_averages_of_relativistic_forward_kernel"
    )
    for channel in out["laser_spectral_profile"]["channels"]:
        assert channel["incident_wavelength_quadrature_weights_sum"] == pytest.approx(1.0)
        assert "Klein_Nishina_kernel" in channel["convolution_approximation"]


def test_full_rank_but_ill_conditioned_data_inverse_is_not_reported_as_primary():
    x, y, r, h, *_ = _synthetic_case()
    out = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(x.size),
        signal_uncertainty=np.full_like(y, 0.01),
        maximum_data_condition_number=1.0,
    )
    assert out["optimizer"]["data_only_jacobian_rank"] == out["optimizer"]["parameter_count"]
    assert out["optimizer"]["data_only_scaled_jacobian_condition_number"] > 1.0
    assert out["optimizer"]["data_condition_number_acceptable"] is False
    assert out["primary_result_status"] == "ill_conditioned_from_measured_data"
    assert out["primary_outputs_reportable_as_identified"] is False
    assert out["electron_temperature_profile_eV"] is None
    assert out["electron_velocity_projection_profile_m_s"] is None
    assert out["parameter_covariance"] is None
    assert out["identifiability"]["data_condition_number_limit"] == 1.0


def test_general_relativistic_bin_quadrature_integrates_smooth_spectrum_on_nonuniform_grid():
    from tokamak_analysis.thomson_joint_spatial_spectral import (
        _bin_averaged_callable_spectrum, _wavelength_edges,
    )
    from scipy.special import erf

    x = np.array([529.0, 529.7, 531.1, 533.8, 537.0, 541.0])
    centre, sigma, amplitude = 533.2, 1.35, 2.7
    averaged = _bin_averaged_callable_spectrum(
        x,
        lambda wavelength: amplitude * np.exp(
            -0.5 * ((wavelength - centre) / sigma) ** 2
        ),
        quadrature_order=24,
    )
    edges = _wavelength_edges(x)
    root_two = np.sqrt(2.0)
    exact_integrals = (
        amplitude * sigma * np.sqrt(np.pi / 2.0)
        * (erf((edges[1:] - centre) / (root_two * sigma))
           - erf((edges[:-1] - centre) / (root_two * sigma)))
    )
    np.testing.assert_allclose(
        averaged, exact_integrals / np.diff(edges), rtol=2e-11, atol=1e-13
    )


def test_general_relativistic_bin_quadrature_rejects_invalid_kernel_values():
    from tokamak_analysis.thomson_joint_spatial_spectral import (
        _bin_averaged_callable_spectrum,
    )
    x = np.linspace(530.0, 534.0, 9)
    with pytest.raises(ValueError, match="one finite value per quadrature wavelength"):
        _bin_averaged_callable_spectrum(x, lambda wavelength: np.full(3, 1.0))
    with pytest.raises(ValueError, match="non-negative"):
        _bin_averaged_callable_spectrum(
            x, lambda wavelength: -np.ones_like(wavelength)
        )


def test_finite_bin_maxwellian_average_conserves_gaussian_area_on_nonuniform_grid():
    from tokamak_analysis.thomson_joint_spatial_spectral import _bin_averaged_maxwellian_spectrum, _wavelength_edges
    from scipy.special import ndtr
    x = np.array([529.0, 530.0, 531.4, 533.2, 536.0, 540.0])
    center, sigma, peak = 533.1, 1.7, 2.4
    averaged = _bin_averaged_maxwellian_spectrum(x, center, sigma, peak)
    edges = _wavelength_edges(x)
    exact_area = peak*np.sqrt(2.0*np.pi)*sigma*(ndtr((edges[-1]-center)/sigma)-ndtr((edges[0]-center)/sigma))
    assert np.sum(averaged*np.diff(edges)) == pytest.approx(exact_area, rel=2e-13)


def test_finite_bin_kappa_average_matches_projected_student_t_and_area():
    from scipy.special import stdtr
    from tokamak_analysis.thomson_joint_spatial_spectral import _bin_averaged_kappa_spectrum, _wavelength_edges
    x = np.array([529.0, 530.0, 531.4, 533.2, 536.0, 540.0])
    center, sigma, kappa, area = 533.1, 1.7, 3.2, 4.5
    averaged = _bin_averaged_kappa_spectrum(x, center, sigma, kappa, area)
    edges = _wavelength_edges(x)
    nu = 2.0*kappa-1.0
    scale = sigma*np.sqrt((nu-2.0)/nu)
    probability = stdtr(nu, (edges[1:]-center)/scale)-stdtr(nu, (edges[:-1]-center)/scale)
    positive = ((edges[:-1]-center)/scale) > 0.0
    probability[positive] = stdtr(nu, -(edges[:-1][positive]-center)/scale)-stdtr(nu, -(edges[1:][positive]-center)/scale)
    np.testing.assert_allclose(averaged, area*probability/np.diff(edges), rtol=2e-13, atol=1e-14)
    assert np.sum(averaged*np.diff(edges)) == pytest.approx(area*probability.sum(), rel=2e-13)


def test_joint_inverse_declares_exact_finite_bin_forward_physics_for_maxwellian_and_kappa():
    x, y, r, h, *_ = _synthetic_case()
    maxwellian = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532.0, scattering_angle_rad=np.pi/2,
        instrument_response=np.eye(x.size), signal_uncertainty=np.full_like(y, 0.01),
        regularization_strength=0.0,
    )
    assert maxwellian["spectral_bin_integration"] == "exact_Gaussian_CDF_finite_volume_bin_averages"
    kappa = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532.0, scattering_angle_rad=np.pi/2,
        instrument_response=np.eye(x.size), signal_uncertainty=np.full_like(y, 0.01),
        regularization_strength=0.0, physical_model="kappa", kappa_index=4.0,
    )
    assert kappa["spectral_bin_integration"] == "exact_Student_t_CDF_finite_volume_bin_averages_nu_2kappa_minus_1"


def test_gaussian_spatial_psf_is_finite_volume_and_preserves_edge_loss():
    from tokamak_analysis.thomson_joint_spatial_spectral import gaussian_spatial_psf_matrix

    radius = np.linspace(0.0, 1.0, 5)
    identity = gaussian_spatial_psf_matrix(radius, 0.0)
    np.testing.assert_array_equal(identity, np.eye(radius.size))

    psf = gaussian_spatial_psf_matrix(radius, 0.08)
    assert psf.shape == (radius.size, radius.size)
    assert np.all(np.isfinite(psf))
    assert np.all(psf >= 0.0)
    assert np.all(psf.sum(axis=0) <= 1.0 + 1e-12)
    assert np.all(psf.sum(axis=0) > 0.0)
    # The finite radial domain truncates the Gaussian tails at its edges.
    assert psf.sum(axis=0)[0] < 1.0
    assert psf.sum(axis=0)[-1] < 1.0
    assert np.allclose(psf, psf[::-1, ::-1], rtol=1e-12, atol=1e-14)


def test_joint_spatial_spectral_inverse_composes_measured_mixing_with_gaussian_psf():
    from tokamak_analysis.thomson_joint_spatial_spectral import (
        gaussian_spatial_psf_matrix,
    )

    wavelength, _, radius, mixing, te, flow = _synthetic_case()
    psf = gaussian_spatial_psf_matrix(radius, 0.05)
    sigma = 2.0 * 532.0 * np.sqrt(te / 510998.95069) * np.sin(np.pi / 4.0)
    center = 532.0 * (
        1.0 - 2.0 * np.sin(np.pi / 4.0) * flow / 299792458.0
    )
    intrinsic = np.exp(
        -0.5 * ((wavelength[None, :] - center[:, None]) / sigma[:, None]) ** 2
    )
    measured = mixing @ psf @ intrinsic + 0.01

    out = reconstruct_joint_thomson_spatial_spectra(
        wavelength, measured, radius, mixing,
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(wavelength.size),
        signal_uncertainty=np.full_like(measured, 0.01),
        regularization_strength=0.0,
        spatial_psf_sigma_m=0.05,
    )
    assert out["primary_result_status"] == "identified_from_measured_data"
    np.testing.assert_allclose(out["effective_spatial_mixing_matrix"], mixing @ psf)
    np.testing.assert_allclose(out["spatial_psf_matrix"], psf)
    assert out["spatial_psf_contract"]["source"] == "finite_volume_gaussian_psf_from_sigma"
    assert out["spatial_psf_contract"]["renormalized"] is False
    np.testing.assert_allclose(out["electron_temperature_profile_eV"], te, rtol=0.04)
    np.testing.assert_allclose(
        out["electron_velocity_projection_profile_m_s"], flow, atol=2.5e3
    )



def test_joint_spatial_spectral_inverse_propagates_gaussian_psf_calibration_uncertainty():
    from tokamak_analysis.thomson_joint_spatial_spectral import (
        gaussian_spatial_psf_matrix,
    )

    wavelength, _, radius, mixing, te, flow = _synthetic_case()
    psf_sigma = 0.05
    psf = gaussian_spatial_psf_matrix(radius, psf_sigma)
    sigma = 2.0 * 532.0 * np.sqrt(te / 510998.95069) * np.sin(np.pi / 4.0)
    center = 532.0 * (
        1.0 - 2.0 * np.sin(np.pi / 4.0) * flow / 299792458.0
    )
    intrinsic = np.exp(
        -0.5 * ((wavelength[None, :] - center[:, None]) / sigma[:, None]) ** 2
    )
    measured = mixing @ psf @ intrinsic + 0.01
    common = dict(
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(wavelength.size),
        signal_uncertainty=np.full_like(measured, 0.01),
        regularization_strength=0.0,
        spatial_psf_sigma_m=psf_sigma,
    )
    nominal = reconstruct_joint_thomson_spatial_spectra(
        wavelength, measured, radius, mixing, **common
    )
    propagated = reconstruct_joint_thomson_spatial_spectra(
        wavelength, measured, radius, mixing,
        spatial_psf_sigma_uncertainty_m=0.005, **common
    )
    psf_covariance = propagated["psf_calibration_parameter_covariance"]
    assert psf_covariance is not None
    assert psf_covariance.shape == propagated["parameter_covariance"].shape
    assert np.all(np.isfinite(psf_covariance))
    assert np.min(np.linalg.eigvalsh(0.5 * (psf_covariance + psf_covariance.T))) >= -1e-10
    np.testing.assert_allclose(
        propagated["parameter_covariance"],
        nominal["parameter_covariance"] + psf_covariance,
        rtol=2e-5, atol=1e-12,
    )
    assert propagated["spatial_psf_contract"]["sigma_uncertainty_m"] == 0.005
    assert "linearized_GLS_parameter_covariance" in propagated["spatial_psf_contract"]["uncertainty"]
    assert propagated["psf_sigma_parameter_sensitivity"] is not None


def test_joint_spatial_spectral_inverse_rejects_unattached_psf_width_uncertainty():
    wavelength, spectra, radius, mixing, *_ = _synthetic_case()
    with pytest.raises(ValueError, match="requires spatial_psf_sigma_m"):
        reconstruct_joint_thomson_spatial_spectra(
            wavelength, spectra, radius, mixing,
            laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2.0,
            instrument_response=np.eye(wavelength.size),
            spatial_psf_sigma_uncertainty_m=0.005,
        )

def test_joint_spatial_spectral_inverse_validates_measured_psf_contract():
    wavelength, spectra, radius, mixing, *_ = _synthetic_case()
    invalid = np.eye(radius.size)
    invalid[0, 0] = 1.2
    with pytest.raises(ValueError, match="columns must have positive mass no greater than one"):
        reconstruct_joint_thomson_spatial_spectra(
            wavelength, spectra, radius, mixing,
            laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2.0,
            instrument_response=np.eye(wavelength.size),
            spatial_psf_matrix=invalid,
        )
    with pytest.raises(ValueError, match="supply spatial_psf_matrix or spatial_psf_sigma_m"):
        reconstruct_joint_thomson_spatial_spectra(
            wavelength, spectra, radius, mixing,
            laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2.0,
            instrument_response=np.eye(wavelength.size),
            spatial_psf_matrix=np.eye(radius.size),
            spatial_psf_sigma_m=0.05,
        )

def test_huber_total_variation_penalty_is_quadratic_for_small_and_linear_for_large_gradients():
    radius = np.linspace(0.0, 1.0, 5)
    small_gradient = 1.0e-3 * radius
    large_gradient = 100.0 * radius
    small_residual = _huber_total_variation_residual(
        small_gradient, radius, huber_scale=0.05
    )
    large_residual = _huber_total_variation_residual(
        large_gradient, radius, huber_scale=0.05
    )
    # In the quadratic regime, sum(residual**2) approximates integral |u'|^2 dr.
    assert np.sum(small_residual**2) == pytest.approx(1.0e-6, rel=2e-3)
    # In the linear regime, doubling a large constant gradient doubles,
    # rather than quadruples, the Huber-TV objective.
    doubled = _huber_total_variation_residual(
        200.0 * radius, radius, huber_scale=0.05
    )
    assert np.sum(doubled**2) / np.sum(large_residual**2) == pytest.approx(
        2.0, rel=2e-3
    )


def test_joint_spatial_inverse_exposes_edge_preserving_regularization():
    x, y, r, h, *_ = _synthetic_case()
    out = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
        signal_uncertainty=np.full_like(y, 0.01),
        regularization_strength=0.02,
        regularization_kind="huber_total_variation",
        tv_huber_scale=0.05,
    )
    assert out["regularization"]["edge_preserving"] is True
    assert out["regularization"]["kind"] == (
        "length_weighted_huber_total_variation_on_log_Te_flow_and_log_amplitude"
    )
    assert out["regularization"]["huber_scale"] == pytest.approx(0.05)


def test_joint_spatial_inverse_rejects_unknown_regularization_kind():
    x, y, r, h, *_ = _synthetic_case()
    with pytest.raises(ValueError, match="regularization_kind"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, h, laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
            regularization_kind="invented_smoother",
        )

def test_edge_preserving_spatial_regularization_is_available_through_public_dispatcher():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    x, y, r, h, *_ = _synthetic_case()
    out = run_advanced_thomson_physics(
        "thomson_joint_spatial_spectra",
        x, y, r, h,
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2,
        instrument_response=np.eye(x.size),
        signal_uncertainty=np.full_like(y, 0.01),
        regularization_strength=0.02,
        regularization_kind="huber_total_variation",
        tv_huber_scale=0.05,
    )
    assert out["public_physics_entrypoint"] == "thomson_joint_spatial_spectra"
    assert out["regularization"]["edge_preserving"] is True

def test_joint_inverse_propagates_systematic_covariance_into_profile_uncertainty():
    x, y, r, h, *_ = _synthetic_case()
    measurement = np.eye(y.size) * 1.0e-4
    systematic = np.eye(y.size) * 3.0e-4
    base = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(x.size),
        joint_spectral_covariance=measurement,
    )
    combined = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(x.size),
        joint_spectral_covariance=measurement,
        systematic_covariance=systematic,
    )
    provenance = combined["spectral_covariance_provenance"]
    assert combined["noise_model"] == "measurement_plus_systematic_channel_major_covariance"
    assert provenance["systematic_covariance_supplied"] is True
    assert provenance["systematic_covariance_shape"] == [y.size, y.size]
    assert provenance["systematic_covariance_trace"] == pytest.approx(
        float(np.trace(systematic))
    )
    assert provenance["cross_covariance_assumed_zero"] is True
    np.testing.assert_allclose(
        combined["parameter_covariance"],
        4.0 * base["parameter_covariance"],
        rtol=3.0e-3, atol=1.0e-12,
    )
    components = combined["spectral_covariance_parameter_components"]
    np.testing.assert_allclose(
        components["measurement"] + components["systematic"],
        combined["measurement_data_parameter_covariance"],
        rtol=2.0e-8, atol=1.0e-12,
    )
    np.testing.assert_allclose(
        components["systematic"],
        0.75 * combined["measurement_data_parameter_covariance"],
        rtol=3.0e-3, atol=1.0e-12,
    )


def test_joint_inverse_rejects_non_psd_systematic_covariance():
    x, y, r, h, *_ = _synthetic_case()
    invalid = np.eye(y.size)
    invalid[0, 0] = -1.0
    with pytest.raises(ValueError, match="systematic_covariance must be positive semidefinite"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, h, laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(x.size),
            signal_uncertainty=np.full_like(y, 0.01),
            systematic_covariance=invalid,
        )


def test_joint_inverse_carries_systematic_covariance_through_regularization_sensitivity():
    x, y, r, h, *_ = _synthetic_case()
    systematic = np.eye(y.size) * 2.0e-5
    out = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(x.size),
        signal_uncertainty=np.full_like(y, 0.01),
        systematic_covariance=systematic,
        regularization_strength=1.0e-5,
        regularization_sensitivity_strengths=[1.0e-7, 1.0e-3],
    )
    assert out["spectral_covariance_provenance"]["systematic_covariance_supplied"] is True
    assert len(out["regularization_sensitivity"]["candidate_diagnostics"]) == 2



def test_joint_inverse_propagates_full_spatial_mixing_operator_covariance():
    x, y, r, h, *_ = _synthetic_case()
    mixing_covariance = np.eye(h.size) * 2.5e-7
    out = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
        signal_uncertainty=np.full_like(y, 0.01),
        spatial_mixing_matrix_covariance=mixing_covariance,
        regularization_strength=0.0,
    )
    calibration = out["spatial_mixing_calibration_parameter_covariance"]
    assert calibration is not None
    assert calibration.shape == out["parameter_covariance"].shape
    assert np.all(np.isfinite(calibration))
    assert np.linalg.eigvalsh(0.5*(calibration+calibration.T))[0] >= -1e-10
    assert out["spatial_mixing_covariance_contract"]["supplied"] is True
    assert out["spatial_mixing_covariance_contract"]["parameter_order"] == (
        "row_major_channel_then_true_spatial_node"
    )
    assert out["covariance_decomposition"]["spatial_mixing_calibration_covariance_available"] is True
    assert np.trace(out["parameter_covariance"]) > np.trace(
        out["measurement_data_parameter_covariance"]
    )


def test_joint_inverse_propagates_spectrum_operator_cross_covariance_and_validates_joint_block():
    x, y, r, h, *_ = _synthetic_case()
    mixing_covariance = np.eye(h.size) * 2.5e-7
    measurement_covariance = np.eye(y.size) * 1.0e-4
    cross = np.zeros((y.size, h.size))
    cross[0, 0] = 1.0e-6
    out = reconstruct_joint_thomson_spatial_spectra(
        x, y, r, h, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
        joint_spectral_covariance=measurement_covariance,
        spatial_mixing_matrix_covariance=mixing_covariance,
        spectral_mixing_cross_covariance=cross,
        regularization_strength=0.0,
    )
    assert out["spatial_mixing_covariance_contract"]["cross_covariance_with_measured_spectra_supplied"] is True
    assert out["spatial_mixing_cross_covariance_parameter_contribution"] is not None
    assert np.any(np.abs(out["spatial_mixing_cross_covariance_parameter_contribution"]) > 0.0)

    with pytest.raises(ValueError, match="requires measured signal_uncertainty"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, h, laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
            spatial_mixing_matrix_covariance=mixing_covariance,
            spectral_mixing_cross_covariance=cross,
        )
    invalid_cross = np.ones((y.size, h.size)) * 1e-2
    with pytest.raises(ValueError, match="joint measured-spectrum/spatial-mixing covariance"):
        reconstruct_joint_thomson_spatial_spectra(
            x, y, r, h, laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
            joint_spectral_covariance=measurement_covariance,
            spatial_mixing_matrix_covariance=mixing_covariance,
            spectral_mixing_cross_covariance=invalid_cross,
        )
