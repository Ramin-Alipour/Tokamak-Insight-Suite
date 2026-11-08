import numpy as np
import pytest

from tokamak_analysis.thomson_temperature_tensor import (
    infer_full_temperature_tensor_from_thomson_projections,
)


def _views():
    q = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ], dtype=float)
    q /= np.linalg.norm(q, axis=1)[:, None]
    tensor = np.array([[800.0, 90.0, -35.0],
                       [90.0, 420.0, 55.0],
                       [-35.0, 55.0, 260.0]])
    projected = np.einsum("vi,ij,vj->v", q, tensor, q)
    return q, tensor, projected


def test_full_temperature_tensor_recovers_non_gyrotropic_tensor_with_gls_covariance():
    q, expected, projected = _views()
    covariance = np.eye(q.shape[0]) * 0.04
    result = infer_full_temperature_tensor_from_thomson_projections(
        projected, q, projected_temperature_covariance_eV2=covariance,
    )
    np.testing.assert_allclose(result["electron_temperature_tensor_eV"], expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result["principal_temperature_eigenvalues_eV"], np.linalg.eigvalsh(expected))
    assert result["view_dyad_rank"] == 6
    assert result["parameter_covariance_eV2"].shape == (6, 6)
    assert result["reportable_as_inferred_parameter"]



def test_exactly_six_unweighted_views_do_not_claim_parameter_uncertainty():
    q, _, projected = _views()
    result = infer_full_temperature_tensor_from_thomson_projections(
        projected[:6], q[:6],
    )
    assert result["degrees_of_freedom"] == 0
    assert result["parameter_covariance_eV2"] is None
    assert result["parameter_covariance_status"] == (
        "not_reported_without_absolute_covariance_or_redundant_views"
    )
    assert result["reportable_as_inferred_parameter"] is True

def test_full_temperature_tensor_rejects_rank_deficient_view_geometry():
    q = np.tile([1.0, 0.0, 0.0], (6, 1))
    with pytest.raises(ValueError, match="not identifiable"):
        infer_full_temperature_tensor_from_thomson_projections(
            np.full(6, 100.0), q,
        )


def test_full_temperature_tensor_rejects_non_positive_definite_measurement_covariance():
    q, _, projected = _views()
    covariance = np.eye(q.shape[0])
    covariance[-1, -1] = 0.0
    with pytest.raises(ValueError, match="positive definite"):
        infer_full_temperature_tensor_from_thomson_projections(
            projected, q, projected_temperature_covariance_eV2=covariance,
        )


def test_unweighted_tensor_fit_does_not_claim_a_chi_square_probability():
    q, _, projected = _views()
    result = infer_full_temperature_tensor_from_thomson_projections(projected, q)
    assert result["consistency_p_value"] is None
    assert result["consistency_status"] == "not_testable_without_absolute_covariance_or_redundant_views"


def test_full_temperature_tensor_enforces_positive_semidefinite_boundary():
    q = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
    ])
    q /= np.linalg.norm(q, axis=1)[:, None]
    # The unconstrained six-view solution has eigenvalue -50 eV.
    projected = np.array([100.0, 100.0, 100.0, 250.0, 100.0, 100.0])
    result = infer_full_temperature_tensor_from_thomson_projections(projected, q)
    assert np.min(np.linalg.eigvalsh(result["electron_temperature_tensor_eV"])) >= -1e-10
    assert result["positive_semidefinite_constraint_active"] is True
    assert result["parameter_covariance_eV2"] is None
    assert result["parameter_covariance_status"] == "not_reported_psd_boundary"


def test_full_temperature_tensor_derives_pressure_tensor_from_independent_density():
    q, expected, projected = _views()
    density = 2.0e19
    result = infer_full_temperature_tensor_from_thomson_projections(
        projected, q,
        projected_temperature_covariance_eV2=0.04 * np.eye(q.shape[0]),
        electron_density_m3=density,
        electron_density_uncertainty_m3=0.1e19,
    )
    np.testing.assert_allclose(
        result["electron_pressure_tensor_Pa"],
        density * 1.602176634e-19 * expected,
        rtol=1e-10, atol=1e-10,
    )
    assert result["electron_pressure_parameter_covariance_Pa2"].shape == (6, 6)
    assert result["electron_pressure_covariance_status"] == (
        "independent_temperature_gls_and_density_uncertainty_delta_method"
    )


def test_full_temperature_tensor_does_not_infer_pressure_without_density():
    q, _, projected = _views()
    result = infer_full_temperature_tensor_from_thomson_projections(projected, q)
    assert result["electron_pressure_tensor_Pa"] is None
    assert result["electron_pressure_parameter_covariance_Pa2"] is None


def test_full_temperature_tensor_rejects_density_uncertainty_without_density():
    q, _, projected = _views()
    with pytest.raises(ValueError, match="requires electron_density_m3"):
        infer_full_temperature_tensor_from_thomson_projections(
            projected, q, electron_density_uncertainty_m3=1.0e17,
        )


def test_raw_spectral_fit_recovers_general_non_gyrotropic_temperature_tensor():
    from tokamak_analysis.thomson_workflow import analyze_thomson_spectra

    c, e, me = 299792458.0, 1.602176634e-19, 9.1093837015e-31
    wavelength = np.linspace(529.0, 535.0, 241)
    laser, angle = 532.0, np.pi/2.0
    q = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,-1,1]],dtype=float)
    q /= np.linalg.norm(q,axis=1)[:,None]
    expected = np.array([[800.0,90.0,-35.0],[90.0,420.0,55.0],[-35.0,55.0,260.0]])
    expected_flow_vector = np.array([1.2e4, -0.8e4, 0.6e4])
    flows = q @ expected_flow_vector
    qfactor=2*np.sin(angle/2)
    edges=np.empty(wavelength.size+1)
    edges[1:-1]=0.5*(wavelength[:-1]+wavelength[1:])
    edges[0]=wavelength[0]-0.5*(wavelength[1]-wavelength[0])
    edges[-1]=wavelength[-1]+0.5*(wavelength[-1]-wavelength[-2])
    from scipy.special import ndtr
    spectra=[]
    for direction,flow in zip(q,flows):
        tq=float(direction@expected@direction)
        centre=laser*(1-qfactor*flow/c)
        sigma=laser*qfactor*np.sqrt(e*tq/me)/c
        z=(edges-centre)/sigma
        probability=ndtr(z[1:])-ndtr(z[:-1])
        positive=z[:-1]>0
        probability[positive]=ndtr(-z[:-1][positive])-ndtr(-z[1:][positive])
        spectra.append(2.0*probability/np.diff(edges)+0.02)
    # A calibrated response is column-stochastic (true input bin -> measured
    # output bin). This deliberately asymmetric operator has non-unit row sums
    # and verifies that the forward model checks/applies the physical convention.
    response = np.eye(wavelength.size)
    response[1, 0] = 0.1
    response[0, 0] = 0.9
    response[0, 1] = 0.2
    response[1, 1] = 0.8
    widths = np.diff(edges)
    spectra = [(response @ (spectrum * widths)) / widths for spectrum in spectra]
    result=analyze_thomson_spectra(
        wavelength,np.asarray(spectra),model="general_temperature_tensor_maxwellian",
        laser_wavelength_nm=laser,
        scattering_angle_rad=angle,scattering_vector_unit=q,
        instrument_response_matrix=response,
        signal_uncertainty=np.full((len(q),len(wavelength)),2e-4),
        initial_temperature_tensor_eV=np.eye(3)*500.0,
        electron_density_m3=2.0e19,
        electron_density_uncertainty_m3=0.1e19,
        maximum_projected_rms_beta=0.1,
    )
    np.testing.assert_allclose(result["electron_temperature_tensor_eV"],expected,rtol=2e-3,atol=0.5)
    np.testing.assert_allclose(result["electron_velocity_projection_m_s"],flows,rtol=1e-5,atol=2.0)
    np.testing.assert_allclose(result["electron_velocity_vector_m_s"], expected_flow_vector, rtol=1e-5, atol=2.0)
    assert result["flow_model"].startswith("one_shared_cartesian_3d_bulk_velocity_vector")
    assert result["electron_temperature_flow_joint_covariance"].shape == (9, 9)
    assert result["electron_velocity_vector_covariance_m2_s2"].shape == (3, 3)
    assert result["view_dyad_rank"]==6
    assert result["jacobian_rank"]==result["jacobian_parameter_count"]
    assert result["reportable_as_inferred_parameter"] is True
    assert np.max(np.abs(result["residual_spectra"]))<1e-8
    assert result["temperature_tensor_parameter_covariance_eV2"].shape==(6,6)
    np.testing.assert_allclose(result["electron_pressure_tensor_Pa"],2.0e19*1.602176634e-19*expected,rtol=2e-3,atol=1e-5)
    assert result["electron_pressure_parameter_covariance_Pa2"].shape==(6,6)


def test_raw_spectral_tensor_fit_rejects_nonidentifiable_view_geometry():
    from tokamak_analysis.thomson_temperature_tensor import fit_general_temperature_tensor_thomson_spectra
    wavelength=np.linspace(529.0,535.0,41)
    spectra=np.ones((6,wavelength.size))
    q=np.tile([1.0,0.0,0.0],(6,1))
    with pytest.raises(ValueError,match="not identifiable"):
        fit_general_temperature_tensor_thomson_spectra(
            wavelength,spectra,laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi/2,scattering_vector_unit=q,
            instrument_response_matrix=np.eye(wavelength.size),
        )


def test_raw_tensor_fit_rejects_indefinite_model_discrepancy_even_if_measurement_covariance_masks_it():
    from tokamak_analysis.thomson_temperature_tensor import fit_general_temperature_tensor_thomson_spectra
    wavelength=np.linspace(529.0,535.0,41)
    spectra=np.ones((6,wavelength.size))
    q=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]],dtype=float)
    q/=np.linalg.norm(q,axis=1)[:,None]
    size=spectra.size
    bad=np.eye(size)
    bad[0,0]=-0.25
    with pytest.raises(ValueError,match="model_discrepancy_covariance must be positive semidefinite"):
        fit_general_temperature_tensor_thomson_spectra(
            wavelength,spectra,laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi/2,scattering_vector_unit=q,
            instrument_response_matrix=np.eye(wavelength.size),
            measurement_covariance=10.0*np.eye(size),
            model_discrepancy_covariance=bad,
        )


def test_raw_tensor_fit_rejects_asymmetric_measurement_covariance():
    from tokamak_analysis.thomson_temperature_tensor import fit_general_temperature_tensor_thomson_spectra
    wavelength=np.linspace(529.0,535.0,41)
    spectra=np.ones((6,wavelength.size))
    q=np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1]],dtype=float)
    q/=np.linalg.norm(q,axis=1)[:,None]
    covariance=np.eye(spectra.size)
    covariance[0,1]=0.1
    with pytest.raises(ValueError,match="measurement_covariance must be symmetric"):
        fit_general_temperature_tensor_thomson_spectra(
            wavelength,spectra,laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi/2,scattering_vector_unit=q,
            instrument_response_matrix=np.eye(wavelength.size),
            measurement_covariance=covariance,
        )



def test_raw_spectral_tensor_fit_requires_one_cartesian_flow_seed():
    from tokamak_analysis.thomson_temperature_tensor import fit_general_temperature_tensor_thomson_spectra

    wavelength = np.linspace(529.0, 535.0, 41)
    q, _, _ = _views()
    spectra = np.ones((q.shape[0], wavelength.size))
    with pytest.raises(ValueError, match="Cartesian 3-vector"):
        fit_general_temperature_tensor_thomson_spectra(
            wavelength, spectra,
            laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2,
            scattering_vector_unit=q,
            instrument_response_matrix=np.eye(wavelength.size),
            initial_electron_velocity_vector_m_s=[1.0, 2.0],
        )


def test_shared_flow_projection_is_exactly_consistent_across_views():
    from tokamak_analysis.thomson_temperature_tensor import fit_general_temperature_tensor_thomson_spectra

    wavelength = np.linspace(529.0, 535.0, 241)
    q, expected_tensor, _ = _views()
    velocity = np.array([1.1e4, -0.7e4, 0.4e4])
    c, e, me = 299792458.0, 1.602176634e-19, 9.1093837015e-31
    qfactor = 2.0 * np.sin(np.pi / 4.0)
    edges = np.empty(wavelength.size + 1)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    from scipy.special import ndtr

    spectra = []
    for direction in q:
        flow = float(direction @ velocity)
        tq = float(direction @ expected_tensor @ direction)
        centre = 532.0 * (1.0 - qfactor * flow / c)
        sigma = 532.0 * qfactor * np.sqrt(e * tq / me) / c
        z = (edges - centre) / sigma
        probability = ndtr(z[1:]) - ndtr(z[:-1])
        positive = z[:-1] > 0
        probability[positive] = ndtr(-z[:-1][positive]) - ndtr(-z[1:][positive])
        spectra.append(2.0 * probability / np.diff(edges) + 0.02)

    result = fit_general_temperature_tensor_thomson_spectra(
        wavelength, np.asarray(spectra),
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2,
        scattering_vector_unit=q,
        instrument_response_matrix=np.eye(wavelength.size),
        signal_uncertainty=np.full((q.shape[0], wavelength.size), 2e-4),
        initial_temperature_tensor_eV=np.eye(3) * 500.0,
    )
    np.testing.assert_allclose(result["electron_velocity_vector_m_s"], velocity, rtol=1e-5, atol=2.0)
    np.testing.assert_allclose(result["electron_velocity_projection_m_s"], q @ velocity, rtol=1e-5, atol=2.0)
    assert result["reportable_as_inferred_parameter"] is True



def test_tensor_spectral_fit_registry_exposes_shared_cartesian_flow_contract():
    from tokamak_analysis.analysis_registry import get_analysis

    contract = get_analysis(83)["execution_contract"]
    inputs = {item["name"] for item in contract["input_specs"]}
    outputs = {item["name"] for item in contract["output_specs"]}
    assert "initial_electron_velocity_vector_m_s" in inputs
    assert "electron_density_temperature_tensor_cross_covariance" in inputs
    assert {
        "electron_velocity_vector_m_s",
        "electron_velocity_vector_covariance_m2_s2",
        "electron_temperature_flow_joint_covariance",
        "electron_density_temperature_tensor_cross_covariance",
        "electron_pressure_covariance_equation",
    } <= outputs
    assert "one shared Cartesian bulk-flow vector" in contract["algorithm"]
    assert {
        "spectral_parameter_covariance_components",
        "measurement_parameter_covariance",
        "model_discrepancy_parameter_covariance",
        "systematic_parameter_covariance",
        "model_selection_uncertainty_status",
    } <= outputs



def test_raw_tensor_pressure_covariance_includes_density_temperature_cross_terms():
    from tokamak_analysis.thomson_temperature_tensor import fit_general_temperature_tensor_thomson_spectra

    wavelength = np.linspace(529.0, 535.0, 241)
    q, expected_tensor, _ = _views()
    velocity = np.array([1.0e4, -0.6e4, 0.3e4])
    c, e, me = 299792458.0, 1.602176634e-19, 9.1093837015e-31
    qfactor = 2.0 * np.sin(np.pi / 4.0)
    edges = np.empty(wavelength.size + 1)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    from scipy.special import ndtr

    spectra = []
    for direction in q:
        flow = float(direction @ velocity)
        tq = float(direction @ expected_tensor @ direction)
        centre = 532.0 * (1.0 - qfactor * flow / c)
        sigma = 532.0 * qfactor * np.sqrt(e * tq / me) / c
        z = (edges - centre) / sigma
        probability = ndtr(z[1:]) - ndtr(z[:-1])
        positive = z[:-1] > 0
        probability[positive] = ndtr(-z[:-1][positive]) - ndtr(-z[1:][positive])
        spectra.append(2.0 * probability / np.diff(edges) + 0.02)

    common = dict(
        wavelength_nm=wavelength,
        measured_spectra=np.asarray(spectra),
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2,
        scattering_vector_unit=q,
        instrument_response_matrix=np.eye(wavelength.size),
        signal_uncertainty=np.full((q.shape[0], wavelength.size), 2e-4),
        electron_density_m3=2.0e19,
        electron_density_uncertainty_m3=1.0e18,
    )
    independent = fit_general_temperature_tensor_thomson_spectra(**common)
    tensor_cov = independent["temperature_tensor_parameter_covariance_eV2"]
    sigma_n = common["electron_density_uncertainty_m3"]
    cross = 0.1 * sigma_n * np.sqrt(np.diag(tensor_cov))
    correlated = fit_general_temperature_tensor_thomson_spectra(
        **common,
        electron_density_temperature_tensor_cross_covariance=cross,
    )
    components = np.array([
        expected_tensor[0,0], expected_tensor[1,1], expected_tensor[2,2],
        expected_tensor[0,1], expected_tensor[0,2], expected_tensor[1,2],
    ])
    expected_covariance = (common["electron_density_m3"] * e)**2 * tensor_cov
    expected_covariance += (e * sigma_n)**2 * np.outer(components, components)
    expected_covariance += e**2 * common["electron_density_m3"] * (
        np.outer(components, cross) + np.outer(cross, components)
    )
    np.testing.assert_allclose(
        correlated["electron_pressure_parameter_covariance_Pa2"],
        expected_covariance,
        rtol=2e-5, atol=1e-20,
    )
    assert correlated["electron_pressure_covariance_status"] == (
        "joint_density_temperature_cross_covariance_delta_method"
    )


def test_raw_tensor_pressure_covariance_rejects_non_psd_density_temperature_joint_covariance():
    from scipy.special import ndtr
    from tokamak_analysis.thomson_temperature_tensor import fit_general_temperature_tensor_thomson_spectra

    wavelength = np.linspace(529.0, 535.0, 161)
    q, tensor, _ = _views()
    c, e, me = 299792458.0, 1.602176634e-19, 9.1093837015e-31
    edges = np.empty(wavelength.size + 1)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    spectra = []
    for direction in q:
        tq = float(direction @ tensor @ direction)
        sigma = 532.0 * 2.0 * np.sin(np.pi / 4.0) * np.sqrt(e * tq / me) / c
        z = (edges - 532.0) / sigma
        probability = ndtr(z[1:]) - ndtr(z[:-1])
        positive = z[:-1] > 0
        probability[positive] = ndtr(-z[:-1][positive]) - ndtr(-z[1:][positive])
        spectra.append(2.0 * probability / np.diff(edges) + 0.02)

    with pytest.raises(ValueError, match="joint density-temperature covariance"):
        fit_general_temperature_tensor_thomson_spectra(
            wavelength, np.asarray(spectra),
            laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2,
            scattering_vector_unit=q,
            instrument_response_matrix=np.eye(wavelength.size),
            signal_uncertainty=np.full((q.shape[0], wavelength.size), 2e-4),
            electron_density_m3=2.0e19,
            electron_density_uncertainty_m3=1.0e18,
            electron_density_temperature_tensor_cross_covariance=np.full(6, 1.0e40),
        )


def test_raw_tensor_fit_rejects_row_stochastic_but_non_column_stochastic_response():
    from tokamak_analysis.thomson_temperature_tensor import fit_general_temperature_tensor_thomson_spectra

    wavelength = np.linspace(529.0, 535.0, 41)
    q, _, _ = _views()
    spectra = np.ones((q.shape[0], wavelength.size))
    response = np.eye(wavelength.size)
    response[0, 0], response[0, 1] = 0.8, 0.2
    response[1, 0], response[1, 1] = 0.1, 0.9
    np.testing.assert_allclose(response.sum(axis=1), 1.0)
    assert not np.allclose(response.sum(axis=0), 1.0)
    with pytest.raises(ValueError, match="columns sum to one"):
        fit_general_temperature_tensor_thomson_spectra(
            wavelength, spectra,
            laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2,
            scattering_vector_unit=q,
            instrument_response_matrix=response,
        )


def test_raw_tensor_fit_separates_measurement_model_discrepancy_and_systematic_parameter_uncertainty():
    from scipy.special import ndtr
    from tokamak_analysis.thomson_temperature_tensor import fit_general_temperature_tensor_thomson_spectra

    wavelength = np.linspace(529.0, 535.0, 61)
    q, tensor, _ = _views()
    c, e, me = 299792458.0, 1.602176634e-19, 9.1093837015e-31
    edges = np.empty(wavelength.size + 1)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    spectra = []
    for direction in q:
        projected = float(direction @ tensor @ direction)
        sigma = 532.0 * 2.0 * np.sin(np.pi / 4.0) * np.sqrt(e * projected / me) / c
        z = (edges - 532.0) / sigma
        probability = ndtr(z[1:]) - ndtr(z[:-1])
        positive = z[:-1] > 0.0
        probability[positive] = ndtr(-z[:-1][positive]) - ndtr(-z[1:][positive])
        spectra.append(1.7 * probability / np.diff(edges) + 0.025)
    spectra = np.asarray(spectra)
    n_data = spectra.size
    measurement = np.eye(n_data) * (2.0e-4)**2
    discrepancy_shape = np.linspace(-1.0, 1.0, n_data)
    discrepancy = (3.0e-5)**2 * np.outer(discrepancy_shape, discrepancy_shape)
    systematic = np.eye(n_data) * (1.0e-5)**2

    result = fit_general_temperature_tensor_thomson_spectra(
        wavelength, spectra,
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        scattering_vector_unit=q,
        instrument_response_matrix=np.eye(wavelength.size),
        measurement_covariance=measurement,
        model_discrepancy_covariance=discrepancy,
        systematic_covariance=systematic,
        initial_temperature_tensor_eV=np.eye(3) * 500.0,
    )
    components = result["spectral_parameter_covariance_components"]
    assert set(components) == {"measurement", "model_discrepancy", "systematic"}
    assert result["model_selection_uncertainty_included"] is False
    assert result["model_selection_uncertainty_status"].startswith("not_included")
    for contribution in components.values():
        assert contribution.shape == result["parameter_covariance"].shape
        assert np.min(np.linalg.eigvalsh(contribution)) >= -1e-10 * max(
            np.max(np.abs(contribution)), 1e-300
        )
    summed = sum(components.values())
    np.testing.assert_allclose(
        summed, result["parameter_covariance"], rtol=2e-5, atol=1e-18
    )
    assert np.trace(components["model_discrepancy"]) > 0.0
    assert np.trace(components["systematic"]) > 0.0


def test_full_temperature_tensor_propagates_multi_view_scattering_geometry_covariance():
    q, expected, projected = _views()
    measurement_covariance = 0.04 * np.eye(q.shape[0])
    geometry_covariance = 1.0e-8 * np.eye(3*q.shape[0])
    result = infer_full_temperature_tensor_from_thomson_projections(
        projected, q,
        projected_temperature_covariance_eV2=measurement_covariance,
        scattering_vector_covariance=geometry_covariance,
    )
    np.testing.assert_allclose(
        result["electron_temperature_tensor_eV"], expected,
        rtol=1e-10, atol=1e-10,
    )
    geometry = result["scattering_geometry_parameter_covariance_eV2"]
    assert geometry.shape == (6, 6)
    assert np.all(np.isfinite(geometry))
    assert np.min(np.linalg.eigvalsh(geometry)) >= -1e-12
    assert np.trace(geometry) > 0.0
    total = result["parameter_covariance_with_geometry_eV2"]
    np.testing.assert_allclose(
        total,
        result["parameter_covariance_eV2"] + geometry,
        rtol=1e-10, atol=1e-12,
    )
    assert result["geometry_covariance_status"] == (
        "finite_difference_directional_delta_method_added_to_parameter_covariance"
    )


def test_full_temperature_tensor_geometry_covariance_rejects_invalid_contracts():
    q, _, projected = _views()
    with pytest.raises(ValueError, match="shape"):
        infer_full_temperature_tensor_from_thomson_projections(
            projected, q, scattering_vector_covariance=np.eye(3),
        )
    invalid = np.eye(3*q.shape[0])
    invalid[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        infer_full_temperature_tensor_from_thomson_projections(
            projected, q, scattering_vector_covariance=invalid,
        )

def test_full_temperature_tensor_rejects_asymmetric_measurement_covariance():
    q, _, projected = _views()
    covariance = np.eye(q.shape[0]) * 0.04
    covariance[0, 1] = 0.01
    with pytest.raises(ValueError, match="covariance must be symmetric"):
        infer_full_temperature_tensor_from_thomson_projections(
            projected, q, projected_temperature_covariance_eV2=covariance,
        )


def test_full_temperature_tensor_accepts_roundoff_level_covariance_asymmetry():
    q, expected, projected = _views()
    covariance = np.eye(q.shape[0]) * 0.04
    covariance[0, 1] = 1.0e-14
    result = infer_full_temperature_tensor_from_thomson_projections(
        projected, q, projected_temperature_covariance_eV2=covariance,
    )
    np.testing.assert_allclose(
        result["electron_temperature_tensor_eV"], expected, rtol=1e-9, atol=1e-9
    )
