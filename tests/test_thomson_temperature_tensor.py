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
    from tokamak_analysis.thomson_temperature_tensor import fit_general_temperature_tensor_thomson_spectra

    c, e, me = 299792458.0, 1.602176634e-19, 9.1093837015e-31
    wavelength = np.linspace(529.0, 535.0, 241)
    laser, angle = 532.0, np.pi/2.0
    q = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1,-1,1]],dtype=float)
    q /= np.linalg.norm(q,axis=1)[:,None]
    expected = np.array([[800.0,90.0,-35.0],[90.0,420.0,55.0],[-35.0,55.0,260.0]])
    flows = np.array([1.0e4,-1.5e4,2.0e4,-0.5e4,0.7e4,-0.9e4,1.2e4])
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
    result=fit_general_temperature_tensor_thomson_spectra(
        wavelength,np.asarray(spectra),laser_wavelength_nm=laser,
        scattering_angle_rad=angle,scattering_vector_unit=q,
        instrument_response_matrix=np.eye(wavelength.size),
        signal_uncertainty=np.full((len(q),len(wavelength)),2e-4),
        initial_temperature_tensor_eV=np.eye(3)*500.0,
        maximum_projected_rms_beta=0.1,
    )
    np.testing.assert_allclose(result["electron_temperature_tensor_eV"],expected,rtol=2e-3,atol=0.5)
    np.testing.assert_allclose(result["electron_velocity_projection_m_s"],flows,rtol=1e-5,atol=2.0)
    assert result["view_dyad_rank"]==6
    assert result["jacobian_rank"]==result["jacobian_parameter_count"]
    assert result["reportable_as_inferred_parameter"] is True
    assert np.max(np.abs(result["residual_spectra"]))<1e-8
    assert result["temperature_tensor_parameter_covariance_eV2"].shape==(6,6)


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
