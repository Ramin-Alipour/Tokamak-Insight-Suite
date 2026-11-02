import numpy as np
import pytest
from tokamak_analysis.thomson_spatial_mixing import reconstruct_profile_with_spatial_mixing

def test_psf_aware_profile_inverse_recovers_profile_with_fixed_operator():
    radius=np.linspace(0,1,9); profile=2+0.4*np.sin(np.pi*radius)
    mixing=0.8*np.eye(9)+0.1*np.roll(np.eye(9),1,axis=1)+0.1*np.roll(np.eye(9),-1,axis=1)
    result=reconstruct_profile_with_spatial_mixing(mixing@profile,mixing,radius,
        uncertainty=np.full(9,0.01),regularization_strength=1e-8,
        parameter_name="electron_temperature",parameter_unit="eV")
    assert result["analysis_status"]=="completed"
    assert result["profile"]==pytest.approx(profile,rel=2e-3,abs=2e-3)
    assert result["profile_covariance"].shape==(9,9)
    assert result["spatial_mixing_operator"]=="researcher_supplied_fixed_forward_operator"
    assert result["experimental_validation"]=="not_performed; software/numerical inference only"

def test_psf_profile_inverse_gcv_and_positive_constraint():
    radius=np.linspace(0,1,7); measured=np.array([0,1,2,2,1,.5,0.])
    result=reconstruct_profile_with_spatial_mixing(measured,np.eye(7),radius,
        covariance=np.eye(7)*0.04,positive=True,gcv_grid_size=31)
    assert result["regularization"]["selection"]=="generalized_cross_validation"
    assert np.all(result["profile"]>=0)
    assert result["profile_covariance"].shape==(7,7)

def test_psf_profile_inverse_rejects_invalid_covariance_and_rank():
    radius=np.linspace(0,1,5)
    with pytest.raises(ValueError,match="positive definite"):
        reconstruct_profile_with_spatial_mixing(np.ones(5),np.eye(5),radius,covariance=np.zeros((5,5)))
    bad=np.eye(5); bad[-1]=bad[0]
    with pytest.raises(ValueError,match="rank deficient"):
        reconstruct_profile_with_spatial_mixing(np.ones(5),bad,radius)


def test_psf_profile_inverse_propagates_correlated_radius_uncertainty_reproducibly():
    radius = np.linspace(0.0, 1.0, 7)
    profile = 2.0 + 0.3*np.sin(np.pi*radius)
    mixing = 0.8*np.eye(7) + 0.1*np.roll(np.eye(7), 1, axis=1) + 0.1*np.roll(np.eye(7), -1, axis=1)
    coordinate_covariance = np.eye(7)*(2.0e-5)**2
    first = reconstruct_profile_with_spatial_mixing(
        mixing@profile, mixing, radius, uncertainty=np.full(7, 0.02),
        regularization_strength=1e-5, radius_covariance_m2=coordinate_covariance,
        radius_mc_samples=40, radius_mc_seed=1234,
    )
    second = reconstruct_profile_with_spatial_mixing(
        mixing@profile, mixing, radius, uncertainty=np.full(7, 0.02),
        regularization_strength=1e-5, radius_covariance_m2=coordinate_covariance,
        radius_mc_samples=40, radius_mc_seed=1234,
    )
    assert first["coordinate_uncertainty_propagation_status"] == "propagated_by_monte_carlo_with_fixed_selected_regularization"
    assert first["coordinate_uncertainty_successful_draws"] >= 32
    assert first["coordinate_induced_profile_covariance"].shape == (7, 7)
    assert first["profile_covariance"] == pytest.approx(second["profile_covariance"])
    assert np.all(np.linalg.eigvalsh(first["profile_covariance"]) >= -1e-12)


def test_psf_profile_inverse_rejects_non_psd_radius_covariance():
    radius = np.linspace(0.0, 1.0, 5)
    invalid = np.eye(5)
    invalid[0, 1] = invalid[1, 0] = 2.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        reconstruct_profile_with_spatial_mixing(
            np.ones(5), np.eye(5), radius, radius_covariance_m2=invalid,
            radius_mc_samples=20,
        )



def test_psf_profile_inverse_propagates_spatial_mixing_matrix_covariance():
    radius = np.linspace(0.0, 1.0, 7)
    profile = 2.0 + 0.3*np.sin(np.pi*radius)
    mixing = 0.8*np.eye(7) + 0.1*np.roll(np.eye(7), 1, axis=1) + 0.1*np.roll(np.eye(7), -1, axis=1) + 0.01*np.ones((7,7))
    mixing_covariance = np.eye(mixing.size) * (2.0e-4)**2
    result = reconstruct_profile_with_spatial_mixing(
        mixing@profile, mixing, radius, uncertainty=np.full(7, 0.02),
        regularization_strength=1e-5,
        spatial_mixing_covariance=mixing_covariance,
        mixing_mc_samples=40, mixing_mc_seed=4321,
    )
    assert result["spatial_mixing_uncertainty_propagation_status"] == (
        "propagated_by_monte_carlo_with_fixed_selected_regularization"
    )
    assert result["spatial_mixing_uncertainty_successful_draws"] >= 32
    assert result["spatial_mixing_induced_profile_covariance"].shape == (7, 7)
    assert np.all(np.linalg.eigvalsh(result["profile_covariance"]) >= -1e-12)


def test_psf_profile_inverse_rejects_invalid_spatial_mixing_covariance():
    radius = np.linspace(0.0, 1.0, 5)
    invalid = np.eye(25)
    invalid[0, 1] = invalid[1, 0] = 2.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        reconstruct_profile_with_spatial_mixing(
            np.ones(5), np.eye(5), radius,
            regularization_strength=1e-4,
            spatial_mixing_covariance=invalid, mixing_mc_samples=20,
        )
