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
