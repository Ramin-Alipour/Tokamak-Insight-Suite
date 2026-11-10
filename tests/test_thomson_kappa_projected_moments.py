import numpy as np
import pytest

from tis.physics.shared.thomson_inference import _projected_kappa_moment_closure


def test_projected_kappa_moments_match_student_t_analytic_relations():
    sigma_nm, laser_nm, angle, kappa = 0.08, 532.0, np.pi/2.0, 4.0
    covariance = np.zeros((6,6))
    covariance[2,2] = 1.0e-8
    covariance[3,3] = 2.5e-3
    covariance[2,3] = covariance[3,2] = 1.0e-6
    result = _projected_kappa_moment_closure(sigma_nm,kappa,laser_nm,angle,covariance)
    sigma_v = 299792458.0*sigma_nm/(laser_nm*2.0*np.sin(angle/2.0))
    expected_mu4 = 3.0*sigma_v**4*(2.0*kappa-3.0)/(2.0*kappa-5.0)
    assert result['student_t_degrees_of_freedom'] == pytest.approx(7.0)
    assert result['velocity_variance_m2_s2'] == pytest.approx(sigma_v**2)
    assert result['central_third_velocity_moment_m3_s3'] == 0.0
    assert result['central_fourth_velocity_moment_m4_s4'] == pytest.approx(expected_mu4)
    assert result['excess_kurtosis'] == pytest.approx(2.0)
    assert result['central_fourth_moment_standard_uncertainty_m4_s4'] > 0.0
    assert result['excess_kurtosis_standard_uncertainty'] == pytest.approx(4.0/3.0*np.sqrt(covariance[3,3]))


@pytest.mark.parametrize('kappa,third,fourth', [
    (1.7,'divergent_absolute_third_moment_for_kappa_at_or_below_2','divergent_fourth_moment_for_kappa_at_or_below_2p5'),
    (2.2,'finite_zero_by_symmetric_equilibrium_kappa_model','divergent_fourth_moment_for_kappa_at_or_below_2p5'),
    (2.5,'finite_zero_by_symmetric_equilibrium_kappa_model','divergent_fourth_moment_for_kappa_at_or_below_2p5'),
])
def test_projected_kappa_never_reports_divergent_moments(kappa,third,fourth):
    result = _projected_kappa_moment_closure(0.08,kappa,532.0,np.pi/2.0)
    assert result['central_third_moment_status'] == third
    assert result['fourth_moment_status'] == fourth
    assert result['central_fourth_velocity_moment_m4_s4'] is None
    assert result['excess_kurtosis'] is None
    assert (result['central_third_velocity_moment_m3_s3'] is None) == (kappa <= 2.0)


def test_projected_kappa_rejects_asymmetric_fit_covariance():
    covariance = np.eye(4)
    covariance[2,3] = 0.5
    with pytest.raises(ValueError,match='symmetric'):
        _projected_kappa_moment_closure(0.08,4.0,532.0,np.pi/2.0,covariance)
