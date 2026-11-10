import numpy as np
import pytest

from tis.products.thomson.physics.thomson_scattering import (
    infer_thomson_channel_path_averaged_densities,
    infer_thomson_path_averaged_density,
)


def _channel_case():
    path = np.array([0.0, 0.2, 0.7, 1.0])
    incident = np.array([[8.0, 9.0, 7.0, 6.0], [5.0, 6.5, 8.0, 9.0]])
    cross_section = np.array([[6.0e-29, 6.1e-29, 6.2e-29, 6.0e-29], [5.8e-29, 5.9e-29, 6.0e-29, 6.1e-29]])
    solid_angle = np.array([[0.012, 0.011, 0.010, 0.009], [0.008, 0.009, 0.010, 0.011]])
    efficiency = np.array([[0.42, 0.43, 0.41, 0.40], [0.35, 0.36, 0.38, 0.39]])
    return path, incident, cross_section, solid_angle, efficiency


def test_channel_path_density_recovers_distinct_kernel_weighted_densities():
    path, incident, cross_section, solid_angle, efficiency = _channel_case()
    truth = np.array([2.1e19, 4.3e19])
    kernels = incident * cross_section * solid_angle * efficiency
    measured = truth * np.trapz(kernels, path, axis=1)
    result = infer_thomson_path_averaged_density(measured, path, incident, cross_section, solid_angle, efficiency)
    np.testing.assert_allclose(result["electron_density_path_averaged_m3"], truth, rtol=2e-14)
    assert result["channel_count"] == 2
    assert result["density_interpretation"].startswith("one_measurement_kernel_weighted_path_average_per_channel")
    assert result["electron_density_covariance_m6"] is None
    np.testing.assert_allclose(np.trapz(result["normalized_path_weight"], path, axis=1), 1.0, rtol=2e-14)


def test_channel_path_density_propagates_full_cross_channel_covariance():
    path, incident, cross_section, solid_angle, efficiency = _channel_case()
    truth = np.array([2.1e19, 4.3e19])
    kernels = incident * cross_section * solid_angle * efficiency
    measured = truth * np.trapz(kernels, path, axis=1)
    n_channel, n_path = incident.shape
    n_input = n_channel + 4 * n_channel * n_path
    covariance = np.zeros((n_input, n_input))
    covariance[:n_channel, :n_channel] = np.array([[1.0e30, 2.0e29], [2.0e29, 1.5e30]])
    covariance[n_channel:, n_channel:] = np.diag(np.full(4 * n_channel * n_path, 1.0e-8))
    result = infer_thomson_channel_path_averaged_densities(
        measured, path, incident, cross_section, solid_angle, efficiency,
        joint_input_covariance=covariance,
    )
    propagated = result["electron_density_covariance_m6"]
    assert propagated.shape == (n_channel, n_channel)
    assert np.all(np.isfinite(propagated))
    assert np.linalg.eigvalsh(propagated).min() >= -1.0e-10 * np.max(np.abs(propagated))
    assert result["electron_density_path_averaged_uncertainty_m3"].shape == (n_channel,)
    assert propagated[0, 1] != 0.0


def test_channel_path_density_rejects_invalid_joint_covariance():
    path, incident, cross_section, solid_angle, efficiency = _channel_case()
    measured = np.array([1.0e-8, 2.0e-8])
    n_input = 2 + 4 * 2 * path.size
    covariance = np.eye(n_input)
    covariance[0, 1] = covariance[1, 0] = 2.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        infer_thomson_channel_path_averaged_densities(
            measured, path, incident, cross_section, solid_angle, efficiency,
            joint_input_covariance=covariance,
        )


def test_channel_path_density_accepts_channel_specific_path_coordinates():
    path, incident, cross_section, solid_angle, efficiency = _channel_case()
    paths = np.vstack((path, path + 0.1))
    truth = np.array([2.1e19, 4.3e19])
    kernels = incident * cross_section * solid_angle * efficiency
    measured = np.array([truth[i] * np.trapz(kernels[i], paths[i]) for i in range(2)])
    result = infer_thomson_channel_path_averaged_densities(
        measured, paths, incident, cross_section, solid_angle, efficiency,
    )
    np.testing.assert_allclose(result["electron_density_path_averaged_m3"], truth, rtol=2e-14)


def test_channel_path_density_is_reachable_through_product_api_entry_87():
    from tis.products.thomson.api import infer_path_averaged_density
    path, incident, cross_section, solid_angle, efficiency = _channel_case()
    truth = np.array([1.8e19, 3.2e19])
    kernels = incident * cross_section * solid_angle * efficiency
    measured = truth * np.trapz(kernels, path, axis=1)
    result = infer_path_averaged_density(
        measured, path, incident, cross_section, solid_angle, efficiency,
    )
    np.testing.assert_allclose(result["electron_density_path_averaged_m3"], truth, rtol=2e-14)
