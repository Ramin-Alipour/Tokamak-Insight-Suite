import numpy as np

from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics


def test_full_temperature_tensor_is_reachable_through_public_analysis1_dispatcher():
    q = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ])
    q /= np.linalg.norm(q, axis=1)[:, None]
    expected = np.array([[800.0, 90.0, -35.0],
                         [90.0, 420.0, 55.0],
                         [-35.0, 55.0, 260.0]])
    projected = np.einsum("vi,ij,vj->v", q, expected, q)
    result = run_advanced_thomson_physics(
        "thomson_full_temperature_tensor",
        projected, q,
        projected_temperature_covariance_eV2=0.04*np.eye(q.shape[0]),
    )
    np.testing.assert_allclose(result["electron_temperature_tensor_eV"], expected, rtol=1e-10, atol=1e-10)
    assert result["public_physics_entrypoint"] == "thomson_full_temperature_tensor"
    assert result["reportable_as_inferred_parameter"] is True


def test_pressure_tensor_closure_is_exposed_through_public_analysis1_dispatcher():
    q = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ])
    q /= np.linalg.norm(q, axis=1)[:, None]
    expected = np.array([[800.0, 90.0, -35.0],
                         [90.0, 420.0, 55.0],
                         [-35.0, 55.0, 260.0]])
    projected = np.einsum("vi,ij,vj->v", q, expected, q)
    result = run_advanced_thomson_physics(
        "thomson_full_temperature_tensor",
        projected, q,
        projected_temperature_covariance_eV2=0.04*np.eye(q.shape[0]),
        electron_density_m3=2.0e19,
        electron_density_uncertainty_m3=1.0e18,
    )
    np.testing.assert_allclose(
        result["electron_pressure_tensor_Pa"],
        2.0e19 * 1.602176634e-19 * expected,
        rtol=1e-10, atol=1e-10,
    )
    assert result["public_physics_entrypoint"] == "thomson_full_temperature_tensor"



def test_raw_temperature_tensor_fit_shared_flow_is_reachable_through_public_dispatcher():
    from scipy.special import ndtr

    c, e, me = 299792458.0, 1.602176634e-19, 9.1093837015e-31
    wavelength = np.linspace(529.0, 535.0, 161)
    q = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ])
    q /= np.linalg.norm(q, axis=1)[:, None]
    tensor = np.array([[800.0, 90.0, -35.0],
                       [90.0, 420.0, 55.0],
                       [-35.0, 55.0, 260.0]])
    velocity = np.array([1.0e4, -0.7e4, 0.5e4])
    edges = np.empty(wavelength.size + 1)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    qfactor = 2.0 * np.sin(np.pi / 4.0)
    spectra = []
    for direction in q:
        flow = float(direction @ velocity)
        tq = float(direction @ tensor @ direction)
        centre = 532.0 * (1.0 - qfactor * flow / c)
        sigma = 532.0 * qfactor * np.sqrt(e * tq / me) / c
        z = (edges - centre) / sigma
        probability = ndtr(z[1:]) - ndtr(z[:-1])
        positive = z[:-1] > 0
        probability[positive] = ndtr(-z[:-1][positive]) - ndtr(-z[1:][positive])
        spectra.append(2.0 * probability / np.diff(edges) + 0.02)

    result = run_advanced_thomson_physics(
        "thomson_full_temperature_tensor_spectral_fit",
        wavelength, np.asarray(spectra),
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2,
        scattering_vector_unit=q,
        instrument_response_matrix=np.eye(wavelength.size),
        signal_uncertainty=np.full((q.shape[0], wavelength.size), 2e-4),
        electron_density_m3=2.0e19,
        electron_density_uncertainty_m3=1.0e18,
        electron_density_temperature_tensor_cross_covariance=np.zeros(6),
    )
    np.testing.assert_allclose(result["electron_velocity_vector_m_s"], velocity, rtol=2e-4, atol=5.0)
    assert result["electron_pressure_covariance_status"] == "joint_density_temperature_cross_covariance_delta_method"
    np.testing.assert_allclose(result["electron_velocity_projection_m_s"], q @ velocity, rtol=2e-4, atol=5.0)
    assert result["public_physics_entrypoint"] == "thomson_full_temperature_tensor_spectral_fit"



def test_raw_temperature_tensor_fit_propagates_scattering_geometry_covariance():
    from scipy.special import ndtr

    c, e, me = 299792458.0, 1.602176634e-19, 9.1093837015e-31
    wavelength = np.linspace(529.0, 535.0, 161)
    q = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ])
    q /= np.linalg.norm(q, axis=1)[:, None]
    tensor = np.array([[800.0, 90.0, -35.0],
                       [90.0, 420.0, 55.0],
                       [-35.0, 55.0, 260.0]])
    velocity = np.array([1.0e4, -0.7e4, 0.5e4])
    edges = np.empty(wavelength.size + 1)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    qfactor = 2.0 * np.sin(np.pi / 4.0)
    spectra = []
    for direction in q:
        flow = float(direction @ velocity)
        tq = float(direction @ tensor @ direction)
        centre = 532.0 * (1.0 - qfactor * flow / c)
        sigma = 532.0 * qfactor * np.sqrt(e * tq / me) / c
        z = (edges - centre) / sigma
        probability = ndtr(z[1:]) - ndtr(z[:-1])
        positive = z[:-1] > 0
        probability[positive] = ndtr(-z[:-1][positive]) - ndtr(-z[1:][positive])
        spectra.append(2.0 * probability / np.diff(edges) + 0.02)
    spectra = np.asarray(spectra)
    sigma = np.full_like(spectra, 2e-4)
    tangent_covariance = np.zeros((3*q.shape[0], 3*q.shape[0]))
    for view, direction in enumerate(q):
        tangent_covariance[3*view:3*view+3, 3*view:3*view+3] = (
            2.5e-7 * (np.eye(3) - np.outer(direction, direction))
        )

    baseline = run_advanced_thomson_physics(
        "thomson_full_temperature_tensor_spectral_fit",
        wavelength, spectra, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2, scattering_vector_unit=q,
        instrument_response_matrix=np.eye(wavelength.size),
        signal_uncertainty=sigma,
    )
    propagated = run_advanced_thomson_physics(
        "thomson_full_temperature_tensor_spectral_fit",
        wavelength, spectra, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2, scattering_vector_unit=q,
        instrument_response_matrix=np.eye(wavelength.size),
        signal_uncertainty=sigma,
        scattering_vector_covariance=tangent_covariance,
        spectral_geometry_cross_covariance=np.zeros((spectra.size, 3*q.shape[0])),
    )
    assert propagated["scattering_vector_covariance_used"] is True
    assert propagated["spectral_geometry_cross_covariance_used"] is True
    assert propagated["geometry_parameter_covariance"].shape == baseline["parameter_covariance"].shape
    assert np.trace(propagated["geometry_parameter_covariance"]) > 0.0
    assert np.trace(propagated["parameter_covariance"]) > np.trace(baseline["parameter_covariance"])
    assert "tangent_plane_scattering_geometry_covariance" in propagated["covariance_status"]
