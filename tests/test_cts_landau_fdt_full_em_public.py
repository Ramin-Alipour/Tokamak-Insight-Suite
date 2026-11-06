"""Public Full-EM Landau--FDT assembly contract regressions.

These tests exercise the public Analysis-1 dispatcher, not only the lower-level
constitutive projection. They establish API reachability and declared scope;
they are not experimental validation or a magnetized CTS three-wave vertex.
"""
import numpy as np
import pytest

from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics


def _inputs():
    axis = np.linspace(-4.0, 4.0, 3)
    species = [
        {
            "name": "electrons",
            "density_m3": 1.0e18,
            "mass_kg": 9.1093837015e-31,
            "charge_C": -1.602176634e-19,
        },
        {
            "name": "ions",
            "density_m3": 1.0e18,
            "mass_kg": 1.67262192369e-27,
            "charge_C": 1.602176634e-19,
        },
    ]
    return {
        "angular_frequency_rad_s": np.array([1.0e5, 2.0e5]),
        "normalized_velocity_x": axis,
        "normalized_velocity_y": axis,
        "normalized_velocity_z": axis,
        "species": species,
        "common_temperature_J": 100.0 * 1.602176634e-19,
        "wave_number_m_inverse": 1.0e-3,
        "coulomb_logarithm": 12.0,
    }


def test_public_full_em_landau_fdt_dispatch_assembles_and_projects_response():
    result = run_advanced_thomson_physics(
        "landau_fdt_full_em_constitutive", **_inputs()
    )

    assert result["public_physics_entrypoint"] == "landau_fdt_full_em_constitutive"
    assert result["automatic_model_substitution"] is False
    assert result["conductivity_tensor_S_m"].shape == (2, 3, 3)
    assert result["relative_dielectric_tensor"].shape == (2, 3, 3)
    assert result["current_fluctuation_spectral_matrix"].shape == (2, 3, 3)
    assert np.all(np.isfinite(result["conductivity_tensor_S_m"]))
    assert np.all(np.isfinite(result["relative_dielectric_tensor"]))
    assert np.all(np.isfinite(result["current_fluctuation_spectral_matrix"]))
    assert result["cts_three_wave_vertex_connected"] is False
    assert result["quantitative_release_gate"]["magnetized_response_required_for_magnetized_plasma"] is True

    for matrix in result["current_fluctuation_spectral_matrix"]:
        np.testing.assert_allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-18)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-9 * max(
            np.linalg.norm(matrix, ord=2), np.finfo(float).tiny
        )


def test_public_full_em_landau_fdt_rejects_nonzero_flow_outside_its_domain():
    inputs = _inputs()
    inputs["common_bulk_flow_m_s"] = (1.0e3, 0.0, 0.0)
    with pytest.raises(ValueError, match="zero-flow"):
        run_advanced_thomson_physics(
            "landau_fdt_full_em_constitutive", **inputs
        )


def test_landau_current_noise_propagates_to_psd_electric_field_fluctuations():
    from tokamak_analysis.cts_landau_em import landau_full_em_electric_field_fluctuation_spectrum

    omega = np.array([1.0e5, 2.0e5])
    current_noise = np.broadcast_to(np.eye(3, dtype=complex), (2, 3, 3)).copy()
    response = {
        "angular_frequency_rad_s": omega,
        "current_fluctuation_spectral_matrix": current_noise,
        "relative_dielectric_tensor": np.broadcast_to(
            2.0*np.eye(3, dtype=complex), (2, 3, 3)
        ).copy(),
    }
    result = landau_full_em_electric_field_fluctuation_spectrum(
        response, 1.0e-3, propagation_direction=(1.0, 2.0, 0.5)
    )
    spectrum = result["electric_field_fluctuation_spectral_matrix_V2_m2_per_rad_s"]
    assert spectrum.shape == (2, 3, 3)
    for matrix in spectrum:
        np.testing.assert_allclose(matrix, matrix.conj().T, rtol=1e-12, atol=1e-30)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-10*np.linalg.norm(matrix, 2)
    assert np.all(np.isfinite(result["maxwell_operator_condition_number"]))


def test_landau_current_noise_rejects_an_undamped_maxwell_pole():
    from tokamak_analysis.cts_landau_em import landau_full_em_electric_field_fluctuation_spectrum

    omega = np.array([1.0e6])
    response = {
        "angular_frequency_rad_s": omega,
        "current_fluctuation_spectral_matrix": np.eye(3, dtype=complex)[None, :, :],
        "relative_dielectric_tensor": np.eye(3, dtype=complex)[None, :, :],
    }
    # For propagation along x, the longitudinal eigenvalue is -k0^2 and
    # transverse eigenvalues are k^2-k0^2; choose k=omega/c to hit a pole.
    with np.testing.assert_raises(np.linalg.LinAlgError):
        landau_full_em_electric_field_fluctuation_spectrum(
            response, omega[0]/299792458.0, propagation_direction=(1.0, 0.0, 0.0)
        )


def test_landau_current_noise_propagates_to_electric_magnetic_fluctuations_and_poynting_flux():
    from tokamak_analysis.cts_landau_em import landau_full_em_electric_field_fluctuation_spectrum

    omega = np.array([1.0e5, 2.0e5])
    current_noise = np.broadcast_to(np.eye(3, dtype=complex), (2, 3, 3)).copy()
    response = {
        "angular_frequency_rad_s": omega,
        "current_fluctuation_spectral_matrix": current_noise,
        "relative_dielectric_tensor": np.broadcast_to(2.0*np.eye(3), (2, 3, 3)).astype(complex),
    }
    result = landau_full_em_electric_field_fluctuation_spectrum(
        response, 1.0e-3, propagation_direction=(1.0, 2.0, 0.5)
    )
    electric = result["electric_field_fluctuation_spectral_matrix_V2_m2_per_rad_s"]
    magnetic = result["magnetic_field_fluctuation_spectral_matrix_T2_per_rad_s"]
    assert electric.shape == magnetic.shape == (2, 3, 3)
    for tensor in (electric, magnetic):
        for matrix in tensor:
            np.testing.assert_allclose(matrix, matrix.conj().T, rtol=1e-12, atol=1e-40)
            assert np.linalg.eigvalsh(matrix).min() >= -1e-10*np.linalg.norm(matrix, 2)
    assert result["electric_magnetic_cross_spectral_matrix"].shape == (2, 3, 3)
    poynting = result["spectral_poynting_flux_W_m2_per_rad_s"]
    assert poynting.shape == (2, 3)
    mu0 = 4.0e-7*np.pi
    levi = np.zeros((3, 3, 3))
    levi[0, 1, 2] = levi[1, 2, 0] = levi[2, 0, 1] = 1.0
    levi[0, 2, 1] = levi[2, 1, 0] = levi[1, 0, 2] = -1.0
    expected = 0.5/mu0*np.real(np.einsum("ijk,fjk->fi", levi, result["electric_magnetic_cross_spectral_matrix"]))
    np.testing.assert_allclose(poynting, expected)


def test_full_em_field_fluctuations_return_transverse_stokes_and_polarization_bounds():
    from tokamak_analysis.cts_landau_em import landau_full_em_electric_field_fluctuation_spectrum

    omega = np.array([1.0e5, 2.0e5])
    response = {
        "angular_frequency_rad_s": omega,
        "current_fluctuation_spectral_matrix": np.broadcast_to(
            np.eye(3, dtype=complex), (2, 3, 3)
        ).copy(),
        "relative_dielectric_tensor": np.broadcast_to(
            2.0*np.eye(3, dtype=complex), (2, 3, 3)
        ).copy(),
    }
    result = landau_full_em_electric_field_fluctuation_spectrum(
        response, 1.0e-3, propagation_direction=(1.0, 2.0, 0.5)
    )
    stokes = result["transverse_stokes_spectral_density_IQUV"]
    coherency = result["transverse_electric_coherency_spectral_matrix_V2_m2_per_rad_s"]
    degree = result["degree_of_linear_and_circular_polarization"]
    assert stokes.shape == (2, 4)
    assert coherency.shape == (2, 2, 2)
    assert np.all(stokes[:, 0] >= 0.0)
    assert np.all((degree >= 0.0) & (degree <= 1.0 + 1e-8))
    np.testing.assert_allclose(stokes[:, 1:], 0.0, atol=1e-25)


@pytest.mark.parametrize("bad_matrix, message", [
    (np.array([[1.0, 1.0j, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), "Hermitian"),
    (np.diag([1.0, 1.0, -0.1]).astype(complex), "positive semidefinite"),
])
def test_full_em_field_fluctuations_reject_nonphysical_current_covariance(bad_matrix, message):
    from tokamak_analysis.cts_landau_em import landau_full_em_electric_field_fluctuation_spectrum

    response = {
        "angular_frequency_rad_s": np.array([1.0e5]),
        "current_fluctuation_spectral_matrix": bad_matrix[None, :, :],
        "relative_dielectric_tensor": (2.0*np.eye(3, dtype=complex))[None, :, :],
    }
    with pytest.raises(ValueError, match=message):
        landau_full_em_electric_field_fluctuation_spectrum(response, 1.0e-3)


def test_end_to_end_landau_fdt_full_em_spectrum_chains_operator_noise_and_maxwell_green():
    inputs = _inputs()
    result = run_advanced_thomson_physics(
        "landau_fdt_full_em_spectrum", **inputs,
        propagation_direction=(1.0, 2.0, 0.5),
    )
    assert result["public_physics_entrypoint"] == "landau_fdt_full_em_spectrum"
    assert result["automatic_model_substitution"] is False
    assert result["integration_status"] == (
        "conserving_Landau_operator_and_matching_FDT_noise_connected_"
        "through_frequency_resolved_Maxwell_Green_dyadic"
    )
    assert result["cts_three_wave_vertex_connected"] is False
    assert result["constitutive_response"]["conductivity_tensor_S_m"].shape == (2, 3, 3)
    for key in (
        "electric_field_fluctuation_spectral_matrix_V2_m2_per_rad_s",
        "magnetic_field_fluctuation_spectral_matrix_T2_per_rad_s",
        "electric_magnetic_cross_spectral_matrix",
    ):
        tensor = np.asarray(result[key])
        assert tensor.shape == (2, 3, 3)
        assert np.all(np.isfinite(tensor))
        for matrix in tensor:
            np.testing.assert_allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-35)
            assert np.linalg.eigvalsh(matrix).min() >= -1e-9 * max(
                np.linalg.norm(matrix, ord=2), np.finfo(float).tiny
            )
    assert result["transverse_stokes_spectral_density_IQUV"].shape == (2, 4)
    assert result["spectral_poynting_flux_W_m2_per_rad_s"].shape == (2, 3)
    assert result["quantitative_release_gate"]["magnetized_response_required_for_magnetized_plasma"] is True


def test_end_to_end_landau_fdt_full_em_spectrum_rejects_undamped_green_pole():
    inputs = _inputs()
    inputs["angular_frequency_rad_s"] = np.array([1.0e6])
    inputs["wave_number_m_inverse"] = 1.0e6 / 299792458.0
    with pytest.raises(np.linalg.LinAlgError, match="ill-conditioned"):
        run_advanced_thomson_physics(
            "landau_fdt_full_em_spectrum", **inputs,
            propagation_direction=(1.0, 0.0, 0.0),
        )
