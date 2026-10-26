"""Acceptance tests for measured-data inverse magnetic equilibrium fitting."""
import numpy as np
import pytest

from tokamak_analysis.equilibrium_reconstruction import reconstruct_equilibrium_inverse


def machine_case():
    R = np.linspace(1.0, 2.0, 9)
    Z = np.linspace(-0.5, 0.5, 9)
    rr, zz = np.meshgrid(R, Z)
    basis = np.stack((-(rr - 1.5) ** 2 - zz**2,
                      0.2 * (rr - 1.5) ** 2 + 0.1 * zz**2))
    coeff = np.array([2.0, 0.5])
    probe_response = np.array([[1.0, 0.2], [0.1, 1.2], [0.7, -0.4]])
    probe_coil_response = np.array([[0.01], [0.02], [-0.01]])
    probe_currents = np.array([10.0])
    probe_sigma = np.array([0.01, 0.02, 0.01])
    probe_values = probe_response @ coeff + probe_coil_response @ probe_currents
    loop_response = np.array([[0.4, 0.3], [-0.2, 0.8]])
    loop_coil_response = np.array([[0.002], [0.004]])
    loop_currents = np.array([10.0])
    loop_sigma = np.array([0.001, 0.002])
    loop_values = loop_response @ coeff + loop_coil_response @ loop_currents
    return dict(
        R_m=R, Z_m=Z, basis_psi_Wb_rad=basis,
        magnetic_probe_response_T_per_coefficient=probe_response,
        magnetic_probe_values_T=probe_values,
        magnetic_probe_sigma_T=probe_sigma,
        coil_probe_response_T_per_A=probe_coil_response,
        coil_probe_currents_A=probe_currents,
        flux_loop_response_Wb_rad_per_coefficient=loop_response,
        flux_loop_values_Wb_rad=loop_values,
        flux_loop_sigma_Wb_rad=loop_sigma,
        coil_flux_response_Wb_rad_per_A=loop_coil_response,
        coil_flux_currents_A=loop_currents,
    )


def test_inverse_fit_recovers_coefficients_axis_and_edge_flux_from_measurements():
    case = machine_case()
    result = reconstruct_equilibrium_inverse(**case, regularization=0.0)
    np.testing.assert_allclose(result["basis_coefficients"], [2.0, 0.5], atol=1e-10)
    assert result["measurement_count"] == 5
    assert result["weighted_residual_rms"] < 1e-9
    assert result["axis_R_m"] == pytest.approx(1.5)
    assert result["axis_Z_m"] == pytest.approx(0.0)
    assert np.isfinite(result["boundary_flux_Wb_rad"])
    assert result["boundary_mask"].shape == (case["Z_m"].size, case["R_m"].size)
    assert np.all(np.isfinite(result["psi_normalized"]))
    assert set(result["measurement_diagnostics"]) == {"magnetic probes", "flux loops"}


def test_inverse_fit_can_use_probes_or_flux_loops_alone():
    case = machine_case()
    probe_keys = [k for k in case if k.startswith("flux_loop") or k.startswith("coil_flux")]
    probes_only = {k: v for k, v in case.items() if k not in probe_keys}
    result = reconstruct_equilibrium_inverse(**probes_only, regularization=0.0)
    np.testing.assert_allclose(result["basis_coefficients"], [2.0, 0.5], atol=1e-10)
    assert result["measurement_count"] == 3


def test_inverse_fit_requires_actual_measurements_and_uncertainties():
    case = machine_case()
    case.pop("magnetic_probe_values_T")
    with pytest.raises(ValueError, match="all required"):
        reconstruct_equilibrium_inverse(**case)


def test_inverse_fit_rejects_unobservable_basis_and_bad_uncertainty():
    case = machine_case()
    case["magnetic_probe_response_T_per_coefficient"][:, 1] = 0.0
    case["flux_loop_response_Wb_rad_per_coefficient"][:, 1] = 0.0
    with pytest.raises(ValueError, match="unobservable"):
        reconstruct_equilibrium_inverse(**case)
    case = machine_case()
    case["magnetic_probe_sigma_T"][0] = 0.0
    with pytest.raises(ValueError, match="uncertainties positive"):
        reconstruct_equilibrium_inverse(**case)


def test_inverse_fit_requires_paired_coil_response_and_currents():
    case = machine_case()
    case.pop("coil_probe_currents_A")
    with pytest.raises(ValueError, match="must be supplied together"):
        reconstruct_equilibrium_inverse(**case)
