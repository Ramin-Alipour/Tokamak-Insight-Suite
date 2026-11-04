import numpy as np
from scipy.constants import Boltzmann, elementary_charge
from tokamak_analysis.ece_kinetic_modes import integrate_mode_coupled_kinetic_ece_transfer


def test_polarized_transfer_reduces_to_scalar_lte_slab():
    alpha, length, te = 1e-4, 2.0, 150.0
    out = integrate_mode_coupled_kinetic_ece_transfer(
        [80e9, 100e9], [length], np.array([alpha*np.eye(2)]), [te]
    )
    expected = te*elementary_charge/Boltzmann*(1-np.exp(-alpha*length))
    np.testing.assert_allclose(out['brightness_temperature_out_K'], expected, rtol=1e-12)
    np.testing.assert_allclose(out['modal_coherency_out_K'], expected*np.eye(2)[None], rtol=1e-12)


def test_mode_coupling_rotates_polarization_without_absorption():
    import pytest
    from tokamak_analysis.ece_kinetic_modes import integrate_mode_coupled_kinetic_ece_transfer
    initial = np.array([[1.0, 0.0], [0.0, 0.0]], complex)
    coupling = np.array([[0.0, 2.0], [2.0, 0.0]], complex)
    out = integrate_mode_coupled_kinetic_ece_transfer(
        90e9, [0.4], np.zeros((1, 2, 2), complex), [100.0],
        mode_coupling_matrix_rad_m=coupling, incident_coherency_K=initial,
    )
    result = out["modal_coherency_out_K"][0]
    assert np.trace(result).real == pytest.approx(1.0, abs=1e-12)
    assert abs(result[0, 1]) > 0.1


def test_transfer_rejects_nonpositive_absorption_operator():
    import pytest
    from tokamak_analysis.ece_kinetic_modes import integrate_mode_coupled_kinetic_ece_transfer
    with pytest.raises(ValueError, match="positive semidefinite"):
        integrate_mode_coupled_kinetic_ece_transfer(
            90e9, [1.0], np.array([[[-1.0, 0], [0, 0]]], complex), [100.0]
        )
