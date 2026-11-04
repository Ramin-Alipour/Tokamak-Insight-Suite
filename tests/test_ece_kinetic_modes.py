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
