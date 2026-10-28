import numpy as np
import pytest
from tokamak_analysis.poloidal_flux import reconstruct_poloidal_flux, analyze_poloidal_flux


def test_uniform_vertical_field_flux_and_reference_gauge():
    R = np.array([1.0, 1.5, 2.0])
    Z = np.array([-1.0, 0.0, 1.0])
    br = np.zeros((3, 3))
    bz = np.full((3, 3), 2.0)
    out = reconstruct_poloidal_flux(R, Z, br, bz, initial_flux_Wb=7.0)
    expected = 7.0 + (R**2 - R[0]**2)
    np.testing.assert_allclose(out["poloidal_flux_Wb"], np.tile(expected, (3, 1)))
    assert out["poloidal_flux_Wb"][0, 0] == pytest.approx(7.0)
    assert out["diagnostics"]["sample_count"] == 9


def test_radial_field_gives_vertical_flux_gradient():
    R = np.array([1., 2., 3.])
    Z = np.array([0., 1., 2.])
    br = np.full((3, 3), -2.0) / R[None, :]
    bz = np.zeros((3, 3))
    out = analyze_poloidal_flux(R, Z, br, bz)
    np.testing.assert_allclose(out["poloidal_flux_Wb"][:, 0], [0., 2., 4.])


@pytest.mark.parametrize("R,Z,br,bz", [
    ([1], [0,1], np.zeros((2,1)), np.zeros((2,1))),
    ([1,1], [0,1], np.zeros((2,2)), np.zeros((2,2))),
    ([0,1], [0,1], np.zeros((2,2)), np.zeros((2,2))),
    ([1,2], [0,1], np.zeros((3,2)), np.zeros((2,2))),
    ([1,2], [0,1], np.zeros((2,2)), np.full((2,2), np.nan)),
])
def test_invalid_inputs_rejected(R,Z,br,bz):
    with pytest.raises(ValueError):
        reconstruct_poloidal_flux(R,Z,br,bz)


def test_reference_index_and_shape_validation():
    R=np.array([1.,2.]); Z=np.array([0.,1.])
    out=reconstruct_poloidal_flux(R,Z,np.zeros((2,2)),np.ones((2,2)),
                                  initial_flux_Wb=3.,reference_index=(1,1))
    assert out["poloidal_flux_Wb"][1,1] == pytest.approx(3.)
    with pytest.raises(ValueError, match="reference_index"):
        reconstruct_poloidal_flux(R,Z,np.zeros((2,2)),np.ones((2,2)),reference_index=(2,0))
    with pytest.raises(ValueError, match="shape"):
        reconstruct_poloidal_flux(R,Z,np.zeros((3,2)),np.ones((2,2)))



def test_integrability_diagnostic_for_analytic_flux_field():
    # Analytic psi=R^2 Z gives B_Z=(1/R)dpsi/dR=2Z and
    # B_R=-(1/R)dpsi/dZ=-R. The derivative constraints are compatible.
    R = np.linspace(1.0, 2.0, 7)
    Z = np.linspace(-1.0, 1.0, 9)
    br = -np.broadcast_to(R[None, :], (Z.size, R.size))
    bz = np.broadcast_to((2.0 * Z)[:, None], (Z.size, R.size))
    out = reconstruct_poloidal_flux(R, Z, br, bz)
    residual = out["integrability_residual_Wb_m2"]
    assert residual.shape == (Z.size, R.size)
    assert np.isfinite(residual).all()
    assert out["diagnostics"]["rms_integrability_residual_Wb_m2"] < 1e-12


def test_inconsistent_field_is_flagged_by_integrability_diagnostic():
    R = np.linspace(1.0, 2.0, 7)
    Z = np.linspace(0.0, 1.0, 9)
    br = np.zeros((Z.size, R.size))
    bz = np.broadcast_to(Z[:, None], (Z.size, R.size))
    out = reconstruct_poloidal_flux(R, Z, br, bz)
    assert out["diagnostics"]["rms_integrability_residual_Wb_m2"] > 0.1
    assert "integrability residuals" in " ".join(out["assumptions"])


@pytest.mark.parametrize("bad_flux", [True, "1.0", None, np.nan, np.inf])
def test_invalid_initial_flux_gauge_is_rejected(bad_flux):
    R = np.array([1.0, 2.0])
    Z = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="initial_flux_Wb"):
        reconstruct_poloidal_flux(R, Z, np.zeros((2, 2)), np.zeros((2, 2)),
                                  initial_flux_Wb=bad_flux)
