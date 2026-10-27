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
    ([1,2], [0,1], np.zeros((2,2)), np.zeros((2,2))),
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
