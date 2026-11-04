import numpy as np
import pytest

from tokamak_analysis.cts_full_em import spitzer_cts_collision_frequencies


def _plasma(ne=1.0e19, te=1000.0):
    ions = [{"name": "D+", "charge_state": 1, "mass_amu": 2.014}]
    return spitzer_cts_collision_frequencies(
        ne, te, ions, [1.0], 500.0, coulomb_logarithm=15.0
    )


def test_spitzer_cts_collision_frequencies_match_electron_ion_reference():
    result = _plasma()
    expected = 2.91e-6 * (1.0e19 * 1.0e-6) * 15.0 / 1000.0**1.5
    assert result["collision_frequencies_s_inverse"]["e-"] == pytest.approx(expected)
    assert result["collision_frequencies_s_inverse"]["D+"] > 0.0
    assert result["effective_charge_Zeff"] == pytest.approx(1.0)
    assert result["collision_operator_conserves_momentum_and_energy"] is False


def test_spitzer_cts_collision_frequency_scaling():
    base = _plasma()
    doubled_density = _plasma(ne=2.0e19)
    hotter = _plasma(te=2000.0)
    assert doubled_density["collision_frequencies_s_inverse"]["e-"] == pytest.approx(
        2.0 * base["collision_frequencies_s_inverse"]["e-"]
    )
    assert hotter["collision_frequencies_s_inverse"]["e-"] == pytest.approx(
        base["collision_frequencies_s_inverse"]["e-"] / 2.0**1.5
    )


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf"), True])
def test_spitzer_cts_collision_frequency_rejects_invalid_coulomb_logarithm(bad):
    ions = [{"name": "D+", "charge_state": 1, "mass_amu": 2.014}]
    with pytest.raises(ValueError):
        spitzer_cts_collision_frequencies(
            1.0e19, 1000.0, ions, [1.0], 500.0,
            coulomb_logarithm=bad,
        )

def test_species_conductivity_uses_physical_collision_rate_separately_from_numerical_width():
    from tokamak_analysis import cts_full_em

    species = {
        "mass": cts_full_em.ME,
        "charge": -cts_full_em.E_CHARGE,
        "density": 1.0e19,
        "temperature": 1000.0,
        "flow": np.zeros(3),
    }
    common = dict(
        omega=2.0 * np.pi * 1.0e9,
        k_vector=np.array([12.0, 0.0, 20.0]),
        magnetic_field_T=np.array([0.0, 0.0, 2.0]),
        species=species,
        harmonic_order=2,
        quadrature_order=4,
        damping_rate_rad_s=1.0,
    )
    collisionless = cts_full_em._species_conductivity(
        **common, collision_frequency_rad_s=0.0
    )
    collisional = cts_full_em._species_conductivity(
        **common, collision_frequency_rad_s=2.0e5
    )
    assert np.all(np.isfinite(collisional))
    assert not np.allclose(collisional, collisionless)

