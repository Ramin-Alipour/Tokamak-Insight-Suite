import numpy as np
import pytest

from tokamak_analysis.neutron_flux import analyze_neutron_flux, neutron_flux_from_integrated_counts
from tokamak_analysis.wall_recycling import net_wall_particle_sink
from tokamak_analysis.transport import radial_diffusive_particle_flux, poloidal_diffusive_particle_flux
from tokamak_analysis.divertor_analysis import analyze_divertor_particle_flux


def test_neutron_flux_uses_one_canonical_key_and_keeps_legacy_alias():
    result = analyze_neutron_flux([0.0, 1.0, 2.0], [10.0, 20.0, 30.0],
                                  efficiency=0.5, effective_area_m2=2.0)
    np.testing.assert_allclose(result["neutron_flux_m-2_s-1"], result["flux_estimate_m-2_s-1"])
    integrated = neutron_flux_from_integrated_counts(100, 10, 0.5, 2.0)
    np.testing.assert_allclose(integrated["neutron_flux_m-2_s-1"], integrated["flux_estimate_m-2_s-1"])


def test_wall_sink_exposes_canonical_and_legacy_keys():
    result = net_wall_particle_sink([5.0, 8.0], [2.0, 3.0])
    np.testing.assert_allclose(result["net_particle_sink_flux_m-2_s-1"], [3.0, 5.0])
    np.testing.assert_allclose(result["net_wall_sink_flux_m-2_s-1"], result["net_particle_sink_flux_m-2_s-1"])


@pytest.mark.parametrize("function,key", [
    (radial_diffusive_particle_flux, "radial_particle_flux_m-2_s-1"),
    (poloidal_diffusive_particle_flux, "poloidal_particle_flux_m-2_s-1"),
])
def test_transport_flux_uses_canonical_particle_flux_alias(function, key):
    result = function(2.0, 3.0)
    assert result["particle_flux_m-2_s-1"] == pytest.approx(-6.0)
    assert result[key] == pytest.approx(result["particle_flux_m-2_s-1"])


def test_divertor_integrated_flux_contract_has_explicit_profile_units():
    result = analyze_divertor_particle_flux([0.0, 1.0, 2.0], [1.0, 2.0, 1.0])
    assert result["integrated_particle_flux_m-1_s-1"] == pytest.approx(3.0)
    assert result["particle_flux_m-2_s-1"] == [1.0, 2.0, 1.0]
