import pytest

from tokamak_analysis.cts_full_em import full_em_cts_spectral_density


def test_full_em_cts_rejects_exact_zero_shift_until_static_maxwell_limit_exists():
    with pytest.raises(ValueError, match="static-Maxwell limit"):
        full_em_cts_spectral_density(
            [0.0], 140e9, {},
            electron_density_m3=2e19,
            electron_temperature_eV=1200.0,
            ion_temperature_eV=800.0,
            ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
            ion_fractions=[1.0],
            magnetic_field_T=[0.0, 0.0, 2.0],
        )
