"""Expose the unresolved IR-T1 compound-probe Mach/velocity consistency check.

Reference: Alipour et al., Rev. Sci. Instrum. 88, 093516 (2017),
DOI 10.1063/1.4994037; provisional full-discharge digitization of Figs. 10(a)
and 11(a). This test validates transparent arithmetic for stated assumptions;
it does not decide which plotted quantity or physical model is wrong.
"""
import math

import pytest


ELEMENTARY_CHARGE_C = 1.602176634e-19
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
HYDROGEN_ION_MASS_AMU = 1.0073


def sound_speed_km_s(temperature_sum_eV: float) -> float:
    """Ion-acoustic speed sqrt(e*(Te+Ti)/mi), for the stated simple model."""
    return math.sqrt(
        ELEMENTARY_CHARGE_C * temperature_sum_eV
        / (HYDROGEN_ION_MASS_AMU * ATOMIC_MASS_UNIT_KG)
    ) / 1000.0


def test_probe_reference_values_remain_an_explicit_unresolved_consistency_flag():
    mach_mean = 0.511
    velocity_mean_km_s = 17.57
    electron_temperature_eV = 14.0

    # Case A: the electron-temperature-only estimate sometimes used as a
    # simplified comparison; it is not assumed to be the paper's exact model.
    cs_te_only = sound_speed_km_s(electron_temperature_eV)
    velocity_from_mach_te_only = mach_mean * cs_te_only

    # Case B: explicitly conditional Ti/Te = 0.2 assumption.
    cs_te_plus_ti = sound_speed_km_s(electron_temperature_eV * 1.2)
    velocity_from_mach_te_plus_ti = mach_mean * cs_te_plus_ti

    mach_implied_by_figure_velocity_te_only = velocity_mean_km_s / cs_te_only
    mach_implied_by_figure_velocity_te_plus_ti = velocity_mean_km_s / cs_te_plus_ti

    assert cs_te_only == pytest.approx(36.6197, abs=0.001)
    assert cs_te_plus_ti == pytest.approx(40.1149, abs=0.001)
    assert velocity_from_mach_te_only == pytest.approx(18.7127, abs=0.001)
    assert velocity_from_mach_te_plus_ti == pytest.approx(20.4987, abs=0.001)
    assert mach_implied_by_figure_velocity_te_only == pytest.approx(0.4798, abs=0.001)
    assert mach_implied_by_figure_velocity_te_plus_ti == pytest.approx(0.4380, abs=0.001)

    # The figure-derived averages do not close under either illustrative model.
    # Keep this as a visible investigation flag, not a test that the paper is wrong.
    assert abs(velocity_from_mach_te_only - velocity_mean_km_s) / velocity_mean_km_s > 0.06
    assert abs(velocity_from_mach_te_plus_ti - velocity_mean_km_s) / velocity_mean_km_s > 0.16
