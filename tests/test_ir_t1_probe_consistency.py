"""Illustrative IR-T1 compound-probe arithmetic; not a closure test of plotted means.

Reference: Alipour et al., Rev. Sci. Instrum. 88, 093516 (2017),
DOI 10.1063/1.4994037. The values below are provisional digitizations of
Figs. 10(a) and 11(a), not synchronized raw samples.

Important: comparing mean Mach times a sound speed computed from a mean
temperature with a separately digitized mean velocity is not a valid test of
pointwise consistency. In general, mean(M * cs(Te, Ti)) is not equal to
mean(M) * cs(mean(Te), mean(Ti)); the result depends on fluctuations,
covariances, averaging intervals, ion-temperature assumptions, and the
paper's exact definition of sound speed. The calculations here are illustrative
only and cannot establish a discrepancy in the paper or the diagnostic.
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


def test_probe_figure_averages_are_only_illustrative_arithmetic():
    mach_mean = 0.511
    velocity_mean_km_s = 17.57
    electron_temperature_eV = 14.0

    # Case A: electron-temperature-only estimate, not asserted to be the
    # paper's exact model or a physically valid closure test on means.
    cs_te_only = sound_speed_km_s(electron_temperature_eV)
    velocity_from_mean_mach_te_only = mach_mean * cs_te_only

    # Case B: explicitly conditional Ti/Te = 0.2 assumption.
    cs_te_plus_ti = sound_speed_km_s(electron_temperature_eV * 1.2)
    velocity_from_mean_mach_te_plus_ti = mach_mean * cs_te_plus_ti

    mach_implied_by_mean_velocity_te_only = velocity_mean_km_s / cs_te_only
    mach_implied_by_mean_velocity_te_plus_ti = velocity_mean_km_s / cs_te_plus_ti

    assert cs_te_only == pytest.approx(36.6197, abs=0.001)
    assert cs_te_plus_ti == pytest.approx(40.1149, abs=0.001)
    assert velocity_from_mean_mach_te_only == pytest.approx(18.7127, abs=0.001)
    assert velocity_from_mean_mach_te_plus_ti == pytest.approx(20.4987, abs=0.001)
    assert mach_implied_by_mean_velocity_te_only == pytest.approx(0.4798, abs=0.001)
    assert mach_implied_by_mean_velocity_te_plus_ti == pytest.approx(0.4380, abs=0.001)

    # These assertions protect the arithmetic only. Differences between these
    # plug-in estimates and separately digitized averages are not evidence of
    # inconsistency unless the same samples, intervals, and sound-speed model
    # are applied pointwise before averaging.
