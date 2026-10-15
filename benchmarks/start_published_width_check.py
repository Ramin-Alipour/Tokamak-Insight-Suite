#!/usr/bin/env python3
"""Approximate consistency check from the START tokamak spectroscopy paper.

Carolan et al., Plasma Physics Reports 24 (1998), 206-213.
https://www.researchgate.net/publication/252311585_A_doppler_spectroscopy_diagnostic_to_study_the_ion_temperature_rotation_and_confinement_in_Neutral-Beam-heated_START_plasmas

The article describes the C5+ line near 529.05 nm and states that a typical
Gaussian FWHM is about 0.2 nm for Ti=300 eV. This script reverses that relation
to check the scale. Because the published width is explicitly approximate and
is not an extracted raw data point, this is a consistency check, not raw-data
validation.
"""
from math import log

WAVELENGTH_NM = 529.05
ION_MASS_AMU = 12.0
APPROX_FWHM_NM = 0.2
PUBLISHED_APPROX_TI_EV = 300.0


def doppler_ti_ev(fwhm_nm: float) -> float:
    amu_kg = 1.66053906660e-27
    c_m_s = 299792458.0
    elementary_charge_j_per_ev = 1.602176634e-19
    return (
        ION_MASS_AMU * amu_kg * c_m_s**2
        / (8 * log(2) * elementary_charge_j_per_ev)
        * (fwhm_nm / WAVELENGTH_NM) ** 2
    )


if __name__ == "__main__":
    ti = doppler_ti_ev(APPROX_FWHM_NM)
    delta = 100 * (ti / PUBLISHED_APPROX_TI_EV - 1)
    print("START tokamak C5+ line approximate-width consistency check")
    print(f"Line wavelength: {WAVELENGTH_NM:.2f} nm")
    print(f"Approximate published FWHM: {APPROX_FWHM_NM:.3f} nm")
    print(f"Recalculated Ti: {ti:.1f} eV")
    print(f"Difference from approximate 300 eV: {delta:+.1f}%")
    print("Classification: approximate published-width check, not raw-spectrum refit.")
