#!/usr/bin/env python3
"""Reproduce the published Aditya-U C2+ Doppler-temperature arithmetic.

Reference:
G. Shukla et al., Atoms 7 (2019) 93, DOI: 10.3390/atoms7030093.
https://www.mdpi.com/2218-2004/7/3/93

This script reproduces the calculation from the width printed in the paper.
It does NOT refit raw CCD counts: a machine-readable raw spectrum was not
located in the article/supplement search. Both plausible width conventions
are reported because the text gives an instrumental FWHM and a width of
0.0652 nm, but the quoted 40.1 eV matches using 0.0652 nm directly, before
subtracting the instrumental width in quadrature.
"""
from math import log, sqrt

LAMBDA_NM = 464.74
ION_MASS_AMU = 12.0  # C2+ ion mass, rounded to 12 u
WIDTH_NM = 0.0652
WIDTH_SIGMA_NM = 0.0006
INSTRUMENT_FWHM_NM = 0.039
PUBLISHED_TI_EV = 40.1


def doppler_ti_ev(fwhm_nm: float) -> float:
    """Return Ti in eV for a Gaussian Doppler FWHM.

    Ti = m*c^2/(8 ln(2) e) * (FWHM/lambda)^2
    """
    amu_kg = 1.66053906660e-27
    c_m_s = 299792458.0
    elementary_charge_j_per_ev = 1.602176634e-19
    return (
        ION_MASS_AMU * amu_kg * c_m_s**2
        / (8.0 * log(2.0) * elementary_charge_j_per_ev)
        * (fwhm_nm / LAMBDA_NM) ** 2
    )


def main() -> None:
    # Case A: width is already Doppler-only/corrected.
    ti_corrected_width = doppler_ti_ev(WIDTH_NM)
    sigma_ti_a = 2 * ti_corrected_width * WIDTH_SIGMA_NM / WIDTH_NM

    # Case B: width is measured FWHM; remove the instrumental FWHM in quadrature.
    doppler_width = sqrt(WIDTH_NM**2 - INSTRUMENT_FWHM_NM**2)
    ti_after_instrument_correction = doppler_ti_ev(doppler_width)
    sigma_doppler_width = WIDTH_NM / doppler_width * WIDTH_SIGMA_NM
    sigma_ti_b = (
        2 * ti_after_instrument_correction * sigma_doppler_width / doppler_width
    )

    print("Aditya-U / shot 33057 / C2+ line 464.74 nm")
    print(f"Published width: {WIDTH_NM:.4f} ± {WIDTH_SIGMA_NM:.4f} nm")
    print(f"Instrumental FWHM: {INSTRUMENT_FWHM_NM:.3f} nm")
    print(f"Paper's reported Ti: {PUBLISHED_TI_EV:.1f} eV")
    print()
    print("A. If 0.0652 nm is already Doppler-only:")
    print(f"   Ti = {ti_corrected_width:.2f} ± {sigma_ti_a:.2f} eV")
    print(f"   Relative difference from paper = "
          f"{100 * (ti_corrected_width / PUBLISHED_TI_EV - 1):+.2f}%")
    print()
    print("B. If 0.0652 nm is measured width and instrumental broadening is removed:")
    print(f"   Corrected Doppler FWHM = {doppler_width:.5f} nm")
    print(f"   Ti = {ti_after_instrument_correction:.2f} ± {sigma_ti_b:.2f} eV")
    print(f"   Relative difference from paper = "
          f"{100 * (ti_after_instrument_correction / PUBLISHED_TI_EV - 1):+.2f}%")
    print()
    print("Interpretation: case A reproduces the published ~40.1 eV (within ~1.1%).")
    print("Case B gives ~25.5 eV. The paper's width convention must be clarified")
    print("before claiming an independent raw-spectrum validation.")
    print("Classification: published-width arithmetic reproduction, NOT raw-data refit.")


if __name__ == "__main__":
    main()
