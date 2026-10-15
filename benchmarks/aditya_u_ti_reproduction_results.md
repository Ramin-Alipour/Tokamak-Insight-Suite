# Aditya-U published-width ion-temperature reproduction

**Status:** arithmetic reproduction complete; raw-spectrum refit not yet possible from the article's publicly verified files.

## Source and case

Shukla et al., “Poloidal Rotation and Edge Ion Temperature Measurements Using Spectroscopy Diagnostic on Aditya-U Tokamak,” *Atoms* 7 (2019), 93. DOI: https://doi.org/10.3390/atoms7030093

- Tokamak shot: 33057
- Species/line: C²⁺, 464.74 nm
- Chord/radius: edge, r = 21.55 cm
- Reported line width: 0.0652 ± 0.0006 nm
- Reported instrumental FWHM: 0.039 nm
- Published ion temperature: 40.1 eV (paper prints ±0.003 eV)
- Spectra shown for 90–110 ms with 20 ms CCD exposure; the paper's temperature time trace also describes a measurement around 55 ms. Do not conflate these timing descriptions without checking the figure-specific caption/data.

## Independent calculation

For a Gaussian Doppler FWHM, assuming the quoted width is the Doppler-only width:

`Ti_eV = [m_i * c^2 / (8 * ln(2) * e)] * (FWHM_D / lambda_0)^2`

Using carbon mass 12 u, `lambda0 = 464.74 nm`, and `Delta_lambda_D = 0.0652 nm`:

- Recalculated `Ti = 39.675 eV`
- Propagating the stated width uncertainty only, `sigma_T = 2 T sigma_width / width = 0.730 eV`
- Difference from published 40.1 eV: -0.425 eV (-1.06%).

**This reproduces the published central value closely.** It does not reproduce the paper's printed ±0.003 eV uncertainty: the stated ±0.0006 nm width uncertainty alone propagates to approximately ±0.73 eV. That uncertainty discrepancy needs clarification and must not be hidden.

## Instrument-width ambiguity check

The same paper states `Delta_lambda_act² = Delta_lambda_meas² - Delta_lambda_inst²`. If 0.0652 nm is the *measured* width before instrumental correction, then:

- `Delta_lambda_D = sqrt(0.0652² - 0.039²) = 0.0522498 nm`
- Recalculated `Ti = 25.480 eV`
- Width-only propagated uncertainty: approximately ±0.730 eV
- Difference from published 40.1 eV: -14.62 eV (-36.46%).

Thus, the article's reported central value is numerically consistent with using 0.0652 nm directly as the Doppler width. Applying the stated instrumental correction again changes the answer materially. The text/figure must be clarified to determine whether 0.0652 nm is already corrected.

## Scientific classification

- **PASS — published-number arithmetic reproduction:** 39.675 eV from the width gives the reported ~40.1 eV within 1.1%.
- **NOT a raw-spectrum fit:** a machine-readable original CCD count array was not located in the open article/supplement search. The plotted spectra are not being mislabelled as raw arrays.
- **NOT full empirical validation:** this calculation starts from a width already reported by the paper; it cannot independently validate background subtraction, wavelength calibration, line fitting, or instrument deconvolution.

Code and regression checks:
- `benchmarks/aditya_u_ti_reproduction.py`
- `tests/test_aditya_u_ti_reproduction.py`
