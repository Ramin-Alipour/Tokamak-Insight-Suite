# Ion temperature from Doppler spectroscopy

## Scope

This workflow estimates ion temperature from an isolated, wavelength-calibrated emission line. It is an end-to-end numerical path from a sampled spectrum to a temperature estimate, not an automatic determination that the measured width is purely thermal.

## Inputs

- `wavelength_nm`: finite, strictly increasing calibrated wavelength samples (nm).
- `intensity`: finite intensity samples aligned with the wavelength axis.
- `rest_wavelength_nm`: laboratory wavelength (nm) for the identified transition.
- `ion_mass_amu`: mass of the emitting ion/isotope (amu).
- `instrumental_fwhm_nm`: measured Gaussian instrumental FWHM (nm) for the actual wavelength and instrument configuration.
- Optional `intensity_uncertainty`: positive 1-sigma uncertainty for each intensity sample.
- Optional `instrumental_fwhm_uncertainty_nm`: 1-sigma uncertainty of the instrumental FWHM.

## Method

1. Fit the spectrum with a Gaussian line and local linear background.
2. Convert the fitted Gaussian sigma to observed FWHM: `FWHM_obs = 2 sqrt(2 ln 2) sigma`.
3. Correct compatible Gaussian widths in quadrature: `FWHM_D = sqrt(FWHM_obs^2 - FWHM_inst^2)`. The workflow rejects unresolved lines.
4. Infer temperature from `Ti[eV] = (mi c^2/e) (FWHM_D/lambda0)^2 / (8 ln 2)`.
5. Return fit diagnostics, fitted signal, residuals, covariance, assumptions and (when both width uncertainties are available) first-order propagated temperature uncertainty.

Callable entry point: `tokamak_analysis.ion_temperature.ion_temperature_from_doppler_spectrum`.

## Interpretation limits

- The rest transition and ion species must be independently verified; the rest wavelength must lie inside the measured wavelength window, and the emission peak/fit center must not be truncated at its boundary.
- The workflow rejects flat/no-contrast spectra, failed or non-physical Gaussian fits, and fits whose width is too large for the selected wavelength window.
- Blends, unresolved fine structure, Stark/Zeeman broadening, opacity, rotation/flow distributions and non-thermal populations can bias the result.
- `fit_r_squared` is a diagnostic, not proof of physical validity.
- The reported uncertainty is local statistical propagation. It does not automatically include wavelength-calibration, instrument-model, atomic-data or physical-model systematic uncertainty.
- Use a Voigt or line-specific model when the physical broadening mechanisms require it; do not subtract non-Gaussian widths in quadrature without justification.

## Tests

`tests/test_ion_temperature_spectrum.py` covers synthetic recovery, line-fit diagnostics, malformed wavelength inputs, unresolved instrumental widths, uncertainty-array validation, out-of-window rest wavelengths, boundary-truncated peaks and flat spectra.
