# Charge-exchange recombination spectroscopy (CXRS) — analysis 4/80

## Implemented scope

The built-in `tokamak_analysis.cxrs.analyze_cxrs_line` accepts a calibrated wavelength axis and a single isolated charge-exchange emission line, fits a Gaussian line plus linear local background, subtracts a known Gaussian instrumental width in quadrature, and reports ion temperature and line-of-sight velocity with local fit-covariance uncertainties. It returns the fitted signal, residuals, R², and (when pointwise signal uncertainties are supplied) reduced chi-square.

## Relations

For a non-relativistic Doppler shift, the reported line-of-sight velocity is
`v_los = c (lambda_c - lambda_0) / lambda_0`.

For a Maxwellian ion distribution and Gaussian broadening,
`sigma_D^2 = sigma_obs^2 - sigma_inst^2` and
`T_i[eV] = (m_i c^2/e) (sigma_D/lambda_0)^2`.

All wavelengths and widths use nm consistently in the ratios. The ion mass is supplied in atomic mass units and converted to kg. The code rejects unresolved widths where observed sigma does not exceed the instrumental sigma.

## Inputs

- `wavelength_nm`: finite, strictly increasing calibrated wavelength axis (nm), at least 7 samples.
- `signal`: matching finite spectral signal.
- `rest_wavelength_nm`: positive rest wavelength for the identified transition (nm).
- `ion_mass_amu`: positive emitting-ion mass in u.
- `instrument_sigma_nm`: independently measured Gaussian instrumental standard deviation (nm), not FWHM.
- `viewing_angle_rad`: optional metadata in [0, pi]; no geometric deprojection is performed.
- `signal_uncertainty`: optional positive 1-sigma uncertainty per sample.
- `initial_center_nm`: optional line-center initial estimate.

## Outputs

Line center and uncertainty, observed/intrinsic width, Ti and uncertainty, line-of-sight velocity and uncertainty, amplitude, local linear-background coefficients, R², optional reduced chi-square, degrees of freedom, fitted samples, residuals, assumptions, and limitations.

## Scientific limits

This is a validated numerical line-fit workflow, not a complete device-specific CXRS inversion. It does not calculate impurity density from atomic rates, model neutral-beam attenuation or beam overlap, deblend multiple transitions, correct opacity, fit non-Gaussian instrumental line shapes, or infer toroidal/poloidal rotation from multiple views. Fit covariance excludes systematic uncertainties in wavelength calibration, rest wavelength, ion mass and instrumental width. The viewing angle is recorded only; the returned velocity remains line-of-sight. Independent experimental validation and device-specific atomic-data checks remain necessary.

## Reproducible tests

`tests/test_cxrs.py` covers synthetic recovery of center/velocity/Ti, weighted fit diagnostics, invalid axes and uncertainties, nonphysical parameters, and unresolved instrumental broadening.
