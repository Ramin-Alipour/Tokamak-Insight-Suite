# Thomson Spectral Distribution Analysis (Analysis 7 of 80)

## Scope
Distribution-level statistics for a calibrated Thomson-scattering spectrum: positive net spectral area, centroid, RMS width, skewness, excess kurtosis, negative-bin count, and optional first-order uncertainty for area and centroid. Complements, but does not replace, the Gaussian model fit in the Thomson spectrum module.

## Inputs
- wavelength_nm: finite, strictly increasing calibrated wavelength axis (nm), at least five samples.
- signal: finite one-dimensional signal in detector units or calibrated spectral power density.
- background: optional finite scalar or sample-matched background.
- signal_uncertainty: optional positive finite independent 1-sigma uncertainty per sample.

## Method
Subtract background while preserving signed net samples. Negative bins are retained in net_signal but excluded from positive-weight moments. Use trapezoidal wavelength integration weights to calculate area, centroid and central moments. Optional first-order uncertainty propagation assumes independent sample errors and fixed positive-bin support.

## Scientific limitations
- Moments are descriptive and do not establish a Maxwellian electron distribution.
- No instrument-response deconvolution, stray-light correction, relativistic/collective model, wavelength-dependent response correction, or density inference.
- Background uncertainty, calibration covariance, window truncation and uncertainty in positive-bin selection are excluded.
- Results require comparison with instrument calibration and a validated physical model.

## Example
```python
from tokamak_analysis.thomson_spectral_distribution import analyze_thomson_spectral_distribution
result = analyze_thomson_spectral_distribution(wavelength_nm, signal, background=background, signal_uncertainty=uncertainty)
print(result["centroid_nm"], result["rms_width_nm"], result["skewness"])
```
