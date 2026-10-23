# Thomson Spectral Distribution Analysis (Analysis 7 of 80)

## Implemented workflow
The analysis is available in `tokamak_analysis.thomson_distribution.analyze_thomson_spectral_distribution` and is integrated into the spectral-analysis interface and analysis registry.

## Inputs
- `wavelength_nm`: finite, strictly increasing calibrated wavelength axis in nm; at least nine samples.
- `signal`: finite measured spectral signal with matching length.
- `instrument_sigma_nm`: known positive Gaussian instrumental standard deviation in nm.
- `signal_uncertainty`: optional positive finite per-sample 1-sigma uncertainty.
- `background_fraction`: optional fraction of edge samples used to estimate a local linear background; range 0.02–0.4.

## Outputs and method
The workflow estimates a local linear background from both spectral-window edges, preserves the background-subtracted spectrum, and calculates positive-weight spectral area, centroid, observed RMS width, skewness and excess kurtosis. Spectral moments use trapezoidal wavelength-space quadrature weights, so nonuniform wavelength sampling does not incorrectly overweight densely sampled regions. It subtracts the known Gaussian instrumental variance from the observed variance. If the result is unresolved or non-positive, the instrument-corrected width is explicitly returned as null rather than fabricating a value. A Gaussian-plus-linear-background fit supplies shape diagnostics, residuals, R-squared and optional chi-square metrics when sample uncertainties are supplied.

## Scientific limits
- Spectral moments describe the measured wavelength distribution; they do not uniquely recover the electron velocity distribution.
- Instrument-width correction assumes independent Gaussian variances and a known instrument standard deviation.
- Gaussian-fit parameter uncertainties are local covariance estimates; spectral-moment uncertainties are not propagated.
- No electron temperature or density is inferred; no collective/relativistic/non-Maxwellian forward model, stray-light correction, or complete instrument-response calibration is performed.
- The workflow requires independent calibration checks and experimental validation before scientific interpretation.

## Example
```python
from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution
result = analyze_thomson_spectral_distribution(
    wavelength_nm, signal, instrument_sigma_nm=0.06,
    signal_uncertainty=signal_uncertainty,
)
print(result["spectral_centroid_nm"], result["instrument_corrected_sigma_nm"])
```
