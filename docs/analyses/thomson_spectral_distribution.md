# Thomson Spectral Distribution Analysis (callable registry entry 81; user-facing analysis 7)

## Completion status

**Implemented and integrated within the declared descriptive-analysis scope.** The callable is registered as entry 81 in the current 81-entry analysis registry and exposed as **Thomson spectral distribution** on the dashboard's Spectral Analysis page. The dashboard accepts calibrated wavelength, signal, optional pointwise uncertainty, instrument standard deviation and background-window fraction; executes the analysis; displays fit/moment diagnostics and plots; and exports the full result as JSON. The callable and dashboard interaction have focused automated regression tests.

The numbering differs from the user's 80-item diagnostic shortlist: this is item 7 in that shortlist, while the current software's canonical callable registry assigns it ID 81. This distinction does not change which analysis the dashboard runs.

## Inputs

- `wavelength_nm`: finite, strictly increasing calibrated wavelength axis in nm; at least nine samples.
- `signal`: finite measured spectral signal with matching length.
- `instrument_sigma_nm`: known positive Gaussian instrumental standard deviation in nm.
- `signal_uncertainty`: optional positive finite per-sample 1-sigma uncertainty.
- `background_fraction`: optional fraction of edge samples used to estimate a local linear background; range 0.02–0.4.

## Outputs and method

The workflow estimates a local linear background from both spectral-window edges, preserves the signed background-subtracted spectrum, and calculates positive-weight spectral area, centroid, observed RMS width, skewness and excess kurtosis. Spectral moments use wavelength-space quadrature weights, so nonuniform wavelength sampling does not simply overweight densely sampled regions. It subtracts the known Gaussian instrumental variance from the observed variance. If the result is unresolved or non-positive, the instrument-corrected width and variance are returned as JSON-safe null values rather than fabricated. A Gaussian-plus-linear-background fit supplies fitted parameters, local covariance uncertainties, residuals, R-squared and optional chi-square metrics when sample uncertainties are supplied.

## Scientific limits

- Spectral moments describe the measured wavelength distribution; they do not uniquely recover the electron velocity distribution.
- Instrument-width correction assumes independent Gaussian variances and a known instrument standard deviation.
- Supplied signal uncertainties weight the Gaussian fit and support chi-square metrics. They are not propagated into spectral moments; wavelength-calibration, background-model and instrument-width uncertainties are also not propagated.
- The Gaussian parameter uncertainties are local covariance estimates and can be unreliable for poorly conditioned fits.
- No electron temperature or density is inferred; no collective/relativistic/non-Maxwellian forward model, stray-light correction, or complete instrument-response calibration is performed.
- Use requires appropriate calibration and researcher interpretation. Software tests demonstrate computational behavior, not independent scientific validation.

## API example

```python
from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution

result = analyze_thomson_spectral_distribution(
    wavelength_nm,
    signal,
    instrument_sigma_nm=0.06,
    signal_uncertainty=signal_uncertainty,
)
print(result["spectral_centroid_nm"], result["instrument_corrected_sigma_nm"])
```
