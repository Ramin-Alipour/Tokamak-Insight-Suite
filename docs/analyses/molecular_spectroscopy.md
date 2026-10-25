# Molecular Spectroscopy (Analysis 6 of 81)

## Implemented scope

The module `tokamak_analysis.molecular_spectroscopy` provides two executable workflows:

- `analyze_molecular_spectrum`: validates a calibrated wavelength axis and measured intensity; estimates a constant quantile baseline; detects prominent spectral features; reports approximate peak wavelengths, prominence and FWHM; integrates baseline-subtracted signal; optionally integrates user-selected band windows and compares detected peaks with caller-supplied reference bandheads.
- `fit_custom_molecular_model`: fits a caller-provided model with named parameters, initial values, optional bounds and optional per-sample uncertainties. It returns fitted parameters, covariance-based standard uncertainties, fitted spectrum, residuals, and chi-square metrics when uncertainties are supplied. It rejects underdetermined fits (at least one residual degree of freedom is required) and wraps optimizer failures as actionable `ValueError`s.
- Streamlit dashboard: the Molecular spectroscopy workflow exposes baseline/feature analysis and an optional empirical multi-Gaussian fit with user-entered initial band centers. The Gaussian fit is a descriptive approximation only; its parameters are not molecular constants and must not be interpreted as rotational/vibrational temperatures.

## Input contract

Both Python workflows require finite, one-dimensional, matching wavelength/intensity arrays with at least five samples and a positive, strictly increasing wavelength axis. The dashboard additionally rejects duplicate wavelength samples rather than silently combining them. Wavelength calibration is expected to have been performed before analysis. The generic fitter requires a callable model, unique parameter names and finite initial values. Optional uncertainties must be positive, finite and match the spectrum. Optional band-integration windows must be increasing and fully within the measured wavelength range.

## Physical interpretation and limitations

The baseline and peak detection are exploratory numerical tools, not a complete molecular-band inversion. The integrated signal is in intensity-unit·nm; it is not radiated power unless intensity is calibrated accordingly. Reference matches are nearest-wavelength candidates only, using a reference list supplied by the user. No molecular database, species identification, rotational/vibrational temperature, molecular density, or plasma electron/ion temperature is inferred automatically.

Quantitative rotational/vibrational temperature requires a suitable species- and transition-specific model, reliable molecular constants/line lists, instrument line-shape and spectral-response treatment, optical calibration, blend/background treatment and a defensible uncertainty budget. Model fit quality alone does not validate the physics. Molecular excitation temperatures need not equal electron or ion temperatures. Independent experimental validation remains required.

## Example

```python
import numpy as np
from tokamak_analysis.molecular_spectroscopy import analyze_molecular_spectrum

wavelength_nm = np.linspace(400, 430, 3000)  # replace with calibrated data
intensity = np.zeros_like(wavelength_nm)    # replace with measured spectrum
result = analyze_molecular_spectrum(
    wavelength_nm, intensity,
    reference_bandheads=[(405.0, "user-supplied bandhead")],
)
```
