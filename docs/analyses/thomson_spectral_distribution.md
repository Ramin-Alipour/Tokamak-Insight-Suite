# Thomson spectral distribution analysis — subanalysis 7

## Canonical registry and execution path

This subanalysis is registered as ID 7 in the 80-entry callable registry and belongs to primary Analysis 1 (Thomson Scattering). It accepts measured, calibrated multi-channel Thomson spectra and delegates physical parameter extraction to the canonical forward-inference engine in tokamak_analysis.thomson_inference.

The lower-level function analyze_thomson_spectral_distribution remains available as a supplementary, model-independent description of a single measured spectral shape. It is not the registered physical-inference workflow and must not be interpreted as an electron velocity-distribution measurement.

## Physical-inference inputs

- Calibrated wavelength axis and measured spectra with shape (channel, wavelength)
- Calibrated radius for each channel
- Probe laser wavelength and scattering angle
- Measured instrument-response callable, kernel or matrix
- Optional pointwise signal uncertainty
- Model-specific inputs required by the selected inference model

## Physical outputs

The canonical inference engine returns per-channel electron temperature, supported electron-flow projection, electron density only where identifiable, fit residuals, covariance and model diagnostics. Absolute non-collective density requires calibrated W/nm spectral power and the complete optical collection/polarization geometry.

## Supplementary descriptive helper

The helper analyze_thomson_spectral_distribution accepts one spectrum and either:
- a scalar Gaussian instrument width, or
- a measured general instrument-response matrix.

It uses signed background-subtracted spectral samples for moments; negative residuals are not clipped. A caller-supplied boolean background mask can replace the edge-background fallback. When signal uncertainty is supplied, moment uncertainty is estimated by deterministic-seed parametric Monte Carlo. Instrument-width uncertainty is included in the corrected-width uncertainty. A general response matrix is used in the Gaussian forward fit, but no scalar Gaussian deconvolution is claimed for that case.

## Model limitations

- Maxwellian non-collective inference is restricted to the declared non-relativistic regime.
- Kappa is a parameterized one-dimensional projection, not arbitrary VDF reconstruction.
- Relativistic Maxwell-Juttner is a one-dimensional projected approximation, not full three-dimensional photon-geometry integration.
- The current optical collective Maxwellian path requires externally supplied ion temperatures and ion/electron flow vectors; these are not fitted as independent CTS outputs.
- Microwave CTS and fast-ion CTS inversion are separate capabilities and are not implied by this optical Thomson workflow.
- Shared wavelength-bin/channel covariance and model-discrepancy uncertainty remain incomplete.
- No detector calibration, source-data repair or experimental validation is performed.

## API

```python
from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution_inference

result = analyze_thomson_spectral_distribution_inference(
    wavelength_nm,
    spectra,
    radius_m,
    laser_wavelength_nm=532.0,
    scattering_angle_rad=1.57079632679,
    instrument_response=response_matrix,
)
```
