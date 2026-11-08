# Two-population relativistic optical Thomson inference

## Purpose

This optional Analysis 1 subanalysis fits measured, calibrated optical Thomson
spectra with a two-population isotropic relativistic Maxwell-Juttner model.
The electron distribution is represented as a number-density-weighted sum of
a colder and a hotter Maxwell-Juttner population. Both populations share one
bulk-flow projection along the scattering vector.

The forward model evaluates the two relativistic single-particle Thomson
spectra independently and adds their scattered powers using their
population fractions and effective differential cross-sections. The measured
instrument-response matrix is applied unchanged as a fixed forward operator.

## Fitted parameters

- Cold-population electron temperature.
- Hot-population electron temperature.
- Hot-population number-density fraction.
- Shared electron-flow projection along the scattering vector.

The hot temperature is constrained to exceed the cold temperature to remove
label-switching. The population fraction is physically restricted to [0, 1].
The fit profiles a spectral amplitude and constant detector-space offset as
linear nuisance parameters.

## Input contract

The wavelength axis and spectrum must already be measured and calibrated.
The supplied response matrix must be finite, non-negative, square on the
wavelength grid, and have unit-sum columns; it is used unchanged. The
researcher supplies the laser wavelength, scattering angle, incident
polarization, initial estimates for all four fitted parameters, and either
pointwise spectral uncertainty, full spectral covariance, or neither.

No detector calibration, response measurement, baseline correction,
wavelength correction, or empirical validation is performed.

## Outputs and identifiability

The callable returns both fitted temperatures, hot fraction, shared flow
projection, their local covariance and 1-sigma uncertainties, fitted
spectrum, residual, effective differential cross-section and an explicit
statement that absolute density is not inferred by this normalized
shape-only subanalysis. Rank-deficient population fits are rejected rather
than returning apparently precise component parameters.

This model is a two-component parametric approximation, not an unrestricted
3-D electron velocity distribution and not proof that the plasma contains
exactly two physical populations. A single line of sight cannot identify
anisotropy or transverse flow.

## Literature basis

K. Sakai, K. Tomita, T. Hoshi and R. Yasuhara, “Monte Carlo simulation method
for incoherent Thomson scattering spectra from arbitrary electron
distribution functions,” *Journal of Instrumentation* 21 (2026), C01002,
DOI: 10.1088/1748-0221/21/01/C01002. That paper demonstrates a Monte Carlo
forward method for arbitrary supplied electron VDFs, including relativistic
Maxwellian and kappa distributions. The TIS two-population model is a
transparent low-dimensional parametric approximation motivated by that
arbitrary-VDF direction; it is not the Monte Carlo algorithm from the paper.

No experimental validation is performed or claimed.
