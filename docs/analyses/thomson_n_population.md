# N-population relativistic optical Thomson inference

## Purpose

This Analysis 1 subanalysis fits a measured, calibrated optical Thomson spectrum
with an explicitly selected finite mixture of 2–8 isotropic Maxwell-Juttner
electron populations. It extends the existing two-population parametric model
without claiming that a finite mixture is an unrestricted reconstruction of
the electron velocity distribution.

## Physical model and constraints

Each component is evaluated with the relativistic single-particle Thomson
forward model (or, when explicitly selected, the single-scattering Compton
kernel with polarized Klein-Nishina weighting). The component spectra are
combined by number-density fraction and their component-specific effective
differential cross-sections. The measured instrument response is applied once
as a fixed forward operator.

- Each population has its own positive temperature.
- Temperatures are strictly ordered to remove label switching.
- Number fractions are strictly positive and sum to one; the optimizer uses a
  softmax parameterization.
- All populations share one bulk-flow projection along the scattering vector.
- A positive spectral amplitude and a selected detector-space background basis
  are nuisance parameters and are fitted jointly.
- The model does not infer absolute density, transverse flow, anisotropy or an
  unrestricted 3-D EVDF.

The population count is researcher-selected (2–8); the fit does not select the
number of populations automatically. Initial temperatures and fractions are
optimizer seeds, not measurements. Rank-deficient or ill-conditioned fits are
rejected rather than returning falsely precise component parameters.

## Input and uncertainty contract

Inputs are a strictly increasing calibrated wavelength axis, measured
non-negative spectrum, laser wavelength, scattering angle, measured
column-normalized response matrix, incident polarization, population count,
strictly increasing initial temperatures, positive initial fractions summing
to one, and an initial subluminal flow projection. Optional uncertainty is
either pointwise 1-sigma or a full positive-definite wavelength covariance.
An independent additive model-discrepancy covariance may be supplied. The
background model may be constant, linear, polynomial, cubic spline, or
researcher-supplied templates.

The returned joint covariance includes temperature, fraction, flow, amplitude
and background coefficients, including their cross-covariances. When no
measurement covariance is supplied, covariance is residual-scaled and is not
measurement-noise calibrated. All results are conditional on the fixed
instrument response, selected population count, physical model and supplied
noise model.

## Interpretation and limitations

A finite mixture is a compact parametric approximation to a potentially
non-Maxwellian distribution. It is not proof that the plasma contains exactly
N distinct populations. One scattering geometry identifies only the
scattering-vector flow projection and does not determine transverse flow or
unrestricted anisotropy. Absolute density requires a separate absolute-power
and collection-geometry inference. No experimental validation is claimed.

## Literature context

Sakai et al., “Monte Carlo simulation method for incoherent Thomson scattering
spectra from arbitrary electron distribution functions,” *Journal of
Instrumentation* **21** (2026), C01002, DOI:
[10.1088/1748-0221/21/01/C01002](https://doi.org/10.1088/1748-0221/21/01/C01002).
The TIS implementation is a finite-dimensional Maxwell-Juttner mixture fit;
it is not the paper's Monte Carlo algorithm.
