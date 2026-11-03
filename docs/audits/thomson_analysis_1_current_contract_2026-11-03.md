# Analysis 1 current-contract correction — 2026-11-03

This page supersedes stale statements in older Analysis 1 notes.

- All public Analysis 1 physical inference starts from measured, prepared and
  calibrated diagnostic data. Raw integer photon-count likelihood is not a
  public input path.
- The former `fit_optical_thomson_poisson_counts` API was removed from the
  public runtime and registries. References describing it as available are
  obsolete.
- Optical model-discrepancy covariance is supported in the calibrated
  Maxwellian, finite-variance Kappa and relativistic Maxwell-Juttner inference
  paths as common within-channel, channel-specific, or full channel-major
  covariance. The full channel-major form retains supplied cross-channel
  discrepancy blocks in the joint GLS objective. This remains an optical
  Analysis 1 contract and is not a universal covariance API across CTS.
- Full-EM CTS branch tracking and its optional lossless coupled-mode utility
  are conditional local propagation tools, not a complete dissipative
  full-wave mode-conversion solver.
- Supplied spatial-coordinate covariance is propagated by the profile
  reconstruction API; coordinate/value cross-covariance and automatic
  equilibrium/flux-coordinate mapping are not inferred.
- The ordinary Full-EM bulk optimizer retains spatially uniform fitted plasma
  parameters. A separate spatial-profile inverse now fits node-wise relative
  density and temperature profiles by evaluating the local Full-EM kinetic
  forward model at every supplied scattering-volume quadrature point. It is
  conditioned on reference bulk values, node geometry, beam weights and mode
  polarizations; it does not infer those inputs.

The calibrated forward-model comparison callable is
`tokamak_analysis.thomson_model_comparison.compare_thomson_forward_models`.
It compares completed forward predictions on the same measured bins and
likelihood, and reports Gaussian GLS objective, AIC, AICc and BIC. It does not
fit parameters and does not validate a physical model.

Synthetic software regression tests are not experimental validation.


## Addendum — spatial CTS inversion and relativistic drift (2026-11-03)

The dedicated dispatcher model `full_em_spatial_profiles` is implemented by
`tokamak_analysis.cts_spatial_inference.analyze_full_em_cts_spatial_profiles`.
It fits node-wise relative electron density and electron temperature, with
optional node-wise ion temperature, using full spectral GLS covariance,
measured-response convolution with finite frequency-bin widths, explicit
profile regularization and local identifiability diagnostics. Optional
regularization-grid refits report regularization-choice sensitivity separately
from measurement uncertainty. The ordinary bulk inverse remains unchanged.

The Full-EM forward model also accepts an explicitly supplied
`drifted_relativistic_maxwell_juttner` electron distribution. Its momentum
space density and derivatives are evaluated after a Lorentz boost parallel to
the magnetic field. The drift is part of the distribution and must not be
duplicated as an additional electron-flow frequency shift. This is a forward
population capability; it does not claim that the distribution or drift is
uniquely inferred by the bulk spectral optimizer.

A conserving linearized Landau collision operator with its consistent
fluctuation source, and non-paraxial dissipative full-vector beam propagation,
remain outside the implemented model scope. Classical collision-rate estimates
and paraxial/cold-plasma ray propagation must not be described as those
complete operators.

## Addendum — full cross-channel optical model-discrepancy covariance (2026-11-03)

The canonical optical Analysis 1 API now accepts a complete channel-major
`model_discrepancy_covariance` matrix with shape
`(n_channels*n_wavelength, n_channels*n_wavelength)`. Its off-diagonal
channel blocks are validated and retained rather than discarded. The matrix is
added exactly once to a supplied full measurement covariance; if pointwise
measurement uncertainties are supplied, their diagonal variances are first
assembled in channel-major order. A discrepancy-only matrix must be positive
definite to define GLS weights. The same addition logic applies to the
Maxwellian, finite-variance Kappa and relativistic Maxwell-Juttner joint
optical fits. Existing common and per-channel discrepancy-block forms remain
supported.

The full channel-major GLS route also accepts the selected structured
detector-space background basis for Maxwellian, Kappa and relativistic
Maxwell-Juttner models. Registry copies and the canonical Thomson analysis
documentation describe the expanded covariance shape and background contract.
Regression tests cover a correlated two-channel discrepancy matrix and reject
an indefinite full matrix. These are software regressions, not experimental
validation.



## Addendum — spatial CTS covariance and response parity (2026-11-03)

The dedicated spatial Full-EM CTS profile inverse now accepts an optional
independent `model_discrepancy_covariance` in addition to exactly one
measurement-noise covariance representation. The GLS covariance is formed as
`C_effective = C_measurement + C_model` exactly once. The model term must be
finite, symmetric and positive semidefinite; the effective covariance must be
positive definite. No implicit jitter, eigenvalue clipping or residual-derived
model error is used. The result reports the measurement, discrepancy and
effective covariance separately, and the optional term is forwarded through
regularization-sensitivity refits without double counting.

The same spatial inverse now enforces the common measured frequency-response
contract: finite, non-negative, square and unit-sum columns, applied unchanged
after spatial integration. The Streamlit workflow exposes an optional
model-discrepancy covariance CSV and forwards it to the canonical API. The
Python runtime registry and both byte-identical JSON registry copies declare
the input and output semantics. Regression tests cover additive covariance,
indefinite-matrix rejection, response normalization and dashboard/registry
exposure.

This closes the spatial-profile route's covariance-integration and response
contract gaps within the declared model. It does not supply geometry/profile
cross-covariance, infer missing correlations, or validate the underlying
physical Full-EM model experimentally.
