# Analysis 1 current-contract correction — 2026-11-03

This page supersedes stale statements in older Analysis 1 notes.

- All public Analysis 1 physical inference starts from measured, prepared and
  calibrated diagnostic data. Raw integer photon-count likelihood is not a
  public input path.
- The former `fit_optical_thomson_poisson_counts` API was removed from the
  public runtime and registries. References describing it as available are
  obsolete.
- Optical model-discrepancy covariance is supported in the calibrated
  Maxwellian, finite-variance Kappa and relativistic Maxwell-Juttner
  single-channel inference paths. This does not imply arbitrary cross-channel
  discrepancy covariance or a universal covariance API across CTS.
- Full-EM CTS branch tracking and its optional lossless coupled-mode utility
  are conditional local propagation tools, not a complete dissipative
  full-wave mode-conversion solver.
- Supplied spatial-coordinate covariance is propagated by the profile
  reconstruction API; coordinate/value cross-covariance and automatic
  equilibrium/flux-coordinate mapping are not inferred.
- Full-EM CTS still assumes spatially uniform fitted plasma parameters over
  its sampled scattering volume. A profile-varying kinetic response requires
  evaluating the kinetic forward model at each spatial quadrature point.

The calibrated forward-model comparison callable is
`tokamak_analysis.thomson_model_comparison.compare_thomson_forward_models`.
It compares completed forward predictions on the same measured bins and
likelihood, and reports Gaussian GLS objective, AIC, AICc and BIC. It does not
fit parameters and does not validate a physical model.

Synthetic software regression tests are not experimental validation.
