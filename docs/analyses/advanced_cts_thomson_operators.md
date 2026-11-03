# Advanced CTS/Thomson inverse operators

This implementation adds reusable mathematical operators; it does not silently
claim that every operator is already wired into every dashboard workflow.

## Independent count likelihood

`poisson_deviance_residual` and `poisson_negative_log_likelihood` implement
the independent Poisson count likelihood. They accept integer photon/electron
counts only. Calibrated spectral power-density data must continue to use the
Gaussian/GLS path unless a physically defined count conversion is supplied.

## Independent model discrepancy

`gaussian_process_discrepancy_covariance` supplies a correlated additive
discrepancy covariance (squared-exponential, Matérn-3/2 or Matérn-5/2). Its
amplitude and correlation length are explicit researcher-supplied hyperparameters.
It is independent of reduced-chi-square rescaling and must be fitted or selected
with a declared likelihood/model-comparison procedure before it is included in
a final uncertainty interval.

## Full-EM backgrounds

`cts_em_background` computes non-relativistic Maxwellian free-free emissivity
and integrates LTE transfer using a supplied mode-resolved ECE absorption
coefficient. The ECE absorption coefficient must come from a magnetized
dielectric/radiative-transfer solver. Output is spectral radiance; receiver
etendue and efficiency are required to obtain receiver-plane power.

## Quasi-optical beam covariance

`propagate_paraxial_beam_covariance` propagates a transverse 4-D Gaussian
phase-space covariance through supplied symplectic paraxial transfer matrices.
It is not a plasma ray/dispersion solver and does not derive those matrices.
This separation prevents an unverified scalar-beam approximation from being
misrepresented as full quasi-optics.

## Non-Maxwellian distributions

`reconstruct_multiview_3d_evdf` solves a non-negative, regularized 3-D inverse
problem using per-view spectral kernels. For a relativistic result, those
kernels must be computed with the relativistic per-view scattering physics.
The data-only stacked kernel rank and condition number determine whether the
3-D distribution is identified. A single view or rank-deficient data returns
only a conditional regularized candidate, never a claimed 3-D measurement.

The new utilities are deterministic software components. No experimental
validation is claimed. Existing Full-EM inverse, electrostatic CTS inverse,
optical Thomson fit and dashboard APIs remain separate until explicit
integration work connects these operators and their uncertainty contracts.
