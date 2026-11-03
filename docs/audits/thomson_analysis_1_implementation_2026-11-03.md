# Thomson Analysis 1 — direct-main implementation record

**Date:** 2026-11-03  
**Target branch:** `main`  
**Base commit for this implementation batch:** `286ba72453001d41c3103664e3e9901b3d75455a`  
**Latest implementation commit at record time:** `56eacbb398c993b9fc504197ae958acab238b7b9`

This record describes changes actually committed to `main`. No local
checkout or local editing environment was used.

## Implemented in this batch

1. Rewrote `docs/analyses/advanced_cts_thomson_operators.md` to match the
   calibrated-measured-data contract. Removed the obsolete claim that a
   public raw-count Poisson likelihood is available.
2. Added the calibrated forward-model comparison helper to the public package
   export list and the Thomson diagnostic registry. Its contract explicitly
   compares completed predictions on common bins/likelihood; it does not fit
   parameters or select physical truth.
3. Updated the Thomson analysis documentation with the model-comparison
   inputs, outputs and statistical limitations.
4. Added `model_discrepancy_covariance` to
   `analyze_full_em_cts_spectrum`. The Full-EM GLS covariance is the sum of
   the supplied measurement covariance (full matrix or pointwise diagonal)
   and the independent model-discrepancy covariance.
5. A discrepancy-only weighting route is allowed and explicitly labelled as
   having no supplied measurement-noise covariance. If neither covariance
   source is supplied, the fit remains unweighted and the local covariance is
   residual-variance-scaled.
6. The discrepancy covariance must be finite, symmetric and positive
   semidefinite; the total covariance must be positive definite for Cholesky
   whitening. No jitter, eigenvalue clipping, residual-derived discrepancy or
   silent regularization is applied.
7. Updated the CTS documentation and registry to describe the Electrostatic
   and Full-EM discrepancy-covariance contracts.
8. Added regression tests for covariance addition, discrepancy-only use,
   invalid/singular covariance rejection, public API exposure, and registry
   consistency.

## Deliberately not represented as completed

The following scientific extensions remain outside this implementation batch
and must not be advertised as completed:

- General drifted relativistic electron-VDF inference in the main Full-EM
  spectrum optimizer.
- Finite-orbit-width corrections and a general collisional kinetic response.
- Profile-varying Full-EM inference of node-wise `n_e`, `T_e` and
  species-resolved `T_i` over the scattering volume.
- Full-vector, non-paraxial, dissipative quasi-optical propagation.
- Cross-covariance between measurement noise and model discrepancy.
- Arbitrary cross-channel optical model-discrepancy covariance.

These are physical/model architecture gaps, not issues that can responsibly be
closed by adding an empirical correction or an undocumented approximation.
They require explicit model equations, well-defined calibrated input
contracts, numerical implementation and regression tests.

## Verification

The GitHub Actions workflows triggered by the final push were visible as
queued at the time of this record:

- Continuous software tests:
  https://github.com/Ramin-Alipour/Tokamak-Insight-Suite/actions/runs/41631373119
- ECE dashboard and test suite:
  https://github.com/Ramin-Alipour/Tokamak-Insight-Suite/actions/runs/41631373140

No successful CI result is claimed until those workflows complete. Synthetic
regression tests are software checks only. No experimental validation is
claimed or performed.
