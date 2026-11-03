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


## Relativistic channel-wise structured-background and full-GLS parity

The relativistic Maxwell-Juttner multi-channel fitter now fits a researcher-
selected detector-space background basis simultaneously with its shared
physical parameters and channel amplitudes. Supported bases are constant,
linear, polynomial, cubic spline and researcher-supplied templates; template
matrices may be shared or channel-specific. The full supplied channel-major
spectral covariance, including cross-channel blocks, is used in the same
whitened objective. The complete parameter covariance is transformed from
optimizer coordinates to physical amplitude/background units, and outputs
retain basis coefficients, basis matrices, model identity and parameter
ordering.

The Analysis 1 workflow forwards the same background selection to the
cross-channel relativistic fit. The Streamlit channel-wise background controls
are enabled for the relativistic model and pass the selected basis/template
to the canonical inference API. Python and JSON diagnostic contracts and the
Thomson analysis documentation were synchronized. Regression tests cover
polynomial-background recovery and rank-deficient template rejection.

This is conditional on the supplied calibrated response and positive-definite
measurement covariance. It does not remove common-Te/common-flow assumptions,
and it does not infer background templates or cross-covariances. Software test
execution and CI success are not claimed until confirmed for the exact commit;
no experimental validation is claimed.


The same basis parameterization was also carried through the full
channel-major GLS refit for Maxwellian and Kappa models. The joint optimizer
now uses each channel's selected basis matrix rather than hard-coded affine
background terms, initializes from the fitted basis coefficients, and reports
the background-coefficient covariance and physical/background cross-covariance.
The dashboard's channel-wise basis controls now cover all three optical
physical models. A regression test exercises polynomial-background recovery
under non-zero cross-channel covariance.


## Spatial-profile CTS inverse, drifted EVDF and regularization sensitivity

A dedicated `full_em_spatial_profiles` dispatcher mode was added. It fits
node-wise relative electron-density and electron-temperature profiles, with
optional ion-temperature profiles, over a stable scattering-volume node set.
Each optimizer evaluation calls the local Full-EM kinetic solver at every
measured frequency and spatial quadrature point. The measured response matrix
is applied once after volume integration using the frequency-bin widths, and
the spectral GLS objective retains the supplied full covariance. Profile
covariance is withheld when the augmented Jacobian is rank deficient.

An optional increasing grid of regularization strengths reruns the spatial
inverse and reports the resulting profile spread explicitly as
regularization-choice sensitivity, not measurement uncertainty. The API is
exposed through the CTS dispatcher, Python analysis registry, both diagnostic
registry JSON copies, and a dedicated Streamlit dashboard mode. Tests cover
local-profile forwarding, stable node identity, covariance requirements,
regularization sensitivity, and registry/dashboard exposure.

The Full-EM relativistic kinetic momentum quadrature now accepts an explicit
Lorentz-boosted drifted Maxwell-Juttner electron equilibrium. Momentum
derivatives use the boost-chain-rule derivatives and the parallel quadrature
support is widened for the boosted distribution. The drift must be subluminal
and is encoded in the VDF; a second flow-projection shift is rejected.

These changes are committed directly to `main`. GitHub Actions runs are
triggered by the final push; no successful test or CI result is claimed until
the runs complete. No experimental validation is claimed.
