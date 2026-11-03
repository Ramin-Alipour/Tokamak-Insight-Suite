# Analysis 1 joint-inference operators

The module `tokamak_analysis.thomson_joint_inference` provides reusable
operators for joint multi-channel GLS, profile regularization sensitivity,
reference-library PCA uncertainty, and correlated geometry Monte Carlo.

## Covariance convention

Multi-channel spectra are flattened in channel-major order:
`[channel_0 bins..., channel_1 bins..., ...]`. The complete covariance is
validated as symmetric positive definite and whitened by a Cholesky solve.
Cross-channel covariance terms are retained in the objective and in the local
parameter covariance. Per-channel background bases are nuisance parameters
fitted in the same joint likelihood; physical/background covariance is not
discarded.

## Profile inverse

The linear profile inverse reports the data-only rank separately from the
regularized solution. For every supplied regularization strength it reports
the profile, local covariance, resolution matrix, effective degrees of freedom,
whitened residual objective, roughness, and GCV score. The GCV-selected
profile is a numerical selection, not proof that the data identify every
profile degree of freedom. The full strength scan and pointwise profile span
must be retained in downstream reporting.

## PCA reference library

The PCA uncertainty operator resamples reference spectra or samples a
researcher-supplied full covariance of the flattened reference library. PCA
is refit for each Monte Carlo realization. Reconstruction covariance,
standard deviation, eigenvalue spread, and basis ensembles are returned.
This quantifies finite-library uncertainty only; it does not estimate
measurement noise or guarantee that the reference set spans the true
background process.

## Geometry and beam uncertainty

The geometry ensemble operator samples the complete supplied covariance,
including off-diagonal correlations, and propagates each draw through a
researcher-provided forward calculation. Failed draws are counted and a
minimum success fraction is enforced. The caller must supply a physically
complete beam/ray forward model; this utility does not itself solve
non-paraxial diffraction, dissipative mode conversion, or equilibrium
reconstruction.

## Scope and validation

These routines operate on calibrated measured spectra and explicit forward
operators. Unit tests are numerical software regressions, not experimental
validation. A callback's physics, detector calibration, and validity domain
remain the responsibility of the selected diagnostic model.
