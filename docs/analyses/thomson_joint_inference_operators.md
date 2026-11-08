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


## Full measured spatial-PSF matrix uncertainty in the joint spectral inverse

For the joint detector model
`Y_{cλ}=\mathcal R_c[\sum_{a,b}H_{ca}P_{ab}S_b(λ;p)]+B_c(λ)`,
the derivative with respect to a measured PSF element is
`∂Y_c/∂P_{ab}=H_{ca}\mathcal R_c[S_b]`.
The data-only GLS implicit sensitivity is
`S_P=−(J_pᵀJ_p)^+J_pᵀJ_P`, giving
`C_{p,P}=S_P C_P S_Pᵀ`. An optional measured-spectrum/PSF cross-covariance
`C_{yP}` contributes both terms
`S_y C_{yP}S_Pᵀ+S_P C_{yP}ᵀ S_yᵀ`. The complete joint
spectrum/PSF covariance block is checked for positive semidefiniteness before
the cross term is used. Matrix-element covariance uses row-major order over
the supplied measured PSF matrix and requires that explicit matrix; it cannot
be attached to a generated Gaussian-width PSF. This is a first-order local
linearization conditional on the nominal PSF, mixing matrix, physical model,
response and selected regularization; it does not enforce positivity or
column-mass constraints on random PSF perturbations. The Dashboard exposes
both covariance uploads in the joint spatial-spectral workflow.
