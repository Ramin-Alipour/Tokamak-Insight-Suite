# Optical Thomson anisotropic projected electron VDF kernel

The kinetic-model module now exposes
`project_bi_maxwellian_electron_vdf` for the gyrotropic, non-relativistic
bi-Maxwellian reference distribution.

For magnetic-field unit vector `b` and scattering-vector unit vector `q`,
the one-dimensional projected temperature is

```text
T_q = T_perp + (T_parallel - T_perp) (b · q)^2
sigma_q^2 = e T_q / m_e
f_q(v_q) = exp[-(v_q-u_q)^2/(2 sigma_q^2)] / (sqrt(2 pi) sigma_q)
```

The function evaluates the analytic projected marginal on a supplied velocity
grid. It does not renormalize a truncated grid; instead it reports
`velocity_grid_probability_coverage`, so finite-grid loss is explicit.

**Identifiability:** one scattering direction identifies only `T_q`. It does
not separately identify `T_parallel` and `T_perpendicular`. Independent
scattering pitch angles are required to separate those temperatures, and the
result remains conditional on the gyrotropic bi-Maxwellian assumption.

**Scope:** this is a tested forward-distribution kernel for OPT-03. It is not
yet an end-to-end anisotropic optical Thomson spectral forward model, optimizer,
dashboard selection, or general anisotropic EVDF inversion. No claim is made
that OPT-03 is fully closed.

Regression coverage is in
`tests/test_thomson_anisotropic_projection.py`. GitHub Actions is responsible
for the test result on the commit that contains this note.


## Multi-view measured-spectrum inference

For view i, the directional temperature is

    T_q,i = T_perp + (T_parallel - T_perp) (b_hat_i · q_hat_i)^2
    sigma_v,i^2 = e T_q,i / m_e
    lambda_s/lambda_0 ~= 1 - q_hat_i · v_i/c

The implementation evaluates a Gaussian projected marginal, maps it to the wavelength domain using the non-relativistic elastic-Thomson Doppler relation, convolves it with the supplied measured response operator, and jointly fits T_parallel, T_perpendicular, per-view q-projected flow, amplitude and linear detector background. Measurement and model-discrepancy covariance are retained separately and combined once. The optimizer's numerical success is not sufficient: separate anisotropy temperatures are reportable only when the squared pitch-angle design has rank two and the spectral Jacobian is full rank.

The route is deliberately limited to a gyrotropic bi-Maxwellian with zero heat flux and small-beta kinematics. It does not estimate magnetic geometry, infer a 3-D velocity vector from one view, apply relativistic corrections, or calibrate the instrument. Tests use synthetic spectra and are software regressions, not experimental validation.


## Projected EVDF additive covariance contract

The non-relativistic reconstruct_thomson_evdf inverse accepts optional
model_discrepancy_covariance and systematic_covariance matrices on the
measured wavelength grid. Each matrix must be finite, symmetric and positive
semidefinite. They are added to the signal/background residual covariance
before GLS whitening and the active-set inverse covariance calculation.
A positive-definite sum may also be used as a discrepancy/systematic-only
weighting model when measurement noise is not supplied.

The implementation assumes these additive components are mutually
independent and independent of the signal/background errors. It does not
estimate them from residuals or silently invent their cross-covariances.
The result returns the supplied matrices, inclusion flags and this covariance
provenance. Reported EVDF uncertainty remains local and conditional on the
velocity grid, response, background, regularization and active set; it is not
a posterior interval or experimental validation. Dashboard/API input-shape
coverage and regression tests are in tests/test_thomson_evdf.py.
