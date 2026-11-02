# Thomson Analysis 1 — gap-closure implementation note

Date: 2026-11-02  
Branch: `main`

This note records the implementation scope of the current gap-closure commits. It deliberately distinguishes implemented numerical paths from physical limits and work that remains conditional or incomplete. Passing software tests is not experimental validation.

## Changes applied

### A5 — relativistic projected EVDF

Added `reconstruct_relativistic_projected_thomson_evdf` in `tokamak_analysis/thomson_gap_closure.py`, re-exported from `tokamak_analysis/thomson_evdf.py`.

The inverse uses the exact elastic Thomson Doppler relation for a one-dimensional velocity marginal parallel to the scattering vector and a symmetric incident/scattered geometry:

[
\frac{\lambda_s}{\lambda_i}
=\frac{1-\beta\sin(\theta/2)}{1+\beta\sin(\theta/2)}.
]

It retains the supplied spectral response, finite-volume wavelength integration, non-negativity, GCV or explicit Tikhonov regularization, and a local active-set covariance diagnostic. It reports a normalized projected marginal only; it never infers absolute density.

**Strict limits:** this is not a general relativistic 3-D EVDF inversion. It omits polarization-dependent differential-cross-section weighting, transverse velocity structure, and Compton recoil. The output explicitly records these assumptions. The routine is a separate API and does not replace the existing non-relativistic `reconstruct_thomson_evdf`.

### A7 — structured optical background

The Maxwellian optical inference path now accepts `background_model`, `background_degree`, `background_interior_knots`, and `background_templates`. It fits the Thomson feature and the selected detector-space basis jointly. Supported bases are those already defined by `thomson_background._background_design`: constant, linear, Legendre polynomial, cubic spline, and supplied templates.

The background remains a detector-space nuisance term, not plasma emission. Its basis choice is researcher-selected. This integration currently applies to the non-collective Maxwellian fit path; equivalent structured-background integration into the Kappa and relativistic Maxwell–Jüttner optical fitters remains open.

### A9 — PSF-aware profile reconstruction

Added `reconstruct_psf_aware_thomson_profile` in `tokamak_analysis/thomson_gap_closure.py`, re-exported from `tokamak_analysis/thomson_workflow.py`.

It solves the measured-channel relation (y=H x+\epsilon) using the researcher-supplied spatial mixing/PSF operator, supplied measurement uncertainty or full channel covariance, and explicit second-difference regularization. Data-only rank and condition diagnostics are computed separately from regularization. A regularized candidate is not reported as a primary profile when the data-only operator is rank-deficient.

The PSF is fixed input; its uncertainty is not propagated by this routine. This does not estimate a PSF or recover modes in the operator null space.

### A12 — spatially varying Full-EM CTS forward volume

The Full-EM scattering-volume quadrature now accepts optional per-sample external profile factors:
- `electron_density_relative`
- `electron_temperature_relative`
- `ion_temperature_relative`

It also permits sample-local overrides for ion fractions, flow inputs, polarization, fast-ion populations, and electron-distribution inputs. The factors are applied at each scattering-volume sample before local Full-EM evaluation.

This supports a forward model with externally supplied spatial variation. It does **not** independently infer arbitrary node-wise (n_e(r)), (T_e(r)), or (T_i(r)) profiles. Covariance of the supplied spatial factors and cross-sample profile covariance are not yet propagated.

## Existing capabilities retained and re-audited

| Gap | Current main status | Qualification |
|---|---|---|
| A1 — optical multi-channel covariance | Existing channel/joint spectral covariance inputs retained | Confirm exact covariance flattening and public API coverage in regression tests before declaring fully closed |
| A2 — Full-EM external uncertainty | Existing local external-input Monte Carlo retained | Ray-resolved geometry covariance and cross-sample beam/equilibrium covariance remain unsupported |
| A3 — joint spatial-spectral multi-model | Existing joint inversion remains Maxwellian | Kappa and relativistic Maxwell–Jüttner joint inversions are not implemented by this change |
| A5 — relativistic EVDF | New exact Thomson-limit 1-D kinematic inverse | No polarized cross-section weighting, Compton recoil, or 3-D reconstruction |
| A7 — structured optical background | Added to non-collective Maxwellian path | Kappa and MJ integration remains open |
| A8 — spatial covariance | Existing full supplied channel covariance in profile reconstruction retained | Coordinate uncertainty and PSF uncertainty remain outside this contract |
| A9 — ordinary profile PSF | New fixed-operator inverse added | Requires supplied PSF/mixing matrix; null-space modes remain non-identifiable |
| A10 — electrostatic CTS external covariance | Existing named external covariance/MC retained | Supported parameter names are intentionally limited; this is not covariance of every possible external input |
| A12 — Full-EM spatial variation | New externally prescribed per-sample plasma profile factors | Spatial profiles are not free inversion parameters; their covariance is not propagated |

## Uncertainty and reporting rules

- Supplied measurement covariance is distinct from residual-scaled covariance.
- Residual-scaled covariance from an unweighted fit is not measurement-noise-calibrated.
- Local regularized inverse covariance is conditional on the selected regularization, active set, fixed forward operators, and physical model; it is not a posterior distribution.
- Regularization rows are not independent measurements and must not be counted toward data-only identifiability.
- A successful optimizer is not, by itself, evidence that all physical parameters are identifiable.
- No absolute density is inferred from a normalized EVDF.
- No 3-D flow or EVDF is inferred from a single scattering vector.
- Numerical convergence and software tests do not constitute experimental validation.

## References

1. C. R. Skolar, W. J. Longley, and L. V. Goodwin, “Computing the generalized plasma dispersion function for non-Maxwellian plasmas, with applications to Thomson scattering,” *Physics of Plasmas* 32, 043907 (2025), DOI: [10.1063/5.0261966](https://doi.org/10.1063/5.0261966). This supports numerical generalized plasma-dispersion treatment for non-Maxwellian kinetic response; the present finite-variance projected optical Kappa parameterization must not be described as an implementation of their algorithm.
2. K. Sakai et al., “Monte Carlo simulation method for incoherent Thomson scattering spectra from arbitrary electron distribution functions,” *Journal of Instrumentation* 21, C01002 (2026), DOI: [10.1088/1748-0221/21/01/C01002](https://doi.org/10.1088/1748-0221/21/01/C01002). This motivates arbitrary-distribution relativistic forward modelling; the present projected inverse is not that Monte Carlo method.
3. H. Bindslev, “Theory of Collective Thomson Scattering,” JET-P(95)66 (1995). The Full-EM model retains its declared kinetic/closure assumptions and is not claimed to be exact in every harmonic, collisional, ray, or beam regime.

## Validation state

The code changes and regression tests have been committed to `main`. A CI pass and empirical validation must be reported only after their respective runs have completed and their results have been inspected. No experimental validation is claimed in this note.
