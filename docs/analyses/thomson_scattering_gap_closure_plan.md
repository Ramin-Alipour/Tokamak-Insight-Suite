# Thomson Scattering Analysis 1 — Gap-Closure Engineering Plan

**Branch:** `main`  
**Baseline reviewed:** 2026-11-01  
**Scope:** audit items 7–110, across optical Thomson inference, spectral-distribution analysis, spatial reconstruction, and microwave CTS (electrostatic and full-EM).

## 1. Binding execution contract

All inputs are measured diagnostic data already prepared and calibrated by the instrument/operator. TIS performs physical inference from those measurements. TIS must not calibrate instruments, correct detector counts, repair acquisition data, estimate an unprovided instrument response, or claim experimental validation. Fixed geometry, species composition, equilibrium, polarization, magnetic-field, response-operator and other model metadata may be requested only when the selected forward model mathematically requires them.

Every reported quantity must be classified as:
- **Primary:** identifiable from the measured data and metadata actually supplied.
- **Conditional:** identifiable only after explicitly named auxiliary inputs are supplied; report the missing inputs and the conditional result.
- **Non-identifiable:** not uniquely recoverable under the selected measurement geometry/model; never present as a measured result.

Every numerical physical result must carry units, uncertainty semantics, model identity, and relevant assumptions. Numerical regularization settings and optimizer seeds are not physical measurements.

## 2. Work packages for audit items 7–110

The audit's numbered entries are not all software bugs. They include valid physical identifiability limits, absent numerical capabilities, and intentionally scoped approximations. The following work packages close every implementable gap without falsifying the non-identifiable cases.

### WP-O1 — Optical forward-model completeness and identifiability (items 7–22)
- Retain explicit isotropic Maxwellian, finite-variance Kappa and relativistic Maxwell–Jüttner model selection.
- Make model-specific assumptions and validity screens machine-readable and visible in API/dashboard outputs.
- Separate line-of-sight/scattering-vector flow projection from a 3-D flow vector; multi-view vector recovery is conditional on independent view geometries and joint covariance.
- Keep density dependent on absolute calibrated scattered power and the required laser/collection/geometry metadata; without absolute power, classify density as unavailable/non-identifiable rather than inventing a normalization.
- Expose polarization and geometry dependence only where the selected cross-section supports it.
- Do not treat an optimizer's initial guess as a measured parameter.
- Add model comparison only as a statistical comparison among explicitly supported physical models, with no automatic claim that the preferred model is uniquely true.

### WP-O2 — Spectral-distribution and projected EVDF (items 23–35)
- Keep descriptive wavelength-space moments separate from physical velocity-space inversion.
- Extend projected EVDF reconstruction with explicit response-operator handling, background nuisance treatment, spectral covariance, regularization diagnostics and reproducible selection diagnostics.
- Propagate uncertainty conditionally on the chosen regularization, response, background and active set; label local/conditional approximations as such.
- Implement relativistically consistent projected-velocity inversion before exposing EVDF output for Maxwell–Jüttner fits. Until then, return an explicit unavailable status for that model.
- Preserve the single-view limitation: one view cannot identify an unrestricted 3-D EVDF, anisotropy, perpendicular velocity structure, or absolute density without additional independent information.
- Ensure output schemas never merge descriptive moments with EVDF outputs.

### WP-P1 — Spatially resolved inference and profile reconstruction (items 36–76)
- Add a joint multi-channel spectral inverse in which each channel's physical spectrum is fitted together with spatial regularization, rather than fitting channels independently and smoothing only afterward.
- Support full channel-by-channel spectral covariance and cross-channel covariance, including structured/block covariance where computationally appropriate.
- Add a spatial mixing/point-spread operator as an optional measured/fixed forward operator; do not estimate or calibrate it inside TIS.
- Propagate covariance from channel spectra through the joint inverse to profile parameters, including parameter cross-covariance and external-parameter covariance.
- Provide regularization selection diagnostics (including GCV where mathematically applicable), sensitivity/robustness diagnostics, and explicit warnings when the channel count or geometry is insufficient.
- Offer piecewise-smooth and edge-preserving alternatives only as explicitly selected regularization models; quantify their assumptions and avoid silently smoothing physically sharp transitions.
- Distinguish measured channel locations from interpolated profile-grid points; interpolated values are not independent measurements.
- Add conditional equilibrium/flux-coordinate mapping only when equilibrium and mapping metadata are supplied. Do not invent separatrix or boundary constraints.
- Make all profile algorithms expose rank, conditioning, effective degrees of freedom, residual diagnostics and identifiability status.

### WP-C1 — Electrostatic microwave CTS (items 77–85)
- Preserve the unmagnetized longitudinal dielectric model and explicitly reject use where magnetized electromagnetic response is required.
- Expand the species model to accept explicit species-resolved temperatures, flows and supported non-Maxwellian projected distributions where identifiable.
- Add joint uncertainty propagation for external species, geometry, probe frequency and response inputs, including their supplied cross-covariance.
- Add automatic, data-informed optimizer initialization without treating seed values as physical measurements.
- Add robust parameter scaling, profile likelihood or equivalent identifiability diagnostics, and covariance-aware fitting.
- Make global search bounds explicit researcher inputs; never invent scientific parameter limits.
- Keep fast-ion distribution recovery limited to the supported projected 1-D inverse and state its non-identifiability beyond that projection.

### WP-C2 — Magnetized full-electromagnetic CTS (items 86–110)
- Preserve strict separation from electrostatic CTS; no silent model fallback.
- Generalize inversion beyond Maxwellian bulk electrons only where a fully specified kinetic response and source/vertex model are implemented and documented.
- Develop uncertainty propagation for measured spectral covariance, response-operator uncertainty, magnetic field, geometry, polarization/eigenmode, ray path, equilibrium and beam covariance, including cross-covariance where supplied.
- Add numerical convergence controls that run automatically at declared tolerances for gyro-harmonics, momentum/velocity quadrature, ray integration and scattering-volume integration; report convergence evidence separately from physical uncertainty.
- Implement frequency-resolved mode tracking and mode-conversion handling across branch crossings before claiming continuous ray solutions through degeneracies.
- Extend ray/beam propagation beyond cold-plasma geometrical optics only with an explicit validated mathematical formulation and all required boundary/launch metadata; until then, report diffraction, mode conversion and quasi-optical effects as outside the selected model.
- Add spatially varying plasma response and scattering-volume integration beyond uniform fitted plasma parameters when sufficient equilibrium/profile inputs are supplied.
- Propagate absolute coupling only if calibrated launcher/receiver and coupling information is supplied; otherwise retain relative spectral weights.
- Keep finite-orbit-width, collisional and nonlinear-equilibrium effects explicitly outside the model until separately implemented.
- Keep flow components projected along the scattering wavevector unless independent geometries make a joint vector inversion identifiable.
- Retain Bindslev low-temperature closure screens as model-domain diagnostics, not universal physical thresholds.
- Distinguish numerical resonance damping from physical collision frequency.

## 3. Cross-cutting software requirements

1. One canonical implementation contract must drive Python API metadata, root and packaged JSON registries, dashboard forms, documentation and tests.
2. Add schema-level tests for every required/optional input, units, defaults, result classification and unavailable reason.
3. Add formula-contract tests for dimensional consistency, limiting cases, conservation/normalization properties and independently derived analytic special cases.
4. Add synthetic software regression tests for numerical correctness and failure handling. Synthetic tests are software tests, not experimental validation.
5. Add tests for uncertainty propagation, singular/ill-conditioned covariance, rank deficiency, model-domain rejection and non-identifiability.
6. Keep primary measured outputs separate from conditional outputs and numerical diagnostics.
7. Never silently replace a missing physical input with an arbitrary default. Defaults are allowed only for documented numerical initialization/settings, never for unknown physical state.
8. Every added physical model requires primary literature references, explicit equations, assumptions, supported regimes, required inputs and a regression test tied to a limiting case.
9. Any capability not yet mathematically implemented remains explicitly unavailable; documentation alone does not close a computational gap.
10. Keep the two diagnostic registry JSON files synchronized and the executable Python registry consistent with both.

## 4. Implementation order and completion gates

Implementation is staged to keep the main branch usable and every commit reviewable:

1. **Contract and inventory gate:** map each numbered audit item to (a) existing implementation, (b) physical non-identifiability, or (c) a concrete code gap; identify exact functions and tests.
2. **Optical/spectral gate:** complete WP-O1 and WP-O2, including relativistic projected inversion or explicit unavailable classification.
3. **Profile gate:** implement WP-P1 joint inference and covariance propagation.
4. **Electrostatic CTS gate:** implement WP-C1 and its uncertainty/identifiability coverage.
5. **Full-EM CTS gate:** implement WP-C2 in separable increments, preserving model-domain boundaries.
6. **Integration gate:** synchronize API, dashboard, registries, docs and tests; run GitHub Actions on the resulting `main` commit.

A work package is complete only when its code, public contract, documentation and regression tests agree. A plan or a passing subset of tests is not evidence that all gaps are closed.

## 5. Validation policy

No experimental validation is performed or claimed as part of this development phase. GitHub Actions, analytic limiting-case checks and synthetic regression tests establish software behavior only; they do not establish experimental accuracy or validate a diagnostic installation.

## 6. Completion ledger

The detailed audit remains the source of truth for item numbering. Each item 7–110 must be marked with one of these terminal states only after inspection:
- `implemented_and_tested`
- `non_identifiable_by_physics`
- `conditional_on_explicit_inputs`
- `outside_selected_model_with_reason`

Do not mark an item complete merely because it has been described in this plan.


### Local Full-EM CTS external covariance — partial closure of WP-C2

The Full-EM spectrum inversion accepts an ordered joint covariance over local magnetic-field, incident/scattered direction and complex polarization components and propagates it through reproducible full-parameter Monte Carlo refits. Frequency-resolved ray/equilibrium/beam covariance, composition covariance and cross-covariance between distinct external covariance sources remain open. The local-vector covariance route explicitly rejects ray-resolved geometry. This closes only the local-input portion of audit item 102; it does not close the full WP-C2 external uncertainty work package.
