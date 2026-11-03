# CTS TS-01–TS-20 remediation ledger

**Branch:** `main`  
**Purpose:** implementation gate for the 20 CTS findings. This ledger distinguishes code that exists from physics that is not implemented; documentation or a passing unit test is not evidence that a missing physical model exists.

## Verified status from the current main-branch source

| ID | Required capability / contract | Current evidence | Status |
|---|---|---|---|
| TS-01 | Collision operator and fluctuation-dissipation-consistent CTS response | Electrostatic path has a limited BGK-style response; no velocity-dependent Landau operator plus consistent fluctuation source/closure is wired into the spectral-density dispatcher. | OPEN |
| TS-02 | Independent ECE absorption solver | New `kinetic_ece_absorption_coefficient` evaluates mode-resolved Maxwellian kinetic conductivity using the existing magnetized Vlasov kernel and unit-Poynting-flux cold eigenmode. Explicit damping is a numerical regularizer, not collisional physics; no arbitrary relativistic EVDF or conserving collision closure. Not yet wired into CTS background transfer/dashboard. | PARTIAL |
| TS-03 | Relativistic, magnetized thermal bremsstrahlung | Existing background treatment is not a general relativistic magnetized free-free emissivity/opacity solver. | OPEN |
| TS-04 | Full-vector, non-paraxial electromagnetic propagation | New `propagate_stratified_vector_maxwell` integrates the full tangential 4-component Maxwell system for a z-stratified anisotropic dielectric, without paraxial approximation. Transverse diffraction, x/y inhomogeneity and reflected-wave boundary matching remain unsupported; not yet wired into CTS beam transfer. | PARTIAL |
| TS-05 | Mode conversion and branch tracking | The new stratified vector Maxwell propagator retains off-diagonal dielectric coupling and reports cross-polarized power, so polarization conversion is represented in the 1-D initial-value problem. General 3-D branch tracking through degeneracies and two-sided reflection/scattering matrices remain open; dashboard/CTS integration remains open. | PARTIAL |
| TS-06 | General relativistic anisotropic EVDF | Several explicit gyrotropic distributions and constrained shape fitting exist; arbitrary/general EVDF reconstruction and unrestricted joint inference do not. | PARTIAL |
| TS-07 | Consistent physical-background model in every optical path | Background integration is not a universal shared model across all optical analysis routes. | PARTIAL |
| TS-08 | Full CTS geometry/profile covariance | Selected geometry ensembles and spatial covariance paths exist, but do not constitute a complete joint covariance model for every profile, ray, beam and inversion input. | PARTIAL |
| TS-09 | General multi-population plasma model | Multi-ion species and selected fast-ion populations exist; a general, consistently coupled arbitrary multi-population model is not established. | PARTIAL |
| TS-10 | General kinetic response for Kappa distributions | Existing Kappa support is parameterized and restricted; it is not a general kinetic susceptibility/noise response for arbitrary Kappa index and anisotropy. | OPEN |
| TS-11 | Explicit physical-background zero-state | Omitted state now reports `uncharacterized_constant_offset` and `physical_background_zero=null`; only an explicit `assumed_zero` declaration asserts zero. Reference-PCA remains distinct. Tests added; dashboard-wide declaration enforcement remains open. | PARTIAL |
| TS-12 | Explicit CTS profile range / support | Defaults and automatic numerical seeds remain in parts of the CTS workflow; a universal explicit support contract is not established. | OPEN |
| TS-13 | Explicit regularization strength and scope | Regularized inversion requires positive `regularization_strength`; inferred tabulated VDF requires explicit `fast_ion.smoothness`; scope is returned in result metadata. | IMPLEMENTED FOR DECLARED CTS REGULARIZERS |
| TS-14 | One spectral-response contract | Electrostatic and Full-EM paths validate the measured response operator and keep response uncertainty model-specific; cross-path parity still requires regression coverage. | PARTIAL |
| TS-15 | External uncertainty in dashboard | Several external-uncertainty paths and covariance components exist; dashboard coverage is not yet demonstrated for every supported input/model. | PARTIAL |
| TS-16 | Descriptive spectrum analysis separated from physical inference | Descriptive spectral analysis and physical inference are distinct callables, but dashboard/registry separation needs end-to-end verification. | PARTIAL |
| TS-17 | Model-discrepancy uncertainty in all subanalyses | Model-discrepancy covariance is supported in selected CTS paths; universal propagation to every subanalysis is not established. | PARTIAL |
| TS-18 | Multi-parameter identifiability | Jacobian rank and condition diagnostics exist in CTS inversion; complete profile-likelihood/posterior and practical-identifiability coverage is not established. | PARTIAL |
| TS-19 | Consistent documentation and removal of stale claims | CTS documentation contains explicit limitations, but cross-checking every registry, dashboard and legacy document remains necessary. | PARTIAL |
| TS-20 | CI on final commit | Must be verified from GitHub Actions checks for the exact resulting commit; no success is inferred from local reasoning. | NOT VERIFIED |

## 2026-11-03 implementation delta\n\nCommit `345bdf5` adds `tokamak_analysis/thomson_advanced_solvers.py`: a mode-resolved Maxwellian kinetic ECE absorption routine based on the existing magnetized Vlasov conductivity kernel, and a non-paraxial 1-D stratified anisotropic vector-Maxwell propagator with polarization coupling. Commit `bf40dc0` adds vacuum phase/sign and singular-tensor tests. The module is exported from the package namespace. These additions are deliberately classified PARTIAL: they are not a complete relativistic ECE transfer model, do not include a conserving Landau-FDT collision operator, and are not yet connected to the CTS physical-background/beam-transfer/dashboard dispatchers.\n\n## Implementation rules

1. Do not label an externally supplied coefficient as an internally solved physical model.
2. Do not replace a missing collision operator, mode-conversion solver or relativistic emissivity with a fitted scale factor or phenomenological broadening and call it complete.
3. Preserve the distinction between forward-model capability, inversion capability, dashboard exposure and experimental validation.
4. Every physical extension must include unit tests for limiting cases, conservation/normalization, positivity or passivity where applicable, numerical convergence, and a documented validity domain.
5. A CI pass establishes software checks only; it does not validate the physical model against an experiment.

## Release gate

Analysis 1 must not be described as having all TS-01–TS-20 closed until every OPEN/PARTIAL item has a code-level implementation, tests, public API/dashboard integration where applicable, synchronized documentation, and a successful CI run on the exact release commit.
