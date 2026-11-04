# CTS TS-01–TS-20 remediation ledger

**Target:** `main`  
**Audit basis:** source and public entry points currently present on `main`.  
**Rule:** a kernel, a passing unit test, a dashboard control, and a complete physical inference path are different deliverables. Status below is deliberately scoped; no item is called complete outside its stated validity domain.

| ID | Required capability / contract | Verified implementation on main | Status |
|---|---|---|---|
| TS-01 | Collisional CTS with fluctuation-dissipation consistency | `cts_landau_fdt.py` assembles a finite-box 3-D multispecies linearized Landau weak-form operator with test/field-particle terms, discrete number/momentum/energy invariants and matching classical FDT noise. `analyze_landau_fdt_cts` adds self-consistent electrostatic Poisson screening and CTS dynamic-structure-factor output. Common-temperature/common-flow isotropic Maxwellians; non-relativistic finite-box reference discretization. | IMPLEMENTED WITH DECLARED DOMAIN; Full-EM coupling and convergence gate remain |
| TS-02 | Independent ECE absorption solver | `ece_kinetic.py` exposes magnetized non-relativistic Maxwellian and relativistic gyrotropic momentum-space Vlasov conductivity projected onto a supplied local mode and normalized by its Poynting flux. The relativistic route is public through `run_advanced_thomson_physics('relativistic_kinetic_ece')`; the two-mode LTE path remains a separate capability. | PARTIAL: relativistic single-mode and segment-resolved collisionless forward absorption are implemented; conserving collisional response, self-consistent ray/eigenmode coupling and relativistic polarized transfer remain open |
| TS-03 | Relativistic and magnetized bremsstrahlung | `cts_em_background.py` supplies non-relativistic unmagnetized Maxwellian free-free emissivity per steradian with the corrected per-steradian SI coefficient `(6.8e-38 × 0.1)/(4π)` and explicit Gaunt factor; a regression test locks the units/normalization. Relativistic and magnetized-mode emissivity/opacity remain open. | OPEN for relativistic and magnetized-mode emissivity/opacity |
| TS-04 | Full-vector non-paraxial Full-EM propagation | Exact vector angular-spectrum propagation exists for homogeneous isotropic segments; oblique Berreman and the stratified beam wrapper retain anisotropy, diffraction, longitudinal fields and coherent polarization. | PARTIAL: arbitrary 3-D inhomogeneous anisotropic media and dissipative interfaces remain |
| TS-05 | Mode conversion | Coherent Jones reflection/transmission and two-mode opacity/coupling are available for declared stratified boundaries and supplied coupling generators. | PARTIAL: robust branch tracking through degeneracies and arbitrary 3-D conversion remain |
| TS-06 | General relativistic anisotropic EVDF | Selected constrained gyrotropic/drifted distributions and parametric fits exist. | PARTIAL: unrestricted/general EVDF inversion and complete Full-EM coupling remain |
| TS-07 | Shared physical background across all optical paths | Physical and detector-space background modules exist, with several optical workflow integrations. | PARTIAL: universal path parity and end-to-end dashboard verification remain |
| TS-08 | Full CTS geometry/profile covariance | External geometry uncertainty, profile covariance and selected cross-covariance paths exist. | PARTIAL: one joint covariance contract across all ray, beam, profile and inversion inputs remains |
| TS-09 | General multi-population plasma model | N-population relativistic Maxwell-Jüttner optical forward model and selected CTS multi-ion/fast-ion paths exist. | PARTIAL: a general coupled population model across all CTS/Full-EM paths remains |
| TS-10 | General kinetic Kappa response | In addition to the optical Kappa fit, `cts_kappa.kappa_electron_susceptibility` now evaluates the exact 1-D marginal of an isotropic 3-D Kappa distribution for collisionless unmagnetized longitudinal susceptibility; it is exposed through the advanced-physics dispatcher and both registries. | PARTIAL: bounded electrostatic susceptibility only; no conserving collision operator, Kappa fluctuation/noise closure, magnetized dielectric, relativistic Kappa response or Full-EM CTS vertex |
| TS-11 | Explicit zero-background state | CTS spatial inversion requires explicit `background_mode='zero'` or `'spectrum'`; background integrator reports exact zero only when plasma and incident radiance are zero. Other paths retain legacy/optional background contracts. | PARTIAL |
| TS-12 | Explicit CTS profile support | Full-EM spatial inversion requires an explicit positive `profile_bounds_relative`; no scientific range is silently selected. | IMPLEMENTED in spatial Full-EM inverse; audit remaining CTS profile paths |
| TS-13 | Explicit regularization strength | Spatial Full-EM inverse requires explicit per-fit regularization strength and reports its scope separately from measurement covariance. Boolean/string coercion is rejected; zero is allowed only as an explicit per-profile choice, and at least one fitted profile must be regularized. | IMPLEMENTED in spatial Full-EM inverse; audit remaining CTS regularizers |
| TS-14 | Common spectral-response contract | Measured response operators are validated in the principal electrostatic and Full-EM paths and are not silently renormalized. | PARTIAL: cross-path regression parity remains |
| TS-15 | External uncertainty in dashboard | External covariance propagation exists in several APIs and selected dashboard modes. | PARTIAL: complete exposure for every supported input/model remains |
| TS-16 | Descriptive spectrum separated from physical inference | Descriptive spectral moments and physical inference are separate numerical callables. | PARTIAL: registry/dashboard end-to-end separation remains |
| TS-17 | Model-discrepancy uncertainty | Additive PSD model-discrepancy covariance is propagated in selected CTS and Thomson GLS inversions. | PARTIAL: universal propagation to all subanalyses remains |
| TS-18 | Multi-parameter identifiability | Scaled-Jacobian SVD/rank diagnostics exist; profile covariance is withheld when data-only rank is deficient. | PARTIAL: profile likelihood/posterior and practical-identifiability coverage remain |
| TS-19 | Documentation consistency | The scope note and this ledger are synchronized to distinguish implemented kernels from full end-to-end capabilities. | PARTIAL: legacy documentation and all dashboard/registry claims still require a repository-wide pass |
| TS-20 | Cross-subanalysis contract and regression matrix | `tests/test_thomson_analysis1_cross_contracts.py` now checks all four Analysis-1 routes (optical TS, CTS, profile reconstruction, descriptive spectral distribution), distinct executable registry routes, family/diagnostic registry parity, dashboard wiring, public CTS dispatcher shape, and identifiability/covariance output declarations and producers. The registry now declares `parameter_identifiability` for all four routes. | IMPLEMENTED IN TESTS; CI on the exact resulting SHA remains unverified |

## Implementation and release rules

1. Never describe a supplied coefficient, damping rate, Gaunt factor, coupling generator, response matrix or background template as internally inferred physics.
2. Never substitute phenomenological broadening for a missing collision operator, relativistic emissivity, arbitrary-3-D Maxwell solver or general EVDF response.
3. Keep forward-model capability, inversion capability, dashboard exposure and experimental validation distinct.
4. For physical extensions, require limiting-case tests, conservation/normalization, positivity or passivity where applicable, numerical convergence and a documented validity domain.
5. A successful optimizer is not sufficient for a reportable physical measurement. Data-only identifiability, covariance provenance, model discrepancy and external inputs must be reported.
6. CI establishes software checks only. It is not experimental validation.

TS-02 incremental implementation note (2026-11-04): the relativistic ECE absorption entry point is conditional on an explicitly supplied normalized gyrotropic EVDF, density, local wavevector, magnetic field and polarization. It does not infer the EVDF, implement a collisional operator, or close polarized transfer. Resonance-width and quadrature convergence remain numerical release gates.

Analysis 1 must not be advertised as having all TS-01–TS-20 closed until every remaining PARTIAL/OPEN item has code-level implementation, tests, public API/dashboard integration, synchronized documentation, and successful CI on the exact release commit.


TS-01 Full-EM response bridge (2026-11-04): `cts_landau_em.py` projects the same discrete Landau resolvent and its matching FDT source covariance onto the three Cartesian current components, returning a collisional conductivity tensor, dielectric tensor and current-noise spectral matrix for the unmagnetized, zero-flow common-Maxwellian reference domain. It is exposed as `landau_fdt_full_em_response`. This is a genuine kinetic-to-Maxwell constitutive bridge, but it is **not** the magnetized Full-EM CTS three-wave scattering vertex and does not close TS-01. The finite-grid convergence gate remains mandatory.


TS-01 Full-EM public assembly entry point (2026-11-04): added
`analyze_landau_fdt_full_em_constitutive` and dispatcher key
`landau_fdt_full_em_constitutive`. The public route now assembles the same
conserving multispecies Landau operator from velocity axes, species and
equilibrium inputs, then returns its Cartesian conductivity, dielectric and
matching FDT current-noise tensors without requiring callers to manually
construct a `LandauFDTSystem`. Regression coverage checks public dispatch,
tensor shapes, operator provenance and the explicit magnetized-plasma gate.
This improves API integration only: it remains an unmagnetized, zero-flow,
common-temperature Maxwellian constitutive kernel, not a magnetized Full-EM
CTS three-wave vertex or end-to-end measured-spectrum inference. TS-01 remains
PARTIAL until those physical and inference layers are implemented and tested.


TS-20 cross-subanalysis matrix update (2026-11-04): the Analysis-1 registry previously omitted the runtime `parameter_identifiability` output from the optical parent, CTS and descriptive spectral-distribution output contracts (the profile route already declared it). The output is now declared consistently for IDs 1, 30, 46 and 81. Regression coverage checks the four family entry IDs, callable/module-function wiring, unique input/output names, the diagnostic registry's four corresponding records, CTS public dispatcher signature and covariance/identifiability outputs, runtime producers of identifiability metadata, and the Dashboard's CTS selector and descriptive-spectrum display. This is a software-contract correction, not a claim that the descriptive distribution estimates plasma parameters or that any open physical model is complete. The final CI result must still be checked against the exact main SHA.


TS-15 / TS-18 Full-EM uncertainty provenance correction (2026-11-04): the Full-EM identifiability descriptor previously stated unconditionally that external-input uncertainty was not propagated, although the inversion supports supplied local magnetic-field/geometry/polarization covariance and measured-response covariance through perturbation plus full-spectrum refitting. The descriptor is now generated from the actual `external_input_covariance_supplied`, `external_input_covariance_propagated`, `instrument_response_uncertainty_supplied`, and `instrument_response_uncertainty_propagated` result flags. It distinguishes absent covariance (external input held fixed), supplied covariance with insufficient successful refits, and successfully propagated covariance. Nominal magnetic field, composition, geometry and polarization remain external inputs and are not claimed as inferred plasma parameters. Regression tests cover all provenance states and reject impossible propagated-without-supplied states.


TS-10 bounded Kappa API integration (2026-11-04): the collisionless isotropic-Kappa susceptibility kernel is now reachable through `run_advanced_thomson_physics("kappa_electrostatic_susceptibility", ...)`, and its callable is linked from the CTS Analysis-1 and diagnostic registries. The route reports the 1-D marginal scope and does not claim a complete Kappa CTS spectrum or FDT/noise closure. Regression coverage compares direct and dispatched numerical outputs.
