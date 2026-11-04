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

TS-03 regime-diagnostic correction (2026-11-04): the thermal free-free transfer result now reports the path-resolved dimensionless electron temperature parameter kT/(m_e c^2) and frequency-resolved photon parameter h nu/(m_e c^2). These diagnostics make the non-relativistic model scope auditable and explicitly report that no relativistic correction has been applied. They do not close TS-03: a validated relativistic and magnetized-mode emissivity/opacity kernel, limiting-case tests, and end-to-end CTS integration remain required.

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


TS-15 / TS-18 ray-geometry provenance correction (2026-11-04): the full-forward geometry/equilibrium/beam Monte-Carlo refit already added its sample covariance to the CTS fit covariance, but the nested `parameter_identifiability` descriptor was left at the baseline status and did not report that contribution. The geometry-uncertainty wrapper now adds or replaces a `ray_geometry_uncertainty_contribution` conditional record on both insufficient-refit and successful-refit paths, including requested/successful draw counts, and synchronizes the stated uncertainty scope. This keeps nominal magnetic field and geometry as external inputs while accurately reporting their propagated uncertainty.


TS-07 / TS-11 ECE source-function contract correction (2026-11-04): the CTS microwave-background transfer previously treated every supplied ECE absorption coefficient as if Kirchhoff LTE applied, which is not valid for a general kinetic/non-Maxwellian ECE response. The transfer now requires either an explicit path/frequency/mode ECE emissivity or the explicit boolean ece_lte=True to select the Planck source function. The Full-EM physical-background adapter passes these inputs without guessing. Non-LTE emissivity and LTE source paths have regression coverage. This closes the silent-LTE-assumption defect in this transfer path only; it does not establish universal background parity across all CTS/optical paths, nor a relativistic/magnetized free-free model.


TS-06 / TS-10 scalar-input contract correction (2026-11-04): the collisionless isotropic-Kappa susceptibility and relativistic gyrotropic CTS momentum-grid builder now reject Boolean, string and complex physical scalars before numeric coercion. Kappa also validates phase velocity and pole displacement as real finite scalars and records its non-relativistic finite-support scope. Regression tests cover invalid scalar coercions and the declared Kappa limitations. This is an API/numerical-contract hardening, not completion of a general relativistic EVDF inversion, conserving Kappa collision operator, magnetized Kappa dielectric, or Full-EM Kappa closure.


TS-02 kinetic-ECE physical-input contract hardening (2026-11-04): both non-relativistic Maxwellian and relativistic gyrotropic ECE absorption entry points now reject Boolean, string and complex coercion for frequency, density, temperature/resonance-width and real geometry vectors before numerical evaluation. Complex polarization remains supported as required for electromagnetic eigenmodes. Added parameterized regressions in `tests/test_ece_kinetic_input_contract.py`. This closes an input-contract defect only; it does not implement the still-open conserving collisional response, self-consistent mode/ray coupling or relativistic polarized transfer.



TS-04 / TS-05 cold-plasma magnetoionic kernel (2026-11-04): added
`cts_magnetoionic.py` with the multispecies Stix relations
`R=1-Σω_ps²/[ω(ω+Ω_s)]`,
`L=1-Σω_ps²/[ω(ω-Ω_s)]`,
`S=(R+L)/2`, `D=(R-L)/2`, and
`P=1-Σω_ps²/ω²`, using signed species gyrofrequencies
`Ω_s=q_s|B|/m_s`. The field-aligned dielectric tensor is rotated into the
laboratory basis defined by the supplied magnetic-field vector. The local
magnetoionic solver obtains the two roots of
`det[ε_r - n²(I- k_hat k_hatᵀ)]=0` and returns their complex refractive
indices and electric polarization eigenvectors. Both functions are exposed
through `run_advanced_thomson_physics`; tests cover the unmagnetized isotropic
limit, parallel R/L branches, the perpendicular ordinary P branch, invalid
inputs, and explicit cyclotron-pole rejection. This is a new cold, collisionless,
homogeneous local dispersion model. It is not warm kinetic dielectric response,
a Full-EM CTS three-wave vertex, ray tracing, branch tracking through
degeneracies, or distributed mode conversion. TS-04/05 remain PARTIAL.


TS-03 background composition relations (2026-11-04): added
`plasma_charge_state_moments` and included its path-resolved output in the
CTS microwave-background result. For supplied ion populations it reports
`n_{e,quasi}=Σ Z_i n_i`, `Z_eff=Σ Z_i² n_i/n_e`, and
`ρ_q=e(Σ Z_i n_i-n_e)`, plus a dimensionless charge-balance residual.
The calculation never forces quasineutrality or repairs supplied densities;
`Z_eff` is explicitly unavailable at zero electron density. The helper is
available through the advanced physics dispatcher and has neutral, non-neutral
and vacuum regression cases. This improves mixture bookkeeping and exposes
inconsistent inputs; it is not a relativistic/magnetized free-free emissivity
or opacity correction. TS-03 remains OPEN for those kernels.


Full-EM real-frequency preservation correction (2026-11-04): the CTS kernel
previously replaced every measured angular-frequency shift satisfying
`|ω| < damping` with `sign(ω) damping` before assembling the Maxwell wave
tensor. This incorrectly moved the real spectral frequency and changed the
physical wave operator; resonance broadening is an imaginary denominator
regularization and must not be used as a real-frequency floor. The kernel now
uses the measured `ω=2π Δf` unchanged and reserves damping for kinetic
resonance denominators. An exactly zero shift is rejected with an explicit
static-Maxwell-limit message because that limit has not yet been separately
derived and converged. Added a regression test for this boundary and explicit
result metadata stating that real-frequency replacement is not performed.
This is a physical numerical correction; near-zero nonzero bins remain at
their measured frequencies. The exact-zero static limit remains OPEN.


Electrostatic CTS tabulated-fast-ion boundary correction (2026-11-04):
`cts_analysis._component_tabulated` evaluates the Landau susceptibility for a
finite tabulated 1-D ion VDF. The previous integral included only the
piecewise-interior derivative and omitted the distributional endpoint jumps
when the finite-support VDF was extended by zero outside its grid. The
susceptibility now includes
`f(v_min)/(v_min-ζ) - f(v_max)/(v_max-ζ)` in addition to the interior
principal-value integral and the retarded Landau residue
`+iπ f'(v_res)` when the resonance lies inside the support. Added an
independent analytic regression using a constant finite-support VDF, for
which the interior derivative vanishes and the entire response comes from
the two endpoint terms. This corrects the finite-support electrostatic CTS
kinetic response; it does not extend the model to a 3-D fast-ion VDF.


TS-04 local 3-D magnetoionic ray-velocity relation (2026-11-04): added
`cold_plasma_mode_group_velocity`. For the local Maxwell dispersion matrix
`M(ω,k)=ε_r(ω)−(c²/ω²)(k²I−kkᵀ)`, a simple mode eigenvector `e` gives
`v_g,j=−(e†(∂M/∂k_j)e)/(e†(∂M/∂ω)e)`. The implementation evaluates the
frequency derivative by symmetric finite difference and the wavevector
derivative analytically, then returns the full Cartesian group-velocity
vector, ray direction and wave-normal/ray walk-off angle. It is exposed
through `run_advanced_thomson_physics("cold_plasma_mode_group_velocity")`
and registered in both formula registries. Tests cover the parallel
magnetoionic branch against an independent finite-difference dispersion
derivative, oblique walk-off, collision rejection and degenerate-mode
rejection. This is a genuine additional 3-D physical relation, but remains a
local homogeneous cold collisionless result: it is not spatial ray tracing,
branch tracking or mode conversion. TS-04/05 remain PARTIAL.


TS-04 Brillouin energy transport relation (2026-11-04): the local
magnetoionic group-velocity kernel now also evaluates the dispersive wave
energy density
`W=1/4[ε₀ E†∂(ωε_r)/∂ω E+μ₀ H†H]`, the time-averaged Poynting vector
`S=1/2 Re(E×H*)`, and energy velocity `v_energy=S/W`. The result reports
the relative difference between energy velocity and the Hellmann–Feynman
group velocity as a numerical/physical consistency diagnostic. This relation
is restricted to the same transparent, collisionless, simple local mode;
it is not applied to absorbing or degenerate branches. Regression coverage
checks positive Brillouin energy and agreement of energy and group velocity
for a parallel magnetoionic branch. The new relation strengthens the local
3-D propagation physics but does not close the spatial ray-tracing or mode
conversion requirements.


TS-10 finite-support Landau-integral correction (2026-11-04): `generalized_landau_integral` now includes the distributional endpoint terms for a tabulated projected VDF interpreted as zero outside its finite velocity interval:
`f(v_min)/(v_min-ζ)^p - f(v_max)/(v_max-ζ)^p`.
The returned result separates the interior piecewise-linear derivative integral from the finite-support boundary contribution and declares the zero-extension convention. A constant top-hat regression case verifies that the complete derivative response is non-zero and equals the analytic endpoint expression. This corrects the finite-support susceptibility primitive; it does not close TS-10's missing conserving collision, fluctuation/noise, magnetized, relativistic-Kappa or Full-EM response models. The test is committed but has not been executed here; CI on the final main SHA remains required.


TS-10 full-axis Kappa Landau response correction (2026-11-04): the collective Kappa susceptibility no longer truncates the principal-value integral at a fixed number of thermal speeds. It now evaluates the symmetric full-axis identity
`PV ∫ f'(v)/(v-vφ) dv = ∫₀∞ [f'(vφ+x)-f'(vφ-x)]/x dx`
with the retarded Landau term `+iπf'(vφ)`. The infinite interval is mapped to a finite quadrature coordinate. Quadrature order is increased from 384 to 768/1536 as κ approaches the finite-variance boundary κ=3/2; a regression test at κ=1.5001 compares the result with independent adaptive infinite-domain quadrature. The same kernel is called by `collective_kappa_spectral_density` and therefore its existing CTS Kappa fit/dispatcher route. This is a physical numerical-kernel correction, not completion of the broader TS-10 scope: conserving collisional Kappa response, FDT/noise closure, magnetized and relativistic Kappa dielectric response, and Full-EM coupling remain open. Test execution and CI on the final SHA are not claimed.
