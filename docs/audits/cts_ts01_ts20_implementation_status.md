# CTS TS-01–TS-20 remediation ledger

**Target:** `main`  
**Audit basis:** source and public entry points currently present on `main`.  
**Rule:** a kernel, a passing unit test, a dashboard control, and a complete physical inference path are different deliverables. Status below is deliberately scoped; no item is called complete outside its stated validity domain.

| ID | Required capability / contract | Verified implementation on main | Status |
|---|---|---|---|
| TS-01 | Collisional CTS with fluctuation-dissipation consistency | `cts_landau_fdt.py` assembles a finite-box 3-D multispecies linearized Landau weak-form operator with test/field-particle terms, discrete number/momentum/energy invariants and matching classical FDT noise. `analyze_landau_fdt_cts` adds self-consistent electrostatic Poisson screening and CTS dynamic-structure-factor output. Common-temperature/common-flow isotropic Maxwellians; non-relativistic finite-box reference discretization. | IMPLEMENTED WITH DECLARED DOMAIN; Full-EM coupling and convergence gate remain |
| TS-02 | Independent ECE absorption solver | `ece_kinetic.py` computes magnetized non-relativistic Maxwellian Vlasov conductivity and absorption normalized to supplied modal Poynting flux. `ece_kinetic_modes.py` supplies a two-mode opacity and ordered LTE transfer path. | PARTIAL: relativistic/collisional response and self-consistent ray/eigenmode coupling remain |
| TS-03 | Relativistic and magnetized bremsstrahlung | `cts_em_background.py` supplies non-relativistic unmagnetized Maxwellian free-free emissivity with explicit Gaunt factor. | OPEN for relativistic and magnetized-mode emissivity/opacity |
| TS-04 | Full-vector non-paraxial Full-EM propagation | Exact vector angular-spectrum propagation exists for homogeneous isotropic segments; oblique Berreman and the stratified beam wrapper retain anisotropy, diffraction, longitudinal fields and coherent polarization. | PARTIAL: arbitrary 3-D inhomogeneous anisotropic media and dissipative interfaces remain |
| TS-05 | Mode conversion | Coherent Jones reflection/transmission and two-mode opacity/coupling are available for declared stratified boundaries and supplied coupling generators. | PARTIAL: robust branch tracking through degeneracies and arbitrary 3-D conversion remain |
| TS-06 | General relativistic anisotropic EVDF | Selected constrained gyrotropic/drifted distributions and parametric fits exist. | PARTIAL: unrestricted/general EVDF inversion and complete Full-EM coupling remain |
| TS-07 | Shared physical background across all optical paths | Physical and detector-space background modules exist, with several optical workflow integrations. | PARTIAL: universal path parity and end-to-end dashboard verification remain |
| TS-08 | Full CTS geometry/profile covariance | External geometry uncertainty, profile covariance and selected cross-covariance paths exist. | PARTIAL: one joint covariance contract across all ray, beam, profile and inversion inputs remains |
| TS-09 | General multi-population plasma model | N-population relativistic Maxwell-Jüttner optical forward model and selected CTS multi-ion/fast-ion paths exist. | PARTIAL: a general coupled population model across all CTS/Full-EM paths remains |
| TS-10 | General kinetic Kappa response | Parameterized Kappa optical models and restricted generalized-dispersion utilities exist. | OPEN for a general Kappa susceptibility, collision/noise and kinetic response with documented validity |
| TS-11 | Explicit zero-background state | CTS spatial inversion requires explicit `background_mode='zero'` or `'spectrum'`; background integrator reports exact zero only when plasma and incident radiance are zero. Other paths retain legacy/optional background contracts. | PARTIAL |
| TS-12 | Explicit CTS profile support | Full-EM spatial inversion requires an explicit positive `profile_bounds_relative`; no scientific range is silently selected. | IMPLEMENTED in spatial Full-EM inverse; audit remaining CTS profile paths |
| TS-13 | Explicit regularization strength | Spatial Full-EM inverse requires explicit per-fit regularization strength and reports its scope separately from measurement covariance. Boolean/string coercion is rejected; zero is allowed only as an explicit per-profile choice, and at least one fitted profile must be regularized. | IMPLEMENTED in spatial Full-EM inverse; audit remaining CTS regularizers |
| TS-14 | Common spectral-response contract | Measured response operators are validated in the principal electrostatic and Full-EM paths and are not silently renormalized. | PARTIAL: cross-path regression parity remains |
| TS-15 | External uncertainty in dashboard | External covariance propagation exists in several APIs and selected dashboard modes. | PARTIAL: complete exposure for every supported input/model remains |
| TS-16 | Descriptive spectrum separated from physical inference | Descriptive spectral moments and physical inference are separate numerical callables. | PARTIAL: registry/dashboard end-to-end separation remains |
| TS-17 | Model-discrepancy uncertainty | Additive PSD model-discrepancy covariance is propagated in selected CTS and Thomson GLS inversions. | PARTIAL: universal propagation to all subanalyses remains |
| TS-18 | Multi-parameter identifiability | Scaled-Jacobian SVD/rank diagnostics exist; profile covariance is withheld when data-only rank is deficient. | PARTIAL: profile likelihood/posterior and practical-identifiability coverage remain |
| TS-19 | Documentation consistency | The scope note and this ledger are synchronized to distinguish implemented kernels from full end-to-end capabilities. | PARTIAL: legacy documentation and all dashboard/registry claims still require a repository-wide pass |
| TS-20 | CI on exact final commit | GitHub Actions runs are triggered by pushes to `main`; success must be checked against the exact resulting SHA. | NOT VERIFIED until runs complete |

## Implementation and release rules

1. Never describe a supplied coefficient, damping rate, Gaunt factor, coupling generator, response matrix or background template as internally inferred physics.
2. Never substitute phenomenological broadening for a missing collision operator, relativistic emissivity, arbitrary-3-D Maxwell solver or general EVDF response.
3. Keep forward-model capability, inversion capability, dashboard exposure and experimental validation distinct.
4. For physical extensions, require limiting-case tests, conservation/normalization, positivity or passivity where applicable, numerical convergence and a documented validity domain.
5. A successful optimizer is not sufficient for a reportable physical measurement. Data-only identifiability, covariance provenance, model discrepancy and external inputs must be reported.
6. CI establishes software checks only. It is not experimental validation.

Analysis 1 must not be advertised as having all TS-01–TS-20 closed until every remaining PARTIAL/OPEN item has code-level implementation, tests, public API/dashboard integration, synchronized documentation, and successful CI on the exact release commit.
