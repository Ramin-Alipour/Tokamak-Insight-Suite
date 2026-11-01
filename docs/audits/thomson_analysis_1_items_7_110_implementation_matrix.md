# Thomson Analysis 1 — Item-by-item implementation matrix (7–110)

**Reviewed branch:** `main`  
**Review date:** 2026-11-01  
**Purpose:** evidence-based implementation triage for the comprehensive audit. This is an inventory, not a claim that every item is closed.  
**Input boundary:** calibrated/prepared measured diagnostic data only. No instrument calibration, raw-data repair, or experimental validation is introduced.

## Status vocabulary

- `implemented_and_tested`: the described capability exists in source and has direct regression coverage.
- `conditional_on_explicit_inputs`: the capability exists only when the listed physical metadata/covariance is supplied.
- `non_identifiable_by_physics`: the measurement/model cannot uniquely determine the requested quantity.
- `outside_selected_model_with_reason`: deliberately not represented by the selected physical model.
- `open_gap`: a computational capability is absent or incomplete; documentation alone does not close it.
- `partially_implemented`: some of the requested operation exists, but the full scope in the audit item does not.

## A. Optical forward models and identifiability

| Item | Current evidence/status | Required action or scientific disposition |
|---:|---|---|
| 7 | `implemented_and_tested` — Maxwellian branch and isotropic non-relativistic assumption in `thomson_inference.py` / `thomson_kinetic_models.py`. | Retain explicit model label and domain warning. |
| 8 | `non_identifiable_by_physics` — only scattering-vector flow projection is fitted. | Preserve scalar projection; no 3-D vector claim from one view. |
| 9 | `implemented_and_tested` — finite-variance isotropic Kappa model. | Retain finite-variance domain and parameterization. |
| 10 | `outside_selected_model_with_reason` — analytic Student-t projection is used, not Skolar generalized plasma-dispersion algorithm. | Do not relabel as generalized plasma-dispersion-function implementation; evaluate that algorithm separately before adoption. |
| 11 | `partially_implemented` — Kappa shape parameter is fitted, but audit requires explicit information/identifiability diagnostics for spectral coverage and shape sensitivity. | Add profile-likelihood/rank/sensitivity test for Kappa index; return unavailable when not constrained. |
| 12 | `implemented_and_tested` — relativistic isotropic Maxwell–Jüttner branch. | Retain explicit rest-frame distribution assumption. |
| 13 | `non_identifiable_by_physics` — relativistic bulk flow restricted to scattering-vector direction. | Do not infer perpendicular/3-D flow from one view. |
| 14 | `outside_selected_model_with_reason` — elastic Thomson kinematics; photon recoil/Compton effects excluded. | Keep model-domain statement and recoil screen. |
| 15 | `implemented_and_tested` — configured recoil error-budget screen. | Describe as numerical tolerance, not universal boundary. |
| 16 | `implemented_and_tested` — conservative non-relativistic theta/flow-beta screens. | Describe as asymptotic software screens, not universal plasma limits. |
| 17 | `outside_selected_model_with_reason` — microwave CTS is a separate subanalysis. | Keep optical and microwave model dispatchers strictly separate. |
| 18 | `non_identifiable_by_physics` — no unrestricted 3-D EVDF from one view. | Preserve explicit limitation. |
| 19 | `non_identifiable_by_physics` — anisotropic optical distribution not inferred. | Requires independent views or a constrained anisotropic model and additional information. |
| 20 | `open_gap` — no supported multi-component optical population decomposition. | Requires separately justified population model, inputs and identifiability tests; do not infer automatically. |
| 21 | `non_identifiable_by_physics` — one geometry cannot identify three independent flow components. | Multi-view joint inversion is conditional on independent geometries. |
| 22 | `conditional_on_explicit_inputs` — normalized spectral shape alone does not identify absolute density. | Density remains unavailable unless absolute calibrated power and complete optical factors are supplied. |

## B. Absolute density, measurement representation and response

| Item | Current evidence/status | Required action or scientific disposition |
|---:|---|---|
| 23 | `conditional_on_explicit_inputs` — absolute density accepts W/nm declaration. | Keep explicit unit declaration and reject ambiguous units. |
| 24 | `conditional_on_explicit_inputs` — incident laser power is an explicit input. | Propagate supplied uncertainty; never infer missing power. |
| 25 | `conditional_on_explicit_inputs` — interaction length is an explicit input. | Propagate supplied uncertainty; never invent path length. |
| 26 | `conditional_on_explicit_inputs` — solid angle and optical efficiency are explicit inputs. | Keep density unavailable if either is absent. |
| 27 | `conditional_on_explicit_inputs` — directions and polarization enter cross-section/density model. | Validate mutually consistent unit vectors and polarization representation. |
| 28 | `outside_selected_model_with_reason` — density relation assumes optically thin single scattering. | Report the assumption; no multiple-scattering solver exists. |
| 29 | `implemented_and_tested` — density is interpreted as interaction-length-averaged effective density. | Preserve locality caveat. |
| 30 | `open_gap` — no spatial integral over varying density, laser intensity and collection efficiency. | Implement only with explicit along-path profiles and quadrature; otherwise report effective path average. |
| 31 | `outside_selected_model_with_reason` — canonical input is channel × wavelength, not a time-resolved movie tensor. | Time-resolved joint inversion is a separate extension, not an implicit reshape. |
| 32 | `outside_selected_model_with_reason` — optical electron spectrum does not infer ion temperature. | Keep Ti outside this optical Thomson output contract. |
| 33 | `implemented_and_tested` — physical model selection is explicit. | Retain no automatic regime switching. |
| 34 | `implemented_and_tested` — supplied response is a fixed forward operator. | No response calibration or measurement inside TIS. |
| 35 | `implemented_and_tested` — kernel/matrix response representations are distinguished. | Retain uniform-grid requirement for index-space 1-D kernels. |
| 36 | `implemented_and_tested` — response matrix validation enforces non-negative normalized redistribution. | Do not silently accept signed/non-normalized operators. |

## C. Optical uncertainty, nuisance parameters and channel coupling

| Item | Current evidence/status | Required action |
|---:|---|---|
| 37 | `partially_implemented` — full channel-major GLS exists for Maxwellian/Kappa, not relativistic Maxwell–Jüttner. | Add relativistic multi-channel GLS only after a stable joint optimizer/covariance implementation; otherwise keep explicit limitation. |
| 38 | `implemented_and_tested` — relativistic branch uses channel-local fit covariance plus supported external/response Monte Carlo. | Do not claim cross-channel spectral GLS. |
| 39 | `conditional_on_explicit_inputs` — cross-channel covariance requires supplied shared covariance. | Keep independent pointwise-noise assumption explicit when covariance absent. |
| 40 | `implemented_and_tested` — unweighted fit is distinguished when noise is absent. | Preserve residual-scaled covariance caveat. |
| 41 | `implemented_and_tested` — local linear background nuisance in Maxwellian/Kappa fits. | Keep nuisance model visible. |
| 42 | `outside_selected_model_with_reason` — structured/curved/time-varying background is not separated by linear baseline. | Do not claim structured-background removal. |
| 43 | `open_gap` — structured background model selection is absent. | Add only with explicit basis/model selection and identifiable constraints. |
| 44 | `partially_implemented` — model-specific external uncertainty representations supported; unsupported combinations rejected. | Complete cross-model input/covariance matrix and API consistency tests. |
| 45 | `implemented_and_tested` — scalar angle and beam-direction uncertainty representations are mutually exclusive. | Retain double-counting guard. |
| 46 | `implemented_and_tested` — shared 2-D and channel-specific 3-D response uncertainty are distinct. | Preserve distinct covariance semantics. |
| 47 | `implemented_and_tested` — response MC is conditional on supplied uncertainty model. | Never present as independent response validation. |
| 48 | `implemented_and_tested` — density covariance combination has explicit zero-power-row/column rules. | Retain double-counting guards. |
| 49 | `outside_selected_model_with_reason` — missing external-factor correlations are not invented. | Request full covariance where correlations matter. |
| 50 | `outside_selected_model_with_reason` — radius/channel coordinate uncertainty not propagated. | Add only as an explicit coordinate-covariance input in a later spatial inverse. |
| 51 | `outside_selected_model_with_reason` — wavelength-axis calibration uncertainty outside agreed input boundary. | Treat calibrated wavelength axis as fixed; distinguish from laser-wavelength uncertainty. |
| 52 | `implemented_and_tested` — local fit covariance is not presented as complete uncertainty budget. | Preserve separation of statistical, systematic and model-discrepancy uncertainty. |

## D. Spectral-distribution subanalysis

| Item | Current evidence/status | Required action |
|---:|---|---|
| 53 | `implemented_and_tested` — reports wavelength moments, not EVDF. | Keep separate from projected velocity inversion. |
| 54 | `implemented_and_tested` — centroid/variance/skewness/kurtosis from calibrated measured spectrum. | Retain explicit background treatment. |
| 55 | `implemented_and_tested` — exactly one background representation required. | Preserve strict validation. |
| 56 | `implemented_and_tested` — edge fraction is explicit heuristic. | Warn that signal can occupy spectral edges. |
| 57 | `open_gap` — fitted-background parameter uncertainty is not jointly sampled with moment MC. | Add optional background covariance/uncertainty model with explicit parameterization and regression tests. |
| 58 | `implemented_and_tested` — full within-channel covariance is accepted and sampled jointly; independent pointwise uncertainty is separately labelled. | Retain covariance route; ensure documentation does not imply independence when full covariance supplied. |
| 59 | `implemented_and_tested` — signed residuals retained; area/variance instability rejected. | Preserve rejection and warnings. |
| 60 | `implemented_and_tested` — observed wavelength width only. | No instrument-corrected temperature width. |
| 61 | `implemented_and_tested` — response matrix used forward in Gaussian fit; no deconvolution. | Preserve forward-only semantics. |
| 62 | `implemented_and_tested` — fit covariance conditional on fixed background/response. | Keep conditional interpretation. |
| 63 | `implemented_and_tested` — no independent Te/ne/flow outputs in descriptive child. | Parent physical workflow remains source of those quantities. |
| 64 | `implemented_and_tested` — Gaussian descriptor is not a Maxwellian proof. | Preserve caveat. |

## E. Spatial profile reconstruction

| Item | Current evidence/status | Required action |
|---:|---|---|
| 65 | `implemented_and_tested` — PCHIP and GCV smoothing spline are numerical representations. | Do not label them plasma forward models. |
| 66 | `implemented_and_tested` — no extrapolation beyond measured interval. | Preserve strict interval restriction. |
| 67 | `implemented_and_tested` — input coordinate is radius, not flux coordinate. | No implicit equilibrium mapping. |
| 68 | `outside_selected_model_with_reason` — no Grad–Shafranov/equilibrium reconstruction. | Requires an external equilibrium solution and separate mapping contract. |
| 69 | `implemented_and_tested` — GCV spline requires at least five channels. | Preserve explicit insufficient-channel status. |
| 70 | `partially_implemented` — full channel covariance is sampled for uncertainty, but spline weighting is diagonal/pointwise rather than a full-covariance GLS objective. | Implement correlated-error spatial objective and test against diagonal limit. |
| 71 | `implemented_and_tested` — invalid positive-profile draws discarded and success threshold reported. | Preserve warnings. |
| 72 | `open_gap` — optional measured PSF/channel mixing operator not included. | Add a supplied fixed spatial mixing operator to a joint spectral-profile forward model; no estimation/calibration by TIS. |
| 73 | `open_gap` — present workflow fits channels then reconstructs; no joint spatially regularized spectral inverse. | WP-P1 core development. |
| 74 | `implemented_and_tested` — PCHIP and GCV are separate representations; smoothing can suppress sharp gradients. | Expose method choice and sensitivity diagnostics. |
| 75 | `outside_selected_model_with_reason` — no boundary/separatrix/flux prior. | Do not invent boundary conditions; conditional only on supplied equilibrium/constraints. |
| 76 | `implemented_and_tested` — profile grid points explicitly interpolated. | Never count grid points as independent observations. |

## F. Electrostatic microwave CTS

| Item | Current evidence/status | Required action |
|---:|---|---|
| 77 | `implemented_and_tested` — dedicated unmagnetized longitudinal electrostatic model. | Reject magnetized regimes requiring full-EM treatment. |
| 78 | `conditional_on_explicit_inputs` — species/charge states/masses/fractions supplied externally. | Keep composition fixed, not inferred. |
| 79 | `non_identifiable_by_physics` — only scattering-wavevector flow projection. | Multi-view geometry required for vector recovery. |
| 80 | `implemented_and_tested` — tabulated fast-ion branch is projected 1-D VDF. | Preserve projection limitation. |
| 81 | `open_gap` — external uncertainty covariance does not cover all cross-parameter correlations. | Add general supplied external covariance with parameter ordering and PSD validation. |
| 82 | `partially_implemented` — response uncertainty supports measured 2-D/3-D forms, not interchangeable 1-D kernel/callable representation. | Complete response covariance contract or explicitly reject unsupported form. |
| 83 | `implemented_and_tested` — global search requires finite supplied bounds. | Never invent scientific bounds. |
| 84 | `conditional_on_explicit_inputs` — tabulated fast-ion inverse requires explicit regularization choice. | Distinguish numerical regularizer from physical parameter. |
| 85 | `implemented_and_tested` — ion-flow parameter is common projected component. | Do not report species-resolved 3-D flows. |

## G. Magnetized full-electromagnetic CTS

| Item | Current evidence/status | Required action |
|---:|---|---|
| 86 | `implemented_and_tested` — explicit B-field, geometry and mode/eigenmode selection; no electrostatic fallback. | Preserve strict dispatcher separation. |
| 87 | `implemented_and_tested` — Bindslev low-temperature Gamma/X closure for thermal bulk-electron source. | Keep closure identity explicit. |
| 88 | `implemented_and_tested` — low-temperature screen is a model-domain diagnostic. | Do not call it universal threshold. |
| 89 | `partially_implemented` — Maxwellian bulk inversion; relativistic gyrotropic forward model is not a fitted parameterized inverse. | Add only a physically consistent inverse with dedicated tests. |
| 90 | `conditional_on_explicit_inputs` — fast-ion VDF shape is fixed supplied input. | No claim of reconstructed fast-ion distribution. |
| 91 | `partially_implemented` — composition fixed; species-resolved thermal ion temperature and projected flow fitted. | Maintain fixed composition and species output mapping. |
| 92 | `implemented_and_tested` — relativistic kinetic vertex is homogeneous/local. | Preserve local-model assumption. |
| 93 | `outside_selected_model_with_reason` — finite-orbit-width absent. | Requires separate orbit-resolved kinetic formulation. |
| 94 | `outside_selected_model_with_reason` — collisional kinetic corrections absent. | Do not reinterpret numerical damping as collision frequency. |
| 95 | `outside_selected_model_with_reason` — spatially varying relativistic response/nonlinear equilibrium absent. | Requires supplied profiles/equilibrium and spatial kinetic solver. |
| 96 | `implemented_and_tested` — cold-plasma geometrical-optics ray tracing only. | No diffraction/quasi-optical claim. |
| 97 | `partially_implemented` — beam covariance supplied at ray samples; no propagated diffraction covariance. | Extend only with explicit paraxial/Gaussian-beam propagation formulation and boundary data. |
| 98 | `open_gap` — mode conversion and branch tracking through degeneracies absent. | Implement eigenmode continuation and crossing/degeneracy diagnostics. |
| 99 | `partially_implemented` — local field/ray may vary; fitted plasma parameters uniform over sampled volume. | Spatially varying plasma response needs explicit profile/equilibrium input and volume quadrature. |
| 100 | `implemented_and_tested` — beam-overlap integrals are relative spectral-bin weights. | Absolute coupling remains conditional on calibrated launcher/receiver data. |
| 101 | `open_gap` — full-EM inversion does not propagate response-matrix uncertainty. | Add response uncertainty propagation through the full-EM forward/inverse model. |
| 102 | `open_gap` — no full external B/geometry/polarization/ray/equilibrium covariance propagation. | Add ordered joint external covariance and Monte Carlo/linear propagation with validated numerical stability. |
| 103 | `implemented_and_tested` — covariance is conditional on fixed external inputs/settings. | Keep conditional uncertainty label. |
| 104 | `implemented_and_tested` — unweighted covariance is residual-scaled, not measurement-noise calibrated. | Preserve semantics. |
| 105 | `implemented_and_tested` — positive log parameters have no default scientific bounds. | Require user-supplied bounds where optimizer needs them. |
| 106 | `partially_implemented` — opt-in paired harmonic/quadrature convergence assessment exists; not automatic across every forward/inversion path. | Add automated declared-tolerance convergence gates for all applicable paths. |
| 107 | `implemented_and_tested` — damping is numerical resonance regularization. | Never report it as physical collision frequency. |
| 108 | `non_identifiable_by_physics` — full-EM flow is scalar scattering-vector projection. | Preserve single-view limitation. |
| 109 | `conditional_on_explicit_inputs` — frequency-resolved geometry must match frequency axis and explicitly select propagation branch. | Reject mismatches; never invent a branch. |
| 110 | `outside_selected_model_with_reason` — no independent experimental validation claim. | Keep all software tests distinct from experimental validation. |

## H. Six-stage implementation execution record

### Stage 1 — Contract and source inventory
Completed for this pass: repository tree, exact source files, current test files, audit items 7–110, Python runtime registry and GitHub workflow definitions were inspected directly on `main`. This matrix records the initial per-item disposition. The open gaps above are not marked complete.

### Stage 2 — Optical and spectral branch
Current baseline has Maxwellian, finite-variance Kappa and relativistic Maxwell–Jüttner forward models; descriptive wavelength moments; separate projected 1-D EVDF inversion; and an explicit refusal to apply the non-relativistic EVDF inverse to the relativistic Maxwell–Jüttner branch. Remaining work includes Kappa identifiability diagnostics, multi-population model review, structured-background treatment and background-parameter uncertainty propagation. No unrestricted 3-D EVDF or anisotropy is claimed.

### Stage 3 — Spatial profile branch
Current baseline fits measured spectra channel-by-channel and then applies PCHIP/GCV profile representations. Full channel covariance is available in uncertainty sampling in supported paths, but the smoothing objective does not yet perform full-covariance GLS and the workflow is not a joint spatially regularized spectral inverse. The measured PSF/mixing operator and equilibrium mapping remain open/conditional.

### Stage 4 — Electrostatic CTS
Current baseline includes the unmagnetized longitudinal dielectric solver, projected flow, supplied ion composition, finite user-supplied global-search bounds and a projected 1-D fast-ion option. Full cross-covariance for all external inputs remains open; do not infer species composition or 3-D flow.

### Stage 5 — Full-EM CTS and literature/model gate
Current baseline includes magnetized kinetic response, Bindslev thermal-electron source closure, relativistic kinetic forward components, cold-plasma ray geometry, local beam overlap, and opt-in harmonic/quadrature convergence assessment. It does not yet provide response/external-input covariance propagation, mode tracking through degeneracies, propagated diffraction/quasi-optical beam physics, or spatially varying full-EM fitted plasma response.

Recent literature reviewed for method selection:
- Foo et al., *Regularized inversion of optical Thomson scattering spectra for improved determination of plasma parameter evolution*, Rev. Sci. Instrum. 97, 103535 (2026), DOI 10.1063/5.0350497. The paper regularizes time/spatially coupled parameter profiles and demonstrates synthetic test cases; it motivates joint inversion/regularization but does not supply a new scattering physical model or experimental validation of TIS.
- Skolar, Longley & Goodwin, *Computing the generalized plasma dispersion function for non-Maxwellian plasmas, with applications to Thomson scattering*, Phys. Plasmas (2025), DOI 10.1063/5.0261966. This is a numerical kinetic-integral method; the existing analytic Kappa projection must not be described as implementing it.
- Flocken, Korsholm & Rasmussen, *Accuracy-driven optimisation procedure for collective Thomson scattering diagnostics for fusion plasmas*, Fusion Eng. Des. (2026), DOI 10.1016/j.fusengdes.2026.115845. It combines ray tracing, forward modelling and Bayesian inversion for diagnostic-configuration optimization; this is a workflow/optimization contribution, not a replacement physical CTS response.
- Zhang et al., *Spectral decomposition for collective Thomson scattering based on an improved genetic algorithm*, Rev. Sci. Instrum. (2024), DOI 10.1063/5.0215734. Algorithmic spectral decomposition does not itself establish a new physical scattering model.
- Baker et al., *Recovering Non-Maxwellian Velocity Distribution Functions from Collective Thomson Scattering Spectra Using Generalized Basis Representations*, arXiv:2610.37516 (2026 preprint). This is a recent research direction based on synthetic PIC-generated spectra; it is not adopted as a production model without peer-reviewed scrutiny, model-equation audit and independent limiting-case regression tests.

**Model decision:** no new physical model is added merely because a recent paper proposes an optimizer, regularizer or flexible basis. The existing physical shortlist is retained until the full mathematical response, parameter identifiability, uncertainty contract and analytic limiting cases are implemented and tested.

### Stage 6 — Integration and verification gate
The exact root/package diagnostic registry copies, Python runtime registry, dashboard, documentation and tests must be synchronized for every new input/output. The present inventory identifies the integration work; it does not certify closure. GitHub Actions runs associated with the preceding plan commit were queued when inspected. A green result must be checked against the exact resulting commit SHA before claiming software-test success.

## I. Immediate implementation order (no scope expansion)

1. WP-P1 joint multi-channel spectral inversion with full cross-channel covariance and optional supplied spatial mixing operator.
2. WP-O2 background nuisance covariance propagation and explicit Kappa shape identifiability diagnostics.
3. WP-C1 full external covariance support for electrostatic CTS.
4. WP-C2 full-EM response/external covariance propagation, then mode tracking and automatic numerical convergence gates.
5. Only after these, consider broader anisotropic/multi-population EVDF models with a defensible identifiability and uncertainty analysis.

**Completion rule:** an item is closed only when code, API contract, both JSON registry copies, Python registry, dashboard, documentation and direct regression tests agree. No experimental validation is claimed.
