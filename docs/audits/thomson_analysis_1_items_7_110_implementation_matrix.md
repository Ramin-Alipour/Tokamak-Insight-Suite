# Thomson Analysis 1 — Item-by-item implementation matrix (7–110)

**Reviewed branch:** `main`  
**Review date:** 2026-11-02  
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
| 11 | `partially_implemented` — response-convolved finite-difference Jacobian rank/conditioning and nuisance-projected Kappa sensitivity are combined with a profile-likelihood scan that re-optimizes the five nuisance parameters at fixed Kappa. | The interval is conditional on the selected model/noise representation; grid-censored limits remain explicitly censored, and broader coverage/regularization sensitivity assessment remains open. Do not treat an optimizer candidate as identified when local or profile diagnostics fail. |
| 12 | `implemented_and_tested` — relativistic isotropic Maxwell–Jüttner branch. | Retain explicit rest-frame distribution assumption. |
| 13 | `non_identifiable_by_physics` — relativistic bulk flow restricted to scattering-vector direction. | Do not infer perpendicular/3-D flow from one view. |
| 14 | `outside_selected_model_with_reason` — elastic Thomson kinematics; photon recoil/Compton effects excluded. | Keep model-domain statement and recoil screen. |
| 15 | `implemented_and_tested` — configured recoil error-budget screen. | Describe as numerical tolerance, not universal boundary. |
| 16 | `implemented_and_tested` — conservative non-relativistic theta/flow-beta screens. | Describe as asymptotic software screens, not universal plasma limits. |
| 17 | `outside_selected_model_with_reason` — microwave CTS is a separate subanalysis. | Keep optical and microwave model dispatchers strictly separate. |
| 18 | `non_identifiable_by_physics` — no unrestricted 3-D EVDF from one view. | Preserve explicit limitation. |
| 19 | `non_identifiable_by_physics` — anisotropic optical distribution not inferred. | Requires independent views or a constrained anisotropic model and additional information. |
| 20 | `implemented_and_tested` within a deliberately limited parametric scope — `tokamak_analysis.thomson_population_mixture` implements and tests a two-population isotropic relativistic Maxwell–Jüttner optical spectrum and fit; the dashboard exposes the dedicated subanalysis and the Analysis 1 workflow registry lists its callable. | Keep the two-component assumption explicit; do not infer an unrestricted or arbitrary multi-population EVDF. General K-component decomposition, population-specific flows, and full cross-channel joint covariance remain outside this implementation. |
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
| 30 | `implemented_and_tested` — `infer_thomson_path_averaged_density` evaluates the path-integrated single-scattering kernel with explicit path-dependent incident power, differential cross-section, collection solid angle and optical efficiency; it reports only the identifiable weighted path average. | Arbitrary pointwise density profiles remain non-identifiable from one integrated power measurement; multi-view/profile inversion requires independent spatial measurements and a declared response model. |
| 31 | `outside_selected_model_with_reason` — canonical input is channel × wavelength, not a time-resolved movie tensor. | Time-resolved joint inversion is a separate extension, not an implicit reshape. |
| 32 | `outside_selected_model_with_reason` — optical electron spectrum does not infer ion temperature. | Keep Ti outside this optical Thomson output contract. |
| 33 | `implemented_and_tested` — physical model selection is explicit. | Retain no automatic regime switching. |
| 34 | `implemented_and_tested` — supplied response is a fixed forward operator. | No response calibration or measurement inside TIS. |
| 35 | `implemented_and_tested` — kernel/matrix response representations are distinguished. | Retain uniform-grid requirement for index-space 1-D kernels. |
| 36 | `implemented_and_tested` — response matrix validation enforces finite, non-negative, unit-sum columns and preserves the supplied calibrated operator unchanged; no internal renormalization is performed. | Reject invalid operators; never alter the researcher-supplied calibrated response. |

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
| 57 | `implemented_and_tested` — explicit measured background spectra support pointwise or full wavelength covariance; optional `signal_background_cross_covariance` propagates signal/background correlation through a positive-definite joint block covariance and correlated Monte Carlo draws. | Preserve strict joint-covariance positive-definiteness checks; background-mask and edge-fitted nuisance parameters retain their separately documented conditional limitations. |
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
| 70 | `implemented_and_tested` — supplied positive-definite full cross-channel covariance enters a GLS natural-cubic smoothing-spline objective; GCV uses the GLS hat-matrix effective degrees of freedom. | Singular covariance is explicitly rejected rather than silently regularized; pointwise-only uncertainty retains the diagonal independent-error route. |
| 71 | `implemented_and_tested` — invalid positive-profile draws discarded and success threshold reported. | Preserve warnings. |
| 72 | `implemented_and_tested` within declared scope — fixed spatial mixing is applied inside the wavelength-resolved Maxwellian forward model; the dashboard now exposes the fixed matrix upload and regularization control. | The UI treats the matrix as researcher-supplied and fixed; no PSF estimation/calibration is performed. Kappa/relativistic joint spatial inverses remain outside this selected joint model. |
| 73 | `implemented_and_tested` for the declared Maxwellian joint route — workflow dispatcher, canonical diagnostic contract, dashboard controls, fixed spatial/spectral response operators, full channel-major covariance or pointwise uncertainty, detector-local linear backgrounds, and data-only Jacobian identifiability are connected. | Kappa/relativistic joint spatial inversion and additional response/nonuniform-grid sensitivity cases remain separate model extensions; regularization is never counted as independent measured information. |
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
| 81 | `implemented_and_tested` — electrostatic CTS accepts a researcher-supplied ordered positive-semidefinite joint covariance over probe frequency, scalar scattering angle and individual ion fractions; the dashboard now exposes the covariance upload and explicit parameter order. | Preserve the mutually exclusive scalar-uncertainty route and covariance validation; Full-EM external covariance is tracked separately under item 102. |
| 82 | `partially_implemented` — measured 1-D kernel uncertainty is now sampled and propagated for the three non-collective optical fits; shared 2-D and channel-specific 3-D matrix uncertainty remain supported. | Callable-response uncertainty is explicitly unsupported because no covariance representation for an arbitrary callable is defined; do not silently synthesize one. |
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
| 98 | `partially_implemented` — `track_cold_plasma_modes_from_operators` now follows polarization branches by maximum-overlap assignment and phase alignment, and flags degenerate subspaces. | Coupled-mode conversion dynamics are still outside this helper; a degenerate subspace is reported as ambiguous rather than assigning a fictitious unique eigenvector. |
| 99 | `partially_implemented` — local field/ray may vary; fitted plasma parameters uniform over sampled volume. | Spatially varying plasma response needs explicit profile/equilibrium input and volume quadrature. |
| 100 | `implemented_and_tested` — beam-overlap integrals are relative spectral-bin weights. | Absolute coupling remains conditional on calibrated launcher/receiver data. |
| 101 | `partially_implemented` — Full-EM CTS now accepts a supplied non-negative response-matrix uncertainty, samples column-renormalized response perturbations and repeats the complete parameter inversion; successful draw covariance is added to the conditional fit covariance. | Complete covariance cross terms with other uncertainty sources and add response-shape/strong-uncertainty convergence tests; this is Monte Carlo propagation, not response calibration. |
| 102 | `partially_implemented` — Full-EM measured-spectrum inversion now accepts an ordered joint covariance over local magnetic-field, incident/scattered direction and complex polarization components and propagates it through reproducible Monte Carlo full-parameter refits. | Frequency-resolved ray/equilibrium/beam covariance, composition covariance and cross-covariance between distinct external sources remain open; the local-vector contract rejects ray-resolved geometry rather than silently ignoring its uncertainty. |
| 103 | `implemented_and_tested` — covariance is conditional on fixed external inputs/settings. | Keep conditional uncertainty label. |
| 104 | `implemented_and_tested` — unweighted covariance is residual-scaled, not measurement-noise calibrated. | Preserve semantics. |
| 105 | `implemented_and_tested` — positive log parameters have no default scientific bounds. | Require user-supplied bounds where optimizer needs them. |
| 106 | `implemented_and_tested` for the Full-EM measured-spectrum inversion — paired gyro-harmonic/momentum-quadrature refinement is default-enabled, supports explicit increasing refinement sequences and relative L2/L-infinity tolerances, and reports convergence fields; Python registry, both JSON registries, dashboard and regression tests are synchronized. | This is a fixed-parameter numerical discretization assessment, not physical uncertainty and not a universal convergence guarantee for every standalone forward-model call. |
| 107 | `implemented_and_tested` — damping is numerical resonance regularization. | Never report it as physical collision frequency. |
| 108 | `non_identifiable_by_physics` — full-EM flow is scalar scattering-vector projection. | Preserve single-view limitation. |
| 109 | `conditional_on_explicit_inputs` — frequency-resolved geometry must match frequency axis and explicitly select propagation branch. | Reject mismatches; never invent a branch. |
| 110 | `outside_selected_model_with_reason` — no independent experimental validation claim. | Keep all software tests distinct from experimental validation. |

## H. Six-stage implementation execution record

### Stage 1 — Contract and source inventory
Completed for this pass: repository tree, exact source files, current test files, audit items 7–110, Python runtime registry and GitHub workflow definitions were inspected directly on `main`. This matrix records the initial per-item disposition. The open gaps above are not marked complete.

### Stage 2 — Optical and spectral branch
Current baseline has Maxwellian, finite-variance Kappa and relativistic Maxwell–Jüttner forward models; descriptive wavelength moments; separate projected 1-D EVDF inversion; and an explicit refusal to apply the non-relativistic EVDF inverse to the relativistic Maxwell–Jüttner branch. Remaining work includes broader Kappa coverage, structured-background treatment in the optical single-channel physical fits, background-parameter uncertainty propagation, and a broader multi-population model review. The existing two-population relativistic Maxwell–Jüttner subanalysis is implemented and tested, but is not a general K-component population solver. Projected-EVDF regularization sensitivity is now quantified over the successful candidates in the existing GCV grid, with changes in velocity moments and total-variation distance from the selected solution; explicit single-value regularization is clearly labelled as a single-candidate case. The Kappa fit includes local Jacobian identifiability and a conditional nuisance-profile likelihood scan that re-optimizes five nuisance parameters at each fixed Kappa index; finite-grid-censored limits remain explicitly censored. No unrestricted 3-D EVDF or anisotropy is claimed.

### Stage 3 — Spatial profile branch
Channel-wise physical fits and PCHIP/GCV profile representations remain the default profile workflow. Full-covariance GLS smoothing is implemented for supplied positive-definite cross-channel covariance. A separate wavelength-resolved joint spatial-spectral Maxwellian inverse is implemented with a researcher-supplied fixed spatial mixing operator, spectral response, channel-major covariance or pointwise uncertainty, detector-local linear backgrounds and user-selected curvature regularization. The primary workflow wrapper, Python/JSON registry contracts and dashboard now expose the Maxwellian joint route, including a fixed spatial-mixing matrix upload and explicit numerical regularization. The dashboard uses a shared profile result schema; density is explicitly unavailable in this shape-only joint route. Kappa/relativistic joint inverses and broader identifiability/regularization-sensitivity coverage remain open. The mixing operator is supplied, never estimated or calibrated; equilibrium mapping remains conditional on supplied metadata.

### Stage 4 — Electrostatic CTS
Current baseline includes the unmagnetized longitudinal dielectric solver, projected flow, supplied ion composition, finite user-supplied global-search bounds, a projected 1-D fast-ion option, and correlated Monte Carlo propagation for the declared joint covariance over probe frequency, scalar scattering angle and ion fractions. The electrostatic CTS reference-PCA detector-space nuisance model now reports conditional covariance for its fitted background coefficients and the corresponding frequency-resolved background spectrum, using supplied spectral uncertainty or residual-scaled covariance when no measurement uncertainty is supplied. The adaptive genetic optimizer now validates its controls, rejects all-nonfinite objective populations, preserves the best candidate and reports monotone best-so-far history including initial and final populations. These additions do not propagate uncertainty in the reference PCA basis itself and do not reproduce the paper-specific Zhang et al. fitness function. The dashboard exposes both scalar uncertainty and joint covariance routes; do not infer species composition or 3-D flow.

### Stage 5 — Full-EM CTS and literature/model gate
Current baseline includes magnetized kinetic response, Bindslev thermal-electron source closure, relativistic kinetic forward components, cold-plasma ray geometry, local beam overlap, default-enabled paired harmonic/quadrature convergence assessment in the Full-EM measured-spectrum inversion, and Monte Carlo propagation for supplied non-negative response-matrix uncertainty. The Full-EM dashboard now exposes the response-matrix uncertainty input and refit count. Remaining gaps include full joint covariance/cross-terms for external magnetic-field, geometry, polarization, ray and equilibrium inputs; frequency-resolved mode tracking through degeneracies; propagated diffraction/quasi-optical beam physics; spatially varying full-EM fitted plasma response; and automatic convergence gates across every applicable forward/inversion path.

Recent literature reviewed for method selection:
- Foo et al., *Regularized inversion of optical Thomson scattering spectra for improved determination of plasma parameter evolution*, Rev. Sci. Instrum. 97, 103535 (2026), DOI 10.1063/5.0350497. The paper regularizes time/spatially coupled parameter profiles and demonstrates synthetic test cases; it motivates joint inversion/regularization but does not supply a new scattering physical model or experimental validation of TIS.
- Carole et al., *A photonic model of the High-Resolution Thomson Scattering diagnostic of WEST tokamak*, JINST 20 (2025), P07014, DOI 10.1088/1748-0221/20/07/P07014. Recent peer-reviewed diagnostic modelling reference for the Maxwellian optical Thomson regime.
- Skolar, Longley & Goodwin, *Computing the generalized plasma dispersion function for non-Maxwellian plasmas, with applications to Thomson scattering*, Phys. Plasmas 32 (2025), 043907, DOI 10.1063/5.0261966. This is a numerical kinetic-integral method; the existing analytic Kappa projection must not be described as implementing it.
- Sakai et al., *Monte Carlo simulation method for incoherent Thomson scattering spectra from arbitrary electron distribution functions*, JINST 21 (2026), C01002, DOI 10.1088/1748-0221/21/01/C01002. Recent peer-reviewed forward-modelling reference covering relativistic Maxwellian and Kappa distributions; it does not validate the TIS implementation.
- Flocken, Korsholm & Rasmussen, *Accuracy-driven optimisation procedure for collective Thomson scattering diagnostics for fusion plasmas*, Fusion Eng. Des. (2026), DOI 10.1016/j.fusengdes.2026.115845. It combines ray tracing, forward modelling and Bayesian inversion for diagnostic-configuration optimization; this is a workflow/optimization contribution, not a replacement physical CTS response.
- Zhang et al., *Spectral decomposition for collective Thomson scattering based on an improved genetic algorithm*, Rev. Sci. Instrum. (2024), DOI 10.1063/5.0215734. Algorithmic spectral decomposition does not itself establish a new physical scattering model.
- Baker et al., *Recovering Non-Maxwellian Velocity Distribution Functions from Collective Thomson Scattering Spectra Using Generalized Basis Representations*, arXiv:2610.37516 (2026 preprint). This is a recent research direction based on synthetic PIC-generated spectra; it is not adopted as a production model without peer-reviewed scrutiny, model-equation audit and independent limiting-case regression tests.

**Model decision:** no new physical model is added merely because a recent paper proposes an optimizer, regularizer or flexible basis. The existing physical shortlist is retained until the full mathematical response, parameter identifiability, uncertainty contract and analytic limiting cases are implemented and tested.

### Stage 6 — Integration and verification gate
The exact root/package diagnostic registry copies, Python runtime registry, dashboard, documentation and tests must be synchronized for every new input/output. Full-EM paired harmonic/quadrature convergence controls and result fields are now synchronized across those surfaces and covered by regression tests (item 106, within the Full-EM measured-spectrum inversion scope). The present inventory still does not certify closure of the remaining open/partial items. CTS background-coefficient covariance and genetic-search safeguards were added on 2026-11-02 with direct regression tests; their CI status must be checked against the exact resulting commit SHA before claiming software-test success.

## I. Immediate implementation order (no scope expansion)

1. WP-P1 joint multi-channel spectral inversion with full cross-channel covariance and optional supplied spatial mixing operator.
2. WP-O2 background nuisance covariance propagation and explicit Kappa shape identifiability diagnostics.
3. WP-C1 full external covariance support for electrostatic CTS.
4. WP-C2 full-EM response/external covariance propagation, then mode tracking and automatic numerical convergence gates.
5. Only after these, consider broader anisotropic/multi-population EVDF models with a defensible identifiability and uncertainty analysis.

**Completion rule:** an item is closed only when code, API contract, both JSON registry copies, Python registry, dashboard, documentation and direct regression tests agree. No experimental validation is claimed.


## I. Follow-up registry consistency correction (2026-11-02)

A direct main-branch comparison found that the packaged Thomson registry omitted the already-declared `result_classification` output from the root registry. The packaged copy has now been synchronized with the root copy; the existing root/package equality regression test covers the correction. This is a metadata consistency fix only and does not add physical assumptions or claim experimental validation.


## J. EVDF parametric-comparison likelihood correction (2026-11-02)

The projected-EVDF parametric comparison now uses the Gaussian information-criterion likelihood consistent with the supplied noise contract:

- When pointwise uncertainty or full spectral covariance is supplied and whitened, the reported criteria are (AIC=\chi^2+2k) and (BIC=\chi^2+k\ln n), up to parameter-independent constants.
- When no measurement uncertainty is supplied, the residual variance is estimated from the unweighted fit and the criteria retain the (n\ln(RSS/n)) form with the AIC/BIC penalties.
- A parametric candidate whose nonlinear optimizer did not converge is now explicitly unavailable; it is not assigned residual scores or information criteria.

Direct regression tests cover both likelihood cases. This is a statistical inference correction only: it introduces no new physical model, calibration step, input assumption, or experimental-validation claim.


## K. Projected-EVDF regularization sensitivity (2026-11-02)

The projected one-dimensional EVDF inverse now summarizes numerical sensitivity across the successful candidates in its existing logarithmic GCV grid. For each candidate it reports the inferred projected mean velocity, standard deviation, skewness, excess kurtosis and total-variation distance from the selected distribution, together with the candidate GCV score. Range summaries are returned for these metrics. If a researcher explicitly supplies one regularization strength, the result is labelled as a single-candidate case and no sensitivity range is claimed. The Python output, primary diagnostic registry and packaged registry are synchronized; regression tests cover GCV and explicit-value behavior. This is a numerical robustness diagnostic, not a measurement uncertainty, posterior interval or experimental validation.


## J. Scattering-volume local-field propagation regression (2026-11-02)

A source-level trace found that the Full-EM scattering-volume loop recorded each sample's supplied local magnetic field but recursively evaluated every sample using the top-level magnetic_field_T. This made the integrated spectrum insensitive to the per-sample magnetic-field variation even though the returned field diagnostics showed that variation. The recursive forward call now receives that sample's own magnetic_field_T. A regression test independently evaluates each local sample and checks the weighted, path-overlap-scaled spectrum against the integrated result. This closes the local-field propagation defect only; the declared spatially uniform fitted plasma-parameter assumption remains unchanged.


### 2026-11-02 — local Full-EM CTS external-input covariance

The Full-EM measured-spectrum inversion now propagates an explicitly supplied positive-semidefinite joint covariance over the supported local magnetic-field, incident/scattered direction and complex polarization components by reproducible Monte Carlo refitting. The dashboard exposes the covariance upload, ordered component names and draw count. The Python runtime registry, both diagnostic JSON registries, tests and CTS scientific documentation declare the same contract. Frequency-resolved ray/equilibrium/beam covariance, composition covariance and cross-covariance between distinct external sources remain outside this local-vector implementation; ray-resolved inputs are rejected for this covariance route. This is conditional uncertainty propagation, not calibration or experimental validation.
