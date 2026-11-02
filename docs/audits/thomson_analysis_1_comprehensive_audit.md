# Thomson Scattering (Analysis 1) — Comprehensive main-branch audit

**Audit date:** 2026-11-01  
**Reviewed baseline:** `main` at `ecf4242b25b613afabe15177f9edd542e8a6f1d8`  
**Scope:** primary optical Thomson inference, spectral-distribution subanalysis, spatial-profile reconstruction, microwave CTS electrostatic and full-electromagnetic branches, registry/API/tests/documentation.  
**Method:** direct GitHub source and registry inspection. No local checkout or local edits were used. GitHub Actions are the software-test route; this audit does not claim experimental validation.

## Governing input contract

The input is **measured diagnostic data already prepared and calibrated by the instrument/operator**. TIS performs the selected physical forward inference from those calibrated measurements. It does not calibrate the instrument, correct detector counts, repair acquisition data, measure an instrument response, or claim independent experimental validation. Required fixed instrument/geometry metadata may be supplied to the forward model; supplying such metadata is not an invitation for TIS to calibrate it.

The three result classes are:
- **Primary:** uniquely inferred from the selected measured spectrum and supplied fixed metadata.
- **Conditional:** inferable only when explicitly identified auxiliary quantities/metadata are supplied.
- **Non-identifiable:** not uniquely determined by the measured data and selected model; never label as a measured result.

## Executive summary

The main branch already contains a substantial Analysis 1 implementation: three explicit non-collective optical models (Maxwellian, finite-variance Kappa and relativistic Maxwell–Jüttner), a separate microwave CTS dispatcher with electrostatic and magnetized full-EM branches, descriptive spectral moments, and spectrum-to-profile reconstruction.

This review separates **actual repository inconsistencies** from **declared scientific limitations** and **not-yet-implemented capabilities**. A declared limitation is not automatically a coding defect. Several boundaries are intentional consequences of identifiability or the agreed project scope; removing them requires an explicit physical model and additional measured inputs, not a silent code change.

## A. Confirmed contract, documentation and test inconsistencies

1. **Spectral-distribution module docstring contradicted its API.** It referred to scalar Gaussian-width moment deconvolution although the callable accepts a measured response matrix and reports observed-spectrum moments without scalar-width correction. Fixed in `tokamak_analysis/thomson_distribution.py`.
2. **A registry test expected unsupported deconvolution fields.** `tests/test_analysis_registry.py` expected `instrument_sigma_uncertainty_nm`, `instrument_corrected_sigma_nm` and `instrument_width_resolved`, absent from the actual function and canonical contract. Updated the test to assert the supported observed-width contract and the absence of those unsupported outputs.
3. **Primary Thomson uncertainty metadata understated supported propagation.** The registry described laser-wavelength and scalar-angle uncertainty as rejected for the relativistic branch, while the implementation and `test_relativistic_model_propagates_laser_angle_and_polarization_uncertainties` exercise Monte Carlo propagation. The registry wording was corrected for the supported model branches.
4. **Profile-reconstruction uncertainty metadata was stale.** The wrapper forwards the canonical inference inputs; its registry text incorrectly said that external laser/geometry/polarization uncertainty propagation was not implemented for the relativistic model. Corrected to describe the forwarded branch-specific inference contract.
5. **Profile covariance-combination metadata was contradictory.** The profile entry incorrectly prohibited combinations that the primary Analysis 1 contract permits when the scattered-power row/column is zero. Corrected the per-channel and joint covariance descriptions.
6. **The two canonical registry copies must remain identical.** Both root and packaged diagnostic registries were updated together; the existing synchronization test remains the guard against drift.

## B. Optical Thomson physical-model scope and identifiability

7. The Maxwellian optical branch assumes an isotropic, non-relativistic electron distribution.
8. The Maxwellian branch returns only the flow projection along the scattering vector, not a 3-D flow vector.
9. The Kappa branch is an isotropic, finite-variance parameterized distribution; it is not a general non-Maxwellian inversion.
10. The Kappa spectral model uses the analytic one-dimensional Student-t projection; it is not the generalized plasma-dispersion-function algorithm of Skolar et al.
11. The Kappa index is identifiable only when the measured spectral range, resolution and signal information constrain the extra shape parameter; otherwise it must remain rejected/unavailable.
12. The relativistic optical branch assumes an isotropic Maxwell–Jüttner rest-frame distribution.
13. The relativistic optical branch restricts bulk flow to the scattering-vector direction.
14. The relativistic optical branch uses elastic Thomson kinematics and omits photon recoil/Compton physics.
15. Its recoil-correction screen is a configured numerical error-budget threshold, not a universal physical boundary.
16. The non-relativistic θe and flow-β screens are conservative asymptotic software screens, not universal validity limits.
17. Optical collective dielectric inference is not part of the optical model shortlist; microwave CTS is a distinct branch. Do not present the CTS solver as an optical collective solver.
18. No unrestricted arbitrary 3-D electron velocity-distribution reconstruction is implemented.
19. No anisotropic optical electron distribution is inferred.
20. No multi-component optical electron-population decomposition is claimed.
21. A single scattering geometry cannot identify three independent flow components.
22. Absolute electron density is not identifiable from normalized spectral shape alone.
23. Absolute density requires spectral intensity explicitly declared in W/nm.
24. Absolute density additionally requires incident laser power at the scattering volume.
25. Absolute density requires the effective interaction length.
26. Absolute density requires collection solid angle and optical efficiency.
27. Absolute density requires consistent incident/scattered directions and incident polarization for the polarized cross-section.
28. The density relation assumes optically thin, single-scattering Thomson conditions.
29. The reported density is an interaction-length-averaged effective density; it is local only under the stated uniformity condition along the path.
30. The optical density model does not perform spatial integration over a varying laser intensity, collection efficiency or density field along the scattering volume.
31. The canonical optical input is channel × wavelength; it is not a general time × channel × wavelength movie inversion.
32. The optical physical fit does not infer ion temperature from the electron Thomson spectrum.
33. The selected physical model is explicit; the software does not automatically classify the plasma regime or switch models.
34. The supplied measured response is part of the forward operator; TIS does not measure or calibrate it.
35. A 1-D index-space response kernel requires a uniform wavelength grid; nonuniform grids require a measured wavelength-to-wavelength response matrix.
36. A response matrix is interpreted as a non-negative redistribution operator with normalized columns; arbitrary signed operators are outside the declared response model.

## C. Optical uncertainty, nuisance parameters and channel coupling

37. Full channel-major spectral GLS is implemented for non-collective Maxwellian and Kappa fits, but not for the relativistic Maxwell–Jüttner fit.
38. The relativistic branch uses channel-local fit covariance plus its supported external-parameter and response Monte Carlo propagation; it does not return the missing full cross-channel spectral GLS information.
39. Cross-channel covariance cannot be inferred if the user supplies only independent pointwise uncertainties and no shared covariance source.
40. If no measurement uncertainty is supplied, the fit is unweighted and residual-variance-scaled covariance is not a measurement-noise-calibrated uncertainty budget.
41. A local linear intercept-and-slope background nuisance model is used in the non-collective Maxwellian/Kappa fits.
42. Curved, structured, line-contaminated or time-varying background components are not independently separated by that two-parameter baseline.
43. A structured background model would need explicit model selection and measured constraints; silently adding one would change the physical inference.
44. Laser-wavelength, scattering-angle, beam-direction and polarization uncertainties have model-specific support; unsupported combinations must remain rejected rather than silently ignored.
45. Scalar scattering-angle uncertainty and beam-direction component uncertainties describe alternative geometry representations and must not be double-counted.
46. Shared 2-D response-matrix uncertainty represents a common instrument realization across channels; channel-specific 3-D response uncertainty has a different covariance structure.
47. Response-matrix Monte Carlo uncertainty is conditional on the supplied response uncertainty model and does not independently validate the response.
48. Absolute-density covariance combines fitted scattered-power uncertainty and optical-factor uncertainty only under the explicit zero-power-row/column rules; violating those rules double-counts uncertainty.
49. The density covariance model does not invent missing correlations between external optical factors.
50. The supplied radius/channel coordinates are treated as exact; their uncertainty is not propagated through the spatial reconstruction.
51. Wavelength-axis calibration uncertainty is outside the input contract; the wavelength axis is assumed already calibrated. Laser-wavelength uncertainty is a distinct physical input.
52. The uncertainty budget does not become complete merely because a local fit covariance exists; model discrepancy, unprovided external inputs and systematic uncertainties remain separate.

## D. Thomson spectral-distribution subanalysis

53. The subanalysis reports wavelength-space moments and a Gaussian shape descriptor, not an electron velocity distribution.
54. Its centroid, variance, skewness and excess kurtosis are descriptive properties of the measured calibrated spectrum after the explicitly selected background treatment.
55. Exactly one background representation must be supplied: a background spectrum, a boolean background mask, or an explicitly selected edge fraction.
56. The edge-fraction method is a researcher-selected heuristic and can subtract real Thomson signal if the physical spectrum reaches the selected edges.
57. Background-uncertainty propagation is conditional on the selected background representation: an explicitly supplied background can provide pointwise/full covariance and signal-background cross-covariance for joint moment Monte Carlo; a mask-/edge-fitted linear background is re-fitted for each perturbed signal draw. A separate shared systematic covariance for the fitted background coefficients is not an independent input in those mask-/edge-fitted paths. Do not describe all background-parameter uncertainty as unpropagated.
58. **Full within-channel spectral covariance and conditional signal/background cross-covariance are implemented.** When `signal_covariance` is supplied, the Gaussian fit uses the covariance-whitened objective and moment Monte Carlo draws correlated wavelength samples through the covariance Cholesky factor; the pointwise `signal_uncertainty` path remains independent-sample only. An optional `signal_background_cross_covariance` now propagates the cross-correlation between the measured signal and an explicitly supplied fitted background. It requires both marginal covariance models and a positive-definite joint block covariance; absent that input, signal and background errors remain independent. Regression coverage includes `test_distribution_propagates_full_within_channel_covariance`, `test_distribution_propagates_joint_signal_background_cross_covariance`, and rejection of non-positive-definite joint covariance.
59. Signed moments retain negative residuals; near-cancellation, small signed area or non-positive variance makes them unstable or undefined.
60. The reported width is an observed wavelength-space width; it is not an instrument-corrected electron-temperature width.
61. The measured response matrix is used as a forward operator for the Gaussian shape fit; the measured spectrum is not deconvolved.
62. Gaussian shape-fit covariance is conditional on the fixed supplied background and response matrix.
63. The subanalysis does not independently extract Te, ne or flow; those remain outputs of the parent physical Thomson inference.
64. A Gaussian shape fit is a descriptor, not proof that the underlying electron distribution is Maxwellian.

## E. Thomson spatial-profile reconstruction

65. PCHIP and generalized-cross-validation smoothing splines are numerical representations of channel-inferred parameters, not additional plasma-physics forward models.
66. Reconstruction is restricted to the measured radial-coordinate interval; extrapolation is deliberately forbidden.
67. The supplied coordinate is radius, not automatically a flux-surface coordinate; no equilibrium mapping is inferred.
68. The workflow does not reconstruct a Grad–Shafranov equilibrium or infer magnetic flux surfaces.
69. GCV smoothing requires at least five spatial channels; otherwise that method returns an explicit insufficient-channel status.
70. When only pointwise channel uncertainties are supplied, the reconstruction assumes independent channel errors; a full covariance is needed to represent cross-channel correlation.
71. Monte Carlo draws that violate the positive-profile or numerical-fit constraints are discarded; too few successful draws produce a warning rather than an uncertainty band.
72. No spatial point-spread function or cross-channel optical mixing operator is included in the profile reconstruction.
73. The workflow reconstructs profiles after channel-level physical inference; it is not a single spatially regularized joint inversion of all measured spectra.
74. GCV smoothing can suppress sharp spatial transitions; PCHIP and spline outputs should be treated as alternative representations, not additional measured channels.
75. No boundary condition, separatrix constraint, flux-surface prior or equilibrium-informed radial regularization is imposed.
76. The profile grid adds interpolated coordinates, not new independent measurements.

## F. Microwave CTS — electrostatic branch

77. The electrostatic CTS branch is an unmagnetized longitudinal dielectric model; it is not a substitute for magnetized full-EM CTS when magnetic response matters.
78. Ion species, charge states, masses and charge-density fractions are external inputs; composition is not automatically identified from one spectrum.
79. A single CTS geometry identifies only projected flow components along the scattering wavevector, not 3-D flow vectors.
80. The electrostatic branch's tabulated fast-ion option uses a one-dimensional projected VDF; it does not recover a general 3-D fast-ion distribution.
81. External-parameter uncertainty support is limited to the declared independent marginal inputs; a full cross-parameter covariance model is not available for every electrostatic external input.
82. Response-matrix uncertainty propagation requires the supported measured 2-D or channel-specific 3-D response representation; a 1-D kernel/callable is not an interchangeable uncertainty representation.
83. Global-search optimizers require finite researcher-supplied bounds; the software does not invent scientific parameter ranges.
84. Regularized tabulated fast-ion VDF inference requires an explicit regularization choice; regularization strength is a numerical/statistical control, not a physical model.
85. The fitted ion-flow parameterization is a projected component shared by the electrostatic branch as documented; it should not be reported as independent species-resolved 3-D ion flow.

## G. Microwave CTS — magnetized full-electromagnetic branch

86. The full-EM branch requires an explicit magnetic field, wave geometry and polarization/eigenmode selection; it must not silently fall back to electrostatic CTS.
87. Its thermal bulk-electron nonlinear source uses the low-temperature Bindslev Gamma/X closure.
88. The Bindslev closure has a low-temperature asymptotic domain; the optional θe screen is not a universal physical threshold.
89. The measured-spectrum full-EM inversion currently fits the Maxwellian bulk-electron branch; relativistic gyrotropic electron distributions are forward-model capabilities, not fitted inversion parameters.
90. Supplied fast-ion distribution shapes are fixed external inputs in full-EM inversion; they are not reconstructed from the measured spectrum.
91. Ion species and charge-density fractions remain fixed external inputs, although species-resolved thermal-ion temperatures and projected flows are fitted.
92. The relativistic kinetic vertex is a homogeneous local kinetic model.
93. Finite-orbit-width effects are not included.
94. Collisional kinetic corrections are not included in the declared full-EM model.
95. Spatially varying relativistic kinetic response and nonlinear equilibrium evolution are not included.
96. Ray tracing is cold-plasma geometrical optics, not diffraction-resolved quasi-optical propagation.
97. Beam covariance matrices are supplied at ray samples; the software does not propagate beam covariance/diffraction along the path.
98. Mode conversion and branch tracking through degeneracies are not implemented.
99. Scattering-volume integration assumes fitted ne, Te and thermal-ion temperatures are uniform over the sampled volume while local field and ray directions may vary.
100. Beam-overlap path integrals are relative spectral-bin weights; absolute launcher/receiver power coupling is not inferred.
101. Full-EM inversion accepts a supplied measured response-matrix 1σ uncertainty, samples non-negative column-renormalized response operators, refits the complete physical parameter vector, and adds the successful-draw covariance to the conditional fit covariance. This propagation is conditional on fixed external magnetic field, geometry, species composition and polarizations; it is not response calibration.
102. Full-EM inversion does not propagate the full external magnetic-field, geometry, polarization, ray or equilibrium covariance.
103. Full-EM covariance is conditional on fixed external inputs and the selected damping/quadrature/harmonic settings.
104. If no spectral noise/covariance is supplied, its unweighted-fit covariance is residual-scaled and not measurement-noise-calibrated.
105. Positive log-parameters have no default scientific bounds; researcher-supplied bounds are available where needed.
106. Paired gyro-harmonic and momentum/velocity quadrature convergence is default-enabled in the Full-EM measured-spectrum inversion, with explicit refinement sequences and L2/L-infinity tolerances; it is a numerical discretization assessment, not a physical uncertainty estimate.
107. The damping default is numerical resonance regularization, not a physical collision frequency.
108. Full-EM flow outputs are scalar projections along the scattering wavevector; one view cannot uniquely determine a 3-D flow vector.
109. Frequency-resolved ray geometry must match the measured frequency-shift axis and explicitly select the propagation branch; no default mode branch is invented.
110. No independent experimental validation is claimed for either CTS branch.

## H. Recent literature and model-selection decision

Recent sources are useful for model scope and algorithmic development, but they do not automatically authorize a new physical model:

- Yamada et al., *LHD Thomson Scattering Diagnostics*, Journal of Fusion Energy (2025), DOI: 10.1007/s10894-025-00520-4.
- Skolar, Longley & Goodwin, *Computing the generalized plasma dispersion function for non-Maxwellian plasmas, with applications to Thomson scattering*, Physics of Plasmas (2025), DOI: 10.1063/5.0261966.
- Sakai et al., *Monte Carlo simulation method for incoherent Thomson scattering spectra from arbitrary electron distribution functions*, JINST (2026), DOI: 10.1088/1748-0221/21/01/C01002.
- Foo et al., *Regularized inversion of optical Thomson scattering spectra for improved determination of plasma parameter evolution*, Review of Scientific Instruments (2026), DOI: 10.1063/5.0350497. The reported demonstrations include synthetic test problems; this is an inversion/regularization method, not a new scattering physical model and not evidence of experimental validation of TIS.
- Zhang et al., *Spectral decomposition for collective Thomson scattering based on an improved genetic algorithm*, Review of Scientific Instruments (2024), DOI: 10.1063/5.0215734.
- Flocken, Korsholm & Rasmussen, *Accuracy-driven optimisation procedure for collective Thomson scattering diagnostics for fusion plasmas*, Fusion Engineering and Design (2026), DOI: 10.1016/j.fusengdes.2026.115845.

The current shortlist remains deliberately small. A new optimizer, regularizer, neural network or fitting strategy must not be counted as a new physical scattering model. Any new physics branch needs an explicit input contract, identifiable outputs, uncertainty model, reference equations and user approval where it changes the scientific scope.

## I. Changes applied during this pass

The following verified inconsistencies were corrected directly on `main`:

- Clarified the spectral-distribution module docstring: observed moments/width only; no scalar-width deconvolution or instrument-width subtraction.
- Corrected the stale registry test so it tests the actual supported spectral-distribution API and explicitly rejects unsupported corrected-width fields.
- Corrected primary/profile registry descriptions for supported laser-wavelength and scattering-angle uncertainty propagation.
- Corrected profile covariance descriptions so the permitted joint spectral/response plus optical covariance combination is accurately conditioned on zero scattered-power rows/columns.
- Updated both root and packaged `DIAGNOSTIC_REGISTRY.json` copies together.

Relevant commits:
- `5bdca94956188885612185e2d49db5627fea016d`
- `fbfe061390b2d6fcccd41b0d61e85c22c50f756f`
- `180d20ae2415ff2ce69f935f634df757e027392e`
- `d0432837e40a5ef9b043edbf964bc9e30dfc5661`

## J. EVDF extraction decision and implementation status

The spectral-distribution subanalysis (registry entry 81) remains a descriptive wavelength-space moment/shape analysis. It is not conflated with electron velocity-space inference. The separate thomson_evdf.py inverse solver now has automatic generalized-cross-validation (GCV) selection of the second-difference Tikhonov regularization strength when no explicit value is supplied. It reports the candidate grid and GCV scores, plus a local active-set covariance and pointwise standard uncertainty conditional on the selected grid, response, fitted background, regularization and active set.

The primary measured-spectrum inference now automatically attempts a one-dimensional normalized electron-velocity marginal along the scattering vector for each channel. It reuses the same calibrated measured spectrum, fitted physical-model background nuisance component, measured response matrix and supplied spectral uncertainty/covariance. Its velocity interval is derived from the measured wavelength coverage and scattering geometry. Channel-level failures are reported as unavailable with a reason; they do not invalidate the primary physical Thomson fit. The result is not an unrestricted 3-D EVDF, does not infer anisotropy or perpendicular velocity components from one view, and does not independently identify absolute density. Its directional equivalent temperature is not substituted for the scalar physical electron temperature.

The previous uncertainty-registry wording that incorrectly restricted scalar scattering-angle uncertainty to only non-relativistic models has also been corrected in both canonical registry copies. A regression test now guards the relativistic Maxwell-Juttner uncertainty contract.

## K. Follow-on changes applied directly to main

- Corrected primary Thomson registry metadata for laser-wavelength and scalar scattering-angle uncertainty propagation, including the mutually exclusive beam-direction uncertainty representation.
- Added automatic GCV regularization selection to the projected EVDF inverse solver while preserving explicit positive regularization as an override.
- Added conditional active-set EVDF covariance and pointwise standard uncertainty, with its scope stated explicitly.
- Integrated per-channel projected EVDF extraction into the primary Thomson measured-spectrum workflow; it is enabled by default and can be explicitly disabled. The workflow returns the EVDF result or a channel-specific unavailable status.
- Added the enable_evdf_reconstruction and evdf_velocity_points input contract and the electron_velocity_distribution output contract.
- Added regression coverage for GCV selection, conditional covariance, primary-workflow integration and explicit opt-out.
- Updated the Thomson analysis documentation and synchronized both root and packaged diagnostic registries.

Follow-on commit heads include:
- a3215471279c4bbf4d7c4400a0c262a694a56736
- fd2762f3e7ef088d7443e6c26a80629877c383ff
- 4e67b7b71a6eec776054bdcd74b6f45c7dd40e3b
- b208a5f9155dc0868527f550011721b9ca559207
- 3265202783422767255065611ea023da27da3048
- 22b44d11c2b30d3f93c6104ae9197c0db99190ba
- aa4b8f1718194d5ca2671dea2abd67d3c306fd14
- 2c0a85b998072becf59977fe0243a3ef671a6071
- 52a31fb8f1a581d59f5a8a4d393a25f1d78c0ed6
- cc0e199224f4dfadc4744010a1d11a1c787764b1

**Validation status:** no experimental validation was performed or claimed. GitHub Actions software tests are separate from experimental validation. The latest Actions run for the final source/documentation commit must complete before describing this work as software-test-clean. The EVDF covariance is a conditional local approximation, not a posterior or experimental uncertainty validation.


## L. Additional confirmed implementation defect found in the 2026-11-01 review

111. **Relativistic primary inference incorrectly invoked the non-relativistic projected-EVDF inverse.** The primary workflow called `reconstruct_thomson_evdf` for every selected optical model. That inverse uses the first-order non-relativistic Doppler mapping `delta_omega = q·v`, while the relativistic Maxwell-Juttner primary model uses Lorentz-transformed photon-frequency kinematics and polarization-dependent cross-sections. A successful numerical fit therefore did not establish a physically consistent relativistic EVDF. **Fixed on main:** the non-relativistic EVDF inverse is now withheld for `non_collective_relativistic_maxwell_juttner`, with an explicit `unavailable_for_selected_physical_model` status and a conditional identifiability explanation. The primary Maxwell-Juttner fit and its density/uncertainty outputs are unaffected. Added a regression assertion to the relativistic inference test and updated the analysis documentation. A relativistically consistent EVDF inverse remains future work and is not claimed as implemented.



## M. Additional confirmed registry-contract defects found in the 2026-11-01 review

114. **The machine-readable output contracts omitted a result that all four Analysis 1 workflows actually return: `parameter_identifiability`.** The primary optical Thomson fit, CTS dispatcher branches, spatial-profile workflow and spectral-distribution workflow each attach an explicit primary/conditional/non-identifiable classification, but their diagnostic registry output specifications did not declare it. **Fixed on main:** the structured identifiability output is now declared for all four records in both root and packaged diagnostic registries, with a regression test requiring it.

113. **The parent Analysis 1 registry overstated projected-EVDF availability across physical models.** Its primary contract said the automatic EVDF inverse was attempted for each measured spectrum, without recording that the implementation uses non-relativistic Doppler kinematics and must not be applied to the relativistic Maxwell-Juttner branch. **Fixed on main:** the primary registry input/output descriptions and algorithm now explicitly limit the inverse to non-collective Maxwellian/Kappa models and state that it is withheld for Maxwell-Juttner until a relativistically consistent inverse exists. The root and packaged copies remain synchronized, and a regression test guards the model-dependent contract.

112. **The diagnostic registry merged two distinct Thomson workflows into the spectral-distribution entry.** The `Thomson spectral distribution` row described both `analyze_thomson_spectral_distribution` and the separate `reconstruct_thomson_evdf` inverse. Its input schema consequently contained duplicate names (`background_signal`, `signal_uncertainty`, `signal_covariance`) and EVDF-only required inputs (`laser_wavelength_nm`, `scattering_angle_rad`, `regularization_strength`) that the registered spectral-distribution callable does not accept. Its output schema likewise mixed descriptive wavelength moments with projected velocity-distribution outputs. The row also incorrectly marked `calibrated_measurements_to_physics` true despite explicitly describing a shape-only subanalysis. **Fixed on main:** the spectral-distribution row now declares only the nine inputs and descriptive outputs of its actual callable, has no duplicate input/output names, and correctly sets calibrated-measurements-to-physics false. The projected EVDF callable is mapped to the parent primary Analysis 1 workflow, whose output contract already declares `electron_velocity_distribution`. Both root and packaged diagnostic registries were synchronized. A regression test now enforces the separation and parent mapping.


## N. Additional confirmed executable-registry defect found and corrected on 2026-11-01

115. **The Python runtime registry was stale while both JSON diagnostic-registry copies were current.** The executable `tokamak_analysis.analysis_registry.ANALYSES` retained old Analysis 1 and spatial-profile (entry 46) input descriptions claiming that laser-wavelength and scattering-angle uncertainty were rejected for the relativistic Maxwell-Juttner model. The implemented inference and the root/package JSON contracts support those uncertainties through Monte Carlo; scalar-angle and beam-direction-component uncertainty representations are mutually exclusive. This made the Python API contract contradict both the implementation and the already-updated JSON registries, and contradicted the existing registry regression test. **Fixed directly on main:** synchronized both affected Python-registry entries with the implemented model-specific uncertainty propagation and the canonical JSON contract. The existing registry regression `test_thomson_relativistic_geometry_uncertainty_contract_matches_implementation` checks the runtime metadata, while `test_relativistic_model_propagates_laser_angle_and_polarization_uncertainties` exercises the inference path; software-test completion remains unclaimed until GitHub Actions reports the resulting run.

Fix commit: `a5305b23000522cf17c4f2c244801ba150747c74`.



## O. Additional conditional uncertainty capability implemented on 2026-11-01

The spectral-distribution subanalysis now accepts `signal_background_cross_covariance` for the covariance block \(\operatorname{Cov}(s_i,b_j)\), where \(s\) is the measured calibrated spectrum and \(b\) is the explicitly supplied fitted background. The feature:

- requires signal and background marginal covariance models (pointwise 1-sigma inputs are converted to diagonal covariance matrices);
- constructs the full \(2N\times2N\) block covariance and rejects it unless it is positive definite;
- samples signal and background perturbations jointly, preserving their supplied cross-correlation in moment uncertainty propagation;
- leaves the measured spectrum unchanged and does not modify the signal-only Gaussian-fit weighting;
- does not apply to mask- or edge-fitted backgrounds, whose background is re-fitted in each signal draw.

The Python callable, Python execution registry, root and packaged JSON registries, scientific documentation and independent software regression tests were updated directly on `main`. This is a conditional software capability, not an inference of missing covariance and not experimental validation.

Commits:
- `fdbcecdf3800b4e60e16778b47a0d08c6f7e7d0f` — implementation
- `0efbb8d9f08fca807973714459bfffe8c0adfcbb` — regression tests
- `6c5ebff3dc6fe427871909b3e8cabbb68d6f525b` and `2306b6f51684ee34ea0163d9b632f9ab33a8d4be` — root/package JSON registries
- `29fa2c2ae633e23e4e67bdd39f6f99f7f1aec562` — Python runtime registry
- `25debe7cd0e5df8269e613f5ae55de6b4648b238` — scientific documentation
- `e7e8088fd387f8d59acc0613f90035127ec41178` — explicit uncertainty-method provenance
- `f31efd63019349b744e6e0a8bd7314a40b3e795c` — regression assertion for correlated-covariance provenance

**Software CI status:** not yet confirmed for these direct-main commits. No experimental validation was performed or claimed.


## P. Dashboard/API integration updates applied directly to `main` on 2026-11-01

- Exposed the fixed-operator joint wavelength-resolved Maxwellian spatial-spectral inverse from the primary Thomson dashboard, with explicit spatial-mixing-matrix upload and numerical curvature-regularization input.
- Aligned the joint-inverse result with the shared profile schema while explicitly returning absolute density as unavailable; added the structured `parameter_identifiability` result and dashboard warnings when data-only identifiability fails. Optimizer candidates remain non-primary diagnostics.
- The dashboard rejects laser-wavelength, scattering-angle or response-matrix uncertainty inputs in this joint route because those uncertainty paths are not propagated there; it does not silently discard them. Absolute-density input is disabled for this shape-only joint route.
- Added software regression coverage for the shared result schema, identifiability classification and dashboard controls. No experimental validation was performed or claimed.
- Exposed the already-implemented electrostatic CTS joint external-parameter covariance upload and explicit parameter order in the dashboard; documented supported names and mutual exclusivity with scalar uncertainty inputs.
- Exposed Full-EM CTS response-matrix uncertainty and Monte Carlo refit-count controls in the dashboard, and documented the conditional response-uncertainty semantics. The electrostatic and Full-EM response representations remain distinct.
- Updated the item-by-item matrix: items 72–73 are implemented/tested within the declared Maxwellian joint scope; item 81 is implemented/tested for its declared electrostatic external covariance scope. Items 20, 30, 37, 43, 44, 50, 72–73 (model extensions), 82, 89, 91, 97–99, 101–102 and 106 retain their explicit open/partial or model-boundary dispositions as recorded in the matrix.

Relevant direct-main commits include `d3fdb8f`, `ef39793`, `e04726d`, `be8a094`, `95966fb`, `5656d20`, `f677d2b`, `b736ed8`, `33e5f84`, `af97009`, and `1a760c6`. GitHub Actions must complete successfully for the exact final commit before software-test success is claimed. Experimental validation remains not performed.


## Subsequent main-branch implementation: path-integrated density and CTS mode tracking

The calibrated optical Thomson API now includes the conditional helper
`infer_thomson_path_averaged_density`. It evaluates the single-scattering
sightline integral with explicitly supplied path-dependent incident power,
effective differential cross-section, collection solid angle and optical
efficiency. It reports a weighted path-average density only; a single
integrated measurement cannot recover an arbitrary pointwise density profile.

The CTS geometry API now includes
`track_cold_plasma_modes_from_operators`, which follows polarization branches
by maximum-overlap assignment and phase alignment along an ordered sequence of
cold-plasma wave operators. Exact/near-degenerate eigenspaces are explicitly
flagged as ambiguous. This does not solve coupled-mode conversion.

These are software capabilities with independent unit/regression tests. They
do not constitute empirical validation.


## Q. Additional confirmed registry defect corrected on 2026-11-01

116. **The spatial-profile reconstruction execution contract contradicted the implemented relativistic geometry-uncertainty path.** The canonical Python registry entry 46 and both JSON diagnostic-registry copies still described laser-wavelength uncertainty as rejected for the relativistic Maxwell-Juttner model, and described scalar scattering-angle uncertainty as non-relativistic-only. The profile workflow forwards these inputs to the canonical inference engine, whose relativistic branch propagates laser-wavelength and scalar-angle uncertainty by Monte Carlo; scalar-angle and beam-direction-component uncertainty representations are mutually exclusive. **Fixed directly on main:** synchronized the entry-46 Python runtime contract and both root/packaged JSON registry copies with the implemented model-specific propagation. Added an independent registry regression test for entry 46, parallel to the existing primary-entry regression. This is a metadata/API consistency correction; it does not add a new physical model or claim experimental validation.

Commits:
- `3b70ab9cba11c715cb9b5cbf4e0f78d7c71f9a6a` — Python execution registry
- `467e3e6f6eee7fa6ecd0a794bead9366a9445433` — root JSON registry
- `b77647415444564fe0c43f8d3a2faf855e712157` — packaged JSON registry
- `549ea6baac866f0e4998648519f43145842f9644` — regression test

**Experimental validation:** not performed and not claimed. The exact final software commit must pass GitHub Actions before software-test success is reported.


## R. Full-EM CTS response-uncertainty registry integration corrected on 2026-11-01

The executable `tokamak_analysis.cts_full_em.analyze_full_em_cts_spectrum` already accepted `instrument_response_uncertainty` and `response_mc_samples`, performed non-negative column-renormalized response perturbations, repeated the full least-squares parameter inversion, and added successful-draw parameter covariance to the conditional fit covariance. Direct numerical tests in `tests/test_cts_analysis.py` exercise the supplied response-uncertainty route and assert successful propagation.

The CTS registry had drifted from that implementation: the Python runtime registry incorrectly restricted both inputs to electrostatic CTS, and the JSON registry combined a Full-EM-supported description with an electrostatic-only applicability flag. Its uncertainty-model prose also contained contradictory statements, and the Full-EM output contract omitted the actual response Monte Carlo result keys. **Fixed directly on main:** removed the electrostatic-only applicability restriction, synchronized both JSON copies and the Python runtime registry, declared the Full-EM response uncertainty outputs, and included the controls in the Full-EM workflow-mode input list. Added a registry/API integration regression that checks runtime metadata, JSON parity, branch availability and declared outputs.

The separate full external magnetic-field, geometry, polarization, ray and equilibrium covariance remains unimplemented. Response-matrix Monte Carlo is not instrument calibration and is not experimental validation.

Commits:
- `ed937e735a2f20f30e033197aa591f3648399e09` — root diagnostic registry
- `d185fb499cd2b30176511ddaef403ff6c5f28d93` — Python runtime registry
- `0ae638ce1a8448b3c22f8c2a63223f6ea6068bf2` — packaged diagnostic registry
- `b4c201179780b39e35b58a46e565669b6aa73a29` and `f2c7a2de925282a5b21a8fa08987c357770291f2` — registry/API regression coverage

**Experimental validation:** not performed and not claimed.


## P. Full-EM convergence execution-contract synchronization

On 2026-11-01, the Full-EM measured-spectrum inversion's automatic paired gyro-harmonic / momentum-quadrature convergence assessment was synchronized across the user interface and all executable contracts:

- The dashboard now exposes the default-enabled convergence assessment, optional paired increasing harmonic/quadrature refinement sequences, and the relative L2/L-infinity tolerance.
- The Python runtime analysis registry now declares the four convergence inputs and all three returned convergence fields, matching the callable signature and result structure.
- Both root and packaged diagnostic JSON registries declare the same convergence input/output contract.
- Regression coverage checks the Python and JSON contracts, default-enabled behavior, and the explicit separation of numerical convergence from physical uncertainty.
- The CTS scientific note no longer describes the automatic post-fit assessment as opt-in.

This is a numerical discretization assessment at fixed fitted parameters. It does not add a physical uncertainty term, change fitted parameters, or constitute experimental validation. GitHub Actions must complete successfully before this source change is described as software-test-clean.


## S. Additional confirmed root/package registry defect corrected on 2026-11-02

117. **The packaged diagnostic registry omitted the primary Thomson `result_classification` output.** The root `DIAGNOSTIC_REGISTRY.json` declared the per-channel Primary / Conditional / Non-identifiable result contract, but `tis/data/DIAGNOSTIC_REGISTRY.json` did not. Because the Python runtime registry loads the packaged copy, runtime metadata could omit a result field returned by the canonical Thomson workflow, and the existing root/package parity regression would fail. **Fixed directly on main:** the packaged registry was synchronized byte-for-byte with the root registry. The existing `test_analysis_registry` parity assertion covers this correction. No physical model, input assumption or output value was added; this restores the already-implemented result contract. No experimental validation is claimed.

Fix commit: `a1462d79702b44e857332067d34642d41274d3fb`.


## T. Calibrated response preservation and strict optical geometry domain — 2026-11-02

118. **Several Thomson response paths silently renormalized the supplied calibrated response matrix.** The canonical optical inference, projected-EVDF inverse, spectral-distribution analysis and legacy spectrum adapter must treat a supplied calibrated response operator as fixed measured input. Renormalizing its columns changes the operator supplied by the researcher and is an unrequested data/model transformation. **Corrected directly on main:** matrix paths now reject non-finite, negative, zero-column or non-unit-column operators as applicable, and preserve an accepted matrix unchanged; no normalization is performed. Regression tests cover the primary inference, EVDF inverse, spectral-distribution path and legacy adapter, including preservation of the supplied matrix.

119. **The optical scattering-angle validator admitted the exact backscattering endpoint π.** The supported optical geometry contract is an angle strictly inside the open interval (0, π); the endpoint is now rejected explicitly. Added a regression test at exactly π. This is input-domain enforcement, not a calibration or measurement operation.

Relevant direct-main commits:
- `e38d6d79d19e033a304b4ca2eb143fab2337a244` — preserve the calibrated response in primary optical inference.
- `7c1e9ea215c9ccc17165ad29a58fd917d69de7c0fb` and `58fea7dea3a3a956484d174d4d7521955af0a364` — preserve response operators in spectral-distribution and EVDF paths.
- `ef75f0484c7631c8b35f1106e466e2037bad976d` — remove silent normalization from the legacy adapter.
- `f149306964ecb4f5917a962e61890510324632fd` — enforce the strict optical scattering-angle domain.
- `0aae1c5287da46541b2f70a5da99bf094109cfd6` and `2dbea429e4b8c15d9753f4c63b2627f2bc08e7cf` — regression tests.

**Experimental validation:** not performed and not claimed. GitHub Actions for the current main head are pending; software-test success is not asserted until those workflows complete.


## U. CTS calibrated frequency-response preservation — 2026-11-02

120. **The electrostatic and Full-EM CTS branches silently renormalized the supplied measured frequency-response matrix.** This violated the same fixed-calibrated-operator contract used by optical Thomson inference. **Fixed directly on main:** both electrostatic response application and inversion, and both Full-EM response application and inversion, now require finite non-negative square matrices whose columns already sum to one; accepted measured matrices are used unchanged. The Full-EM uncertainty-propagation routine may still normalize its *sampled perturbed realizations* to enforce the declared non-negative unit-column response-operator constraint; it does not renormalize the base supplied operator. Added direct regression coverage for both private response operators and both public inversion entry points, including input-matrix preservation.

Commits:
- `598e0004e353ada77d4eb28c032aa99ae64dc710` — electrostatic CTS response contract.
- `410e0c00c48771def4b70bc8ddf64b41dc244a17` — Full-EM CTS response contract.
- `06d80a2d5d482cf9fb35db2206150c6d5f206691` — regression tests.

**Experimental validation:** not performed and not claimed. GitHub Actions must pass for the final main head before software-test success is stated.



## V. Calibrated 1-D optical response-kernel boundary handling — 2026-11-02

121. **The 1-D measured-kernel route contradicted the fixed-calibrated-response contract and rejected its own finite-window convolution.** It silently divided the supplied kernel by its total area, then required every truncated edge column of the resulting finite-grid matrix to sum to one. A normalized finite-support kernel naturally loses response outside the recorded wavelength interval, so its edge columns are sub-stochastic; the old code therefore rejected the advertised 1-D kernel input. **Fixed directly on main:** the supplied base kernel must already have unit total area and is preserved unchanged. The finite-window operator omits out-of-window tails without folding them back into measured bins; edge-column sums below one explicitly represent response outside the recorded interval. The 2-D and 3-D measured response-matrix contract remains unchanged: non-negative unit-sum columns, used without renormalization. Response-uncertainty Monte Carlo normalizes only perturbed 1-D kernel realizations to the kernel's unit-area constraint and does not renormalize the finite-window edge columns.

Added regression coverage for base-kernel preservation, finite-window edge losses, rejection of non-unit-area base kernels, and the existing end-to-end kernel fit/uncertainty paths. Updated the Analysis 1 scientific contract and dashboard help text. This is finite measurement-window handling, not instrument calibration, detector correction, or experimental validation.

Commits:
- `9fb8d2763b57f5942864e303eeca4b3d6434f1b9` — inference implementation.
- `8918ff3569c8d486e20bc33a08c3f31bbcff7256` — regression tests.
- `59e2b86bca36949f4cbce397e5096913e10259ea` — scientific contract.
- `2a5dfe37474183678033a4c214ca5b4048e93c39` — dashboard contract.

**Experimental validation:** not performed or claimed. GitHub Actions for the resulting main head must complete before software-test success is stated.


## W. Thomson temperature formula contract and geometry endpoint — 2026-11-02

122. **The standalone Thomson temperature formula helper and formula registry retained two inconsistencies.** The helper accepted the exact backscattering endpoint \u03c0 despite the optical Analysis 1 geometry contract requiring \u03b8 strictly inside (0, \u03c0), and the human-readable formula-registry summary still called its intrinsic fit-width input an “instrument-corrected width.” **Fixed directly on main:** the helper now rejects both endpoints, a regression test covers exactly \u03c0, and both root and packaged formula registries describe the input as the intrinsic plasma Doppler-width parameter inferred by the measured-response-convolved physical forward fit. The formula remains a standalone relation for a parameter inferred by the canonical fit; it does not deconvolve or correct measured spectra.

Commits:
- 2b24e7fa3b9ef4377fcbb7976432147ff65e2b89 — strict angle domain in the formula helper.
- f03a967b62da91b72cc5d19459ee48ba46a96d62 — endpoint regression test.
- 17cd55c3e0bc0be1bf3e2b3e6e22a00dc0e5bb96 and 747688bde16c4fabd1b5195cc4e04335c3ea59af — synchronized formula-registry wording.

No experimental validation was performed or claimed. Software-test status remains unconfirmed until GitHub Actions reports the final main-head result.


## X. Thomson model-literature and Kappa-identifiability documentation synchronization — 2026-11-02

A final documentation cross-check found that the scientific note still described Kappa identifiability as a local Jacobian diagnostic only, although the current implementation also performs a conditional profile-likelihood scan that re-optimizes the five nuisance parameters at each fixed Kappa index. The note now describes both diagnostics and explicitly preserves finite-grid censoring semantics.

The selected three-model optical shortlist is now explicitly tied to recent peer-reviewed work: Carole et al. (2025, *Journal of Instrumentation*, DOI 10.1088/1748-0221/20/07/P07014) for Maxwellian optical Thomson diagnostic modelling; Skolar, Longley & Goodwin (2025, *Physics of Plasmas*, DOI 10.1063/5.0261966) for Kappa/non-Maxwellian kinetic-integral methods; and Sakai et al. (2026, *Journal of Instrumentation*, DOI 10.1088/1748-0221/21/01/C01002) for Monte Carlo incoherent Thomson spectra including relativistic Maxwellian and Kappa distributions. The TIS analytic finite-variance Student-t Kappa projection is explicitly distinguished from the Skolar generalized-dispersion algorithm. These are literature/model-selection references, not experimental validation of TIS.

Direct-main documentation commit: recorded in the GitHub commit history. No source algorithm, model, input, output, calibration step or experimental-validation claim was added or removed.


## K. Projected EVDF background-covariance correction (2026-11-02)

The projected one-dimensional EVDF inverse now accepts optional pointwise or full covariance for the explicitly supplied background, in addition to the existing signal uncertainty/covariance. When signal/background cross-covariance is supplied, both marginal covariance models are required and the full joint block covariance is required to be positive definite. The inverse whitens with the residual covariance

\[
C_{y-b}=C_y+C_b-C_{yb}-C_{yb}^{T}.
\]

Background-only uncertainty is explicitly labelled as conditional on treating signal noise as not supplied; it is not represented as a complete measurement-noise budget. The API implementation, root and packaged diagnostic registries, analysis documentation and direct regression tests were updated on the main branch. This is covariance propagation conditional on researcher-supplied uncertainty inputs; it is not background estimation, detector correction, calibration or experimental validation.

## Y. Rejected-primary-fit dependency gate — 2026-11-02

**Confirmed defect:** the parent optical Thomson workflow invoked the spectral-distribution and projected-EVDF subanalyses even when the selected primary physical fit had been rejected. Both child workflows reuse the primary fit's nuisance-background parameters; those parameters must not be treated as a valid basis after a failed identifiability/fit-acceptance gate.

**Fixed directly on main:** the parent workflow now withholds both dependent child analyses with the explicit status `unavailable_primary_fit_rejected`. It does not access or reuse the rejected fit's background coefficients. The independent public spectral-distribution API remains callable directly on a calibrated measured spectrum with its own explicitly supplied background, as documented. A regression test forces primary-fit rejection and verifies that neither child solver is called.

This is an execution-dependency and result-provenance correction only. No measured data are altered, no physical model or output is invented, and no experimental validation is performed or claimed.


## Z. CTS response-operator and spectral-distribution covariance metadata synchronization — 2026-11-02

123. **The CTS registry still described the measured frequency-response matrix as being normalized by the forward model, although both electrostatic and Full-EM implementations now reject a non-unit-column base matrix and apply an accepted calibrated matrix unchanged.** This contradicted the fixed calibrated-operator rule and the 2026-11-02 response-preservation correction. **Fixed directly on main:** both canonical JSON registry copies now require a finite, non-negative square response matrix whose columns already sum to one and state that the supplied base operator is used unchanged. The response-uncertainty description distinguishes the unchanged base matrix from normalized perturbed Monte Carlo realizations.

124. **The Thomson spectral-distribution registry retained a stale statement that signal/background cross-covariance was not represented, despite the implemented `signal_background_cross_covariance` input and joint block-covariance propagation.** **Fixed directly on main:** both canonical JSON copies now describe the separate cross-covariance input and its dependence on the supplied signal and background marginal covariance models.

Added a registry/API regression test that checks root/package parity, verifies the fixed CTS response contract against the electrostatic response-application behavior (including preservation and rejection), and checks the spectral-distribution covariance metadata. These are execution-contract/documentation corrections only. No physical model, input, output, calibration step, measurement, or experimental-validation claim was added or removed.
