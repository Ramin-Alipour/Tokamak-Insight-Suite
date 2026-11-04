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


## AA. Non-uniform wavelength moment quadrature consistency — 2026-11-02

**Confirmed numerical inconsistency:** the spectral-distribution subanalysis calculated positive/negative/signed areas with trapezoidal integration but calculated centroid and central moments with midpoint finite-volume bin weights. On non-uniform calibrated wavelength grids, these are different measures and can produce internally inconsistent area and moment outputs.

**Fixed directly on main:** all wavelength-space areas, centroid and central moments now use the same finite-volume bin-width measure returned by the wavelength-bin-width helper. Added a regression test on a deliberately non-uniform wavelength grid that independently checks area, centroid and variance against that measure.

This change concerns numerical integration of already calibrated measured spectra. It does not calibrate or modify the wavelength axis, alter measured intensities, introduce a physical model, or perform experimental validation.


## AB. Bound-aware data-only identifiability Jacobian — 2026-11-02

**Confirmed numerical robustness gap:** the joint spatial-spectral inverse estimated its data-only Jacobian with central finite differences even when an optimized parameter was at an active physical/numerical bound. The perturbed evaluations could therefore leave the optimizer's declared domain and produce invalid or misleading rank/conditioning diagnostics.

**Fixed directly on main:** the data-only Jacobian now uses central differences in the interior, one-sided differences at active bounds, and zero columns for parameters fixed at identical bounds. A regression test verifies derivative values and confirms that every perturbation remains within the declared bounds. This changes only numerical identifiability diagnostics; it does not alter measured spectra, physical models, calibration, or experimental-validation status.


## P. Removal of undocumented projected-EVDF candidate-shape caps (2026-11-02)

A code review of `tokamak_analysis/thomson_evdf.py` found that the optional parametric comparison silently constrained the Student-t degrees of freedom to `2.05–100` and the super-Gaussian exponent to `1.2–8`. The upper limits and the stricter-than-mathematical lower limits were not supplied by the researcher or justified by the cited physical model. They could bias candidate comparison and make otherwise admissible measured-spectrum fits unavailable.

**Corrected directly on main:** the Student-t candidate now uses only the finite-variance domain \\(\\nu>2\\) and no finite upper cap; the super-Gaussian candidate uses only its normalizability domain \\(p>0\\) and no finite upper cap. The velocity-location and scale bounds remain tied to the explicitly supplied reconstruction grid and its resolution. Added a regression test that inspects all three candidate optimizer bounds and guards against reintroducing undocumented shape-parameter caps. Updated the scientific analysis documentation. This change does not add a physical scattering model, alter measured input data, or claim experimental validation.


## AC. Projected-EVDF non-uniform wavelength quadrature consistency — 2026-11-02

**Confirmed numerical inconsistency:** the projected-EVDF forward operator used full adjacent-interval widths at the first and last wavelength samples, while its interior samples used midpoint finite-volume widths. The optical spectral-distribution analysis uses midpoint cell edges and therefore half-width endpoint bins. On non-uniform calibrated wavelength grids, the EVDF forward model was using a different spectral integration measure and could misweight spectral power near the recorded-window boundaries.

**Fixed directly on main:** added a dedicated finite-volume wavelength-bin helper using midpoint cell edges and half-width endpoint bins, and routed the EVDF latent-spectrum deposition and measured-response convolution through that shared measure. Added regression tests for the exact non-uniform-grid bin widths and rejection of non-monotonic axes. This changes only numerical quadrature of already calibrated measured spectra; it does not calibrate, modify or reorder input data, add a physical model, or perform experimental validation.


## AD. Conditional local Full-EM CTS external-input covariance — 2026-11-02

The Full-EM measured-spectrum inversion now accepts an ordered positive-semidefinite covariance over local magnetic-field components, incident/scattered direction components and real/imaginary components of incident/scattered complex polarization. Reproducible joint Monte Carlo perturbations repeat the complete physical parameter fit and return a separate parameter-covariance contribution with requested/successful draw counts. The API, dashboard, Python runtime registry, both JSON registries, scientific note and regression tests were synchronized directly on `main`.

This capability applies only to local geometry. Frequency-resolved ray/equilibrium/beam covariance, composition covariance and cross-covariance between distinct external covariance sources remain unsupported; the local covariance route rejects ray-resolved geometry. The separate covariance contributions are not asserted to be jointly correlated. No calibration, measured-data modification or experimental validation is performed or claimed.


## AE. Runtime registry background cross-covariance contract synchronization — 2026-11-02

**Confirmed contract inconsistency:** the Thomson spectral-distribution callable, both canonical JSON registries, and the implementation already supported optional `signal_background_cross_covariance` with joint positive-definite block-covariance propagation, but the Python runtime registry still said that signal/background cross-covariance was not represented. This stale statement could mislead API consumers and contradicted the executable contract.

**Fixed directly on main:** updated the Python runtime registry description for `background_signal_covariance` to state that it may be combined with `signal_background_cross_covariance` when the complete joint signal/background block covariance is positive definite. Added a regression assertion to `tests/test_analysis_registry.py` requiring the runtime registry to expose the supported cross-covariance contract.

Commits: `5b288c642a96312d99c8139cb1efc3eceb1c3c50` (runtime registry), `7657bc356da2de29835ac64784c60f85f3ebf326` (regression test). No measured data, physical model, input/output capability, calibration process or experimental-validation claim was changed. GitHub Actions has not yet been used to claim that the new regression passes.


## AF. Two-population relativistic optical Thomson subanalysis — 2026-11-02

The new optional callable tokamak_analysis.thomson_population_mixture.fit_two_population_relativistic_thomson_spectrum fits a measured calibrated optical spectrum with a two-component isotropic Maxwell-Juttner mixture. It returns cold/hot electron temperatures, hot number-density fraction, one shared scattering-vector flow projection, local parameter covariance, fitted spectrum and residual. The measured response operator is applied unchanged. The component temperatures are ordered to prevent label switching, and rank-deficient fits are rejected.

This is a shape-only subanalysis: absolute density is not inferred, and a single view does not reconstruct an unrestricted 3-D or anisotropic EVDF. The model is a low-dimensional parametric approximation motivated by the arbitrary-VDF Monte Carlo forward-modelling direction of Sakai et al. (JINST 21 (2026) C01002; DOI 10.1088/1748-0221/21/01/C01002); it is not the paper's Monte Carlo algorithm. The callable is documented and registered under Analysis 1, with a dedicated dashboard workflow and software regression tests. No experimental validation is performed or claimed.


## AG. Calibrated reference-spectrum PCA nuisance background for electrostatic CTS — 2026-11-02

The electrostatic analyze_cts_spectrum callable now optionally accepts a researcher-supplied ensemble of at least two calibrated background-only reference spectra on the same frequency grid and detector scale as the measured CTS spectrum, together with an explicit PCA component count. The reference mean and selected principal components form a fixed detector-space nuisance basis. At every nonlinear plasma-parameter trial, the CTS amplitude and linear background coefficients are profiled jointly. The physical CTS forward spectrum alone is convolved with the unchanged measured frequency-response operator; the measured spectrum and reference ensemble are not calibrated, baseline-subtracted or otherwise modified.

The result exposes the selected background model, fitted background spectrum, nuisance coefficients and PCA component count. Without a reference ensemble, the prior constant-offset nuisance remains unchanged. The new input is electrostatic-only; the Full-EM branch does not accept it. The implementation is explicitly a reference-spectrum PCA nuisance model, not a reproduction of the physics-informed PCA method of Verdier et al. (Rev. Sci. Instrum. 96 (2025) 043513; DOI 10.1063/5.0238948), which additionally uses CTS-specific modulation/time-series structure. Reference-ensemble uncertainty is not propagated. The callable, dashboard, root/package JSON registries, Python runtime registry, documentation and regression tests were updated directly on main. No experimental validation is performed or claimed.

The final software-test status is determined only by GitHub Actions for the exact resulting main-head commit; no experimental validation is claimed.


## AH. Automatic initialization for the two-population relativistic optical Thomson fit — 2026-11-02

The two-population Maxwell–Jüttner shape fit previously required the researcher to supply all four nonlinear optimizer starting values (cold temperature, hot temperature, hot number fraction and common scattering-vector flow projection). This contradicted the project input contract that optimizer seeds should be determined automatically from the measured spectrum.

**Implemented directly on `main`:** when all four optional seeds are omitted, the fit now derives deterministic moment-based starting values from the measured calibrated spectrum and runs three bounded deterministic starts, selecting the successful candidate with the smallest whitened residual sum of squares. The moment-to-temperature relation is used only to initialize the relativistic optimizer; it is not returned as an inferred temperature and does not replace the relativistic Maxwell–Jüttner forward model. Partial manual seed sets are rejected; all four may be supplied together for reproducibility. The result explicitly records initialization provenance, multistart count and that seeds are not inferred outputs.

The Streamlit workflow now makes all four seed fields optional, explains the all-or-none rule and invokes automatic initialization when they are blank. The Python runtime registry describes the same contract, and regression tests cover automatic initialization and registry metadata. The model remains a shape-only two-isotropic-population parameterization with one shared projected flow; absolute density and unrestricted 3-D EVDF remain non-identifiable/not inferred. The change does not add calibration or experimental validation.

Commits: `c17de1eb6756cd2388308ad8859ee649e9776748` (fit implementation), `ba91377df25d45fdfeff82f766feb5565038ca8e` (fit regression), `49256fde60415091c1e7d4210c97c090e886492b` (dashboard), `730d95f6268c8d34d9fb7bc5d6dda35c084bbc74` (runtime registry), `58fb0e7bca6f635208bb50eb1edaa80b24ef0da2` (registry regression), `01abf52d80fad988e0333be73c21586076d63b78` (automatic-initialization regression uses the model's minimum supported quadrature orders).

GitHub Actions were not triggered; software-test success is not claimed. Experimental validation was not performed or claimed.


## AI. Relativistic optical Thomson full within-channel spectral covariance — 2026-11-02

**Confirmed integration gap:** the relativistic Maxwell-Juttner optical fit accepted only pointwise uncertainty, while the Maxwellian and Kappa paths accepted full channel-major GLS covariance. This left the three optical branches with inconsistent spectral-noise contracts.

**Implemented directly on main:** `fit_relativistic_maxwell_juttner` now accepts either a positive pointwise 1-sigma vector or a symmetric positive-definite wavelength-by-wavelength covariance matrix. For a supplied covariance it scales the covariance consistently with the signal normalization, Cholesky-whitens the residual, computes the local parameter covariance from the whitened Jacobian, and reports GLS-consistent reduced chi-square and explicit covariance semantics. The primary multi-channel inference accepts a full channel-major covariance for the relativistic model only when it is block diagonal by channel, extracts each wavelength covariance block, and applies GLS independently to each channel. Non-zero cross-channel blocks are explicitly rejected rather than silently discarded. This does not yet implement a jointly coupled relativistic multi-channel fit.

Added regression coverage for full within-channel GLS covariance and rejection of unsupported cross-channel covariance. The Python runtime registry and both canonical JSON registries now describe the exact supported contract.

Commits:
- `3150785580f5d21af1c0dc17c3c22958d8ec39e9` — relativistic fitter accepts full wavelength covariance.
- `717e8d2a8f46eb2062da17acdb2dcf4d0fc4dd7a` — primary inference routes block-diagonal channel-major covariance and rejects cross-channel blocks.
- `e8827c3262a3bb92d7995edc7d1047ff6a6e8eca` — regression tests.
- `3562f49908971bda08dc08dddaaa289d823c87f0` — GLS-consistent reduced chi-square.
- `7d29c4b81b322e766bc297f460d9a96809b391da` and `cc26519d1ff7068f76e5495154b7e71bf52276a7` — root and packaged JSON registry synchronization.
- `744ab27515b9f3fb2edf30c3151ea2ec496bf90f` — Python runtime registry synchronization.

**Remaining scope:** cross-channel-correlated relativistic spectra still require a joint relativistic multi-channel forward inversion. The implementation does not claim that local covariance captures model discrepancy, external-parameter uncertainty, or experimental uncertainty not supplied by the researcher. No experimental validation is performed or claimed. The exact final source commit must pass GitHub Actions before software-test success is reported.


## AJ. Reconciliation against the 2026-11-02 requested gap matrix

This section prevents a partially implemented capability or a physical identifiability boundary from being reported as fully closed.

| Requested item | Current disposition on main | Remaining work / scientific boundary |
|---|---|---|
| A1 optical covariance parity | **Partially closed in this change set.** Maxwellian/Kappa retain full channel-major GLS. Relativistic Maxwell-Juttner now supports full within-channel GLS for block-diagonal channel-major covariance. | A non-zero cross-channel covariance matrix still requires a joint relativistic multi-channel forward inversion; it is explicitly rejected. |
| A2 Full-EM external-input covariance | **Partially implemented before this change.** Local magnetic-field, direction and complex-polarization covariance can be propagated by joint Monte Carlo. | Ray-resolved equilibrium, beam-path and propagated beam covariance; composition covariance and cross-source covariance remain unsupported. |
| A3 Joint spatial-spectral model coverage | **Open capability limitation.** The current joint inverse is isotropic non-relativistic Maxwellian with researcher-supplied fixed mixing and spectral response. | Kappa and relativistic joint spatial-spectral inversion require their own forward operators, Jacobians, identifiability tests and uncertainty treatment. Do not relabel the current Maxwellian solver as multi-model. |
| A4 Two-population initialization | **Closed on main.** Automatic moment-based deterministic multi-start initialization is available when all four optional seeds are omitted. | Initialization moments remain seeds only, not inferred physical results. |
| A5 Relativistic projected EVDF | **Open capability limitation.** The non-relativistic EVDF inverse is withheld for Maxwell-Juttner. | A valid extension must invert the exact relativistic frequency ratio with the Lorentz-transformed polarization-dependent differential cross-section and the same measured response; reusing the first-order Doppler kernel is prohibited. |
| A6 Spectral-distribution physical inference | **Not a defect under the declared architecture.** This module is intentionally descriptive wavelength-space statistics and Gaussian shape diagnostics; physical Te/flow inference belongs to the primary forward model, and EVDF remains a separate inverse problem. | Preserve strict labels and result separation. Do not infer Te or Maxwellianity from Gaussian moments alone. |
| A7 Optical background flexibility | **Partially covered.** The primary Maxwellian/Kappa baseline remains intercept+slope; the spectral-distribution subanalysis supports explicit background representations and covariance; electrostatic CTS separately supports fixed reference-spectrum PCA nuisance bases. | General curved, structured and line-contaminated optical background bases are not yet a common, model-consistent nuisance framework across Maxwellian, Kappa and relativistic fits. |
| A8 Profile covariance and coordinate uncertainty | **Partially covered.** GLS smoothing and channel covariance paths exist when supplied/derived covariance is available. | Coordinate uncertainty and its covariance, and uncertainty propagation through all interpolation/profile representations, remain incomplete. |
| A9 Optical mixing/PSF in ordinary profile reconstruction | **Partially covered.** The separate joint spatial-spectral path accepts a fixed researcher-supplied mixing operator. | The ordinary parameter-profile reconstruction does not infer or automatically apply a PSF/mixing operator. Mixing must remain supplied fixed metadata; no PSF should be invented by the software. |
| A10 Electrostatic CTS external covariance | **Partially covered.** The electrostatic CTS API supports declared external-parameter uncertainty/covariance for its implemented ordered parameter set. | This is not covariance for every possible species/composition/geometry/model input. Unsupported inputs and correlations must remain explicit rather than being implied by a generic "full covariance" label. |
| A11 Two-population registry/dashboard/API | **Closed for the current two-population callable.** The callable, runtime registry, canonical JSON registries, dashboard and regression coverage have been synchronized. | Any future new mixture model must follow the same end-to-end contract and registry parity tests. |
| A12 Spatially varying Full-EM CTS plasma | **Open capability limitation.** Current Full-EM supports path-dependent geometry/beam weighting but retains its documented plasma-parameter uniformity assumptions for the sampled volume. | A profile-aware scattering-volume integral must evaluate the kinetic response and fluctuation covariance at each quadrature location and propagate profile/geometry covariance; a weighted average of local spectra is not interchangeable with evaluating the model at average parameters. |

The requested B items are not all software defects. Single-view 3-D velocity recovery, absolute density from normalized shape, arbitrary anisotropy, unresolved spatial scales, and unrestricted ion-composition inference are identifiability limits. Ray diffraction, quasi-optical propagation, collisional corrections and finite-orbit-width physics are omitted model physics requiring explicit additional forward solvers and independently testable assumptions. No fabricated outputs are added to make these items appear resolved.

The C-series contract work is distributed across the existing machine-readable result classification, fit provenance, optimizer-seed metadata, covariance semantics, response/measurement input contracts, synchronized root/package registries, dashboard regressions and audit history. This matrix is a reconciliation of the requested list against the current source; it is not a claim that all A/B/C items are closed.

**Validation:** the source, tests and registries were updated on main. The GitHub combined commit-status endpoint returned no status entries for the final commit at the time of this audit; therefore this audit does not claim that the full test suite passed. No experimental validation was performed or claimed.


---

## Implementation update — 2026-11-02

### Implemented on `main`

The relativistic non-collective optical Thomson branch now includes a dedicated
multi-channel generalized least-squares API:

- `tokamak_analysis.thomson_kinetic_models.fit_relativistic_maxwell_juttner_channels`
- Full positive-definite channel-major covariance of shape
  `(n_channels*n_wavelengths, n_channels*n_wavelengths)`, including non-zero
  cross-channel blocks, is Cholesky-whitened and used in the objective.
- Electron temperature and scattering-vector-parallel bulk flow are shared
  plasma parameters; each detector channel retains independent positive
  amplitude, constant offset and linear spectral-background slope.
- The result returns the complete local parameter covariance, weighted
  residual sum of squares, scaled-Jacobian rank/condition diagnostics,
  per-channel fitted spectra and an explicit covariance-order declaration.
- Rank-deficient or severely ill-conditioned fits are rejected rather than
  reported as identified.
- Density is explicitly not inferred from normalized spectral shapes. Absolute
  density still requires independently calibrated spectral power and the
  complete optical/collection calibration chain.
- The primary Analysis 1 registry now links the new callable and distinguishes
  this shared-parameter multi-channel API from the existing channel-local
  relativistic fitting route.

Regression tests were added in
`tests/test_thomson_mj_multichannel.py` for synthetic parameter recovery,
cross-channel covariance use, output covariance dimensions and rejection of
non-positive-definite covariance.

### Scientific scope and remaining work

This implementation closes the specific limitation that the existing
relativistic Maxwell–Jüttner fitter could only use channel-local spectral
covariance. It does **not** claim to solve the other independent gaps in this
audit. In particular, arbitrary anisotropic/relativistic EVDF inversion,
structured-background inference, jointly estimated spatial PSF, full
electromagnetic CTS ray/beam propagation and mode conversion, finite-orbit
width/collisional CTS corrections, and independent experimental validation
remain separate tasks. The new function assumes common plasma parameters and
geometry across channels; it is not a general multi-population or
channel-dependent-distribution solver.

### Recent methodological references

1. Sakai et al., “Monte Carlo simulation method for incoherent Thomson
   scattering spectra from arbitrary electron distribution functions,”
   *Journal of Instrumentation* **21** (2026), C01002.
   DOI: [10.1088/1748-0221/21/01/C01002](https://doi.org/10.1088/1748-0221/21/01/C01002).
   Relevant to arbitrary-EVDF forward modelling; it motivates a separate
   distribution-aware kernel rather than relabelling the present Maxwell–
   Jüttner model as an arbitrary-EVDF solver.

2. Skolar, Longley & Goodwin, “Computing the generalized plasma dispersion
   function for non-Maxwellian plasmas, with applications to Thomson
   scattering,” *Physics of Plasmas* **32** (2025), 043907.
   DOI: [10.1063/5.0261966](https://doi.org/10.1063/5.0261966).
   Relevant to non-Maxwellian dielectric response and collective scattering.

3. Carole et al., “A photonic model of the High-Resolution Thomson Scattering
   diagnostic of WEST tokamak,” *Journal of Instrumentation* **20** (2025),
   P07014. DOI:
   [10.1088/1748-0221/20/07/P07014](https://doi.org/10.1088/1748-0221/20/07/P07014).
   Relevant to instrument/photonic response modelling and the distinction
   between a plasma spectral model and a diagnostic forward model.

**Verification status:** code and tests were committed to `main`; no GitHub
Actions workflow run was returned for the resulting commits at the time of
this update. Therefore test execution and CI status are pending verification.

## Implementation update — 2026-11-03

### Full-forward ray/equilibrium/beam uncertainty propagation — API building block

Added tokamak_analysis/cts_geometry_uncertainty.py with
propagate_full_em_cts_ray_geometry_uncertainty.

Unlike a post-fit covariance inflation, this routine performs a fresh
Full-EM spectral inversion for every geometry realization. Its sampler
contract requires each realization to regenerate the equilibrium, retrace
incident and receiver rays, resupply the local beam-covariance realization,
and recompute scattering-volume locations, path-length quadrature and
Gaussian overlap. The routine then calculates the sample covariance of the
refitted physical parameter vector, retaining cross-parameter covariance,
and combines it with the baseline conditional local-fit covariance.

The routine rejects a local mean-geometry-only realization and reports failed
draws, requested/successful sample counts, deterministic seed, parameter
ordering, and an explicit conditional uncertainty interpretation. It refuses
to combine this independent ensemble with the existing local
external_input_covariance path because doing so would lose cross-source
correlations.

Regression coverage was added in tests/test_cts_geometry_uncertainty.py
for successful full-geometry ensemble propagation, rejection of local-only
geometry, and input validation.

**Scope boundary:** this is an API-level propagation building block, not a
claim that the Streamlit dashboard, canonical registries, or all serialized
project workflows already expose a geometry-sampler interface. The sampler
must be supplied by the calling research workflow and must actually rerun
the ray/overlap calculation for each draw. Diffraction-aware propagation,
quasi-optical beam evolution, and profile-varying plasma response remain
outside this implementation. A2 is therefore **partially addressed**, not
closed.

**Verification:** the new module and regression tests were committed to
main. A local Python test run and a completed GitHub Actions run have not
yet been observed; no passing-test or experimental-validation claim is made.

## Implementation update — 2026-11-03 (joint optical model and recoil integration)

### A3 — joint spatial-spectral model coverage

The joint spatial-spectral callable already contained Maxwellian, finite-variance
Kappa and relativistic Maxwell-Juttner forward branches. The workflow dispatcher
and dashboard had not exposed that full callable contract: the dashboard disabled
joint mode for Kappa/relativistic selections, and the workflow rejected the
relativistic model. The dispatcher and dashboard now expose all three registered
models. Kappa mode explicitly requests an index seed and whether that index is
fitted or held fixed. Relativistic mode requires the incident-polarization vector
and retains its quadrature controls. This does not add arbitrary EVDF inversion,
anisotropy inference, or full spatially varying geometry.

### A5 — Compton/recoil selection parity

The joint relativistic spatial-spectral forward path now accepts
include_compton_recoil and passes it to the same relativistic Maxwell-Juttner
single-scattering kernel used by the channel-wise path. The maximum no-recoil
fractional correction remains a screening/error-budget control when the
elastic Thomson limit is selected. The dashboard passes the user's explicit
choice, and the runtime, canonical JSON and packaged JSON registries describe
the same model-selection semantics. Recoil-enabled mode means single-scattering
Compton kinematics and polarized Klein-Nishina weighting; it does not mean
multiple scattering or unrestricted relativistic EVDF inversion.

### A7 — structured backgrounds in joint Kappa/relativistic fits

The joint spatial-spectral optimizer already fits the selected detector-space
background basis simultaneously with the plasma spectral model. With the
dispatcher/dashboard change, constant, linear, polynomial, cubic-spline and
researcher-supplied template backgrounds can now be used in the joint Kappa and
relativistic Maxwell-Juttner paths as well as Maxwellian. Template columns remain
detector-space nuisance components, not plasma emission. The separate
channel-wise Kappa and relativistic fit routes still retain their existing
linear-background contract; A7 is therefore **partially addressed**, not closed
for every optical fit path.

### A2 — geometry uncertainty

The full-forward Monte-Carlo API building block added above remains partial:
a caller-supplied sampler must regenerate equilibrium and beam-covariance
realizations and rerun ray tracing/overlap. No dashboard sampler or joint
cross-covariance model between geometry and other external inputs is claimed.

### A12 — profile-aware Full-EM CTS

Still open. Full-EM CTS continues to fit a spatially uniform set of plasma
parameters over the sampled scattering volume while allowing local ray geometry,
magnetic field and relative overlap to vary. The new work does not implement
simultaneous ne(r), Te(r), species-resolved Ti(r) inversion with full profile,
geometry and cross-profile covariance. A weighted sum of local spectra is not
a substitute for a profile-aware nonlinear kinetic forward model. Optical
Thomson spectra also do not independently identify ion temperature; Ti belongs
to the CTS inference path and requires its own identifiable spectral response.

### Registry and verification

The root and packaged DIAGNOSTIC_REGISTRY.json files were synchronized after
updating the joint spatial model and recoil contracts. The Python runtime
registry, dashboard and workflow dispatcher were updated. Regression coverage
was added for the workflow's relativistic-model and Compton-flag forwarding,
and for the new Full-EM geometry-ensemble uncertainty API.

The changes are committed to main. The repository test suite has not been run
in a local checkout during this session. GitHub combined commit status returned
no status entries for the latest direct-main commit; therefore CI success is
not claimed. No independent experimental validation is claimed.


## Implementation update — 2026-11-03: strict calibrated-input boundary

The previous Analysis 1 registry exposed `fit_optical_thomson_poisson_counts`, which accepted raw integer photon counts. Although it was separate from the calibrated-power workflow and did not infer absolute density, its presence contradicted the project-wide rule that every physical-analysis entry point starts from measured data already prepared and calibrated by the instrument/operator. The public function, its private Poisson-likelihood branch in the Maxwellian fitter, and both root/package registry references and claims have now been removed directly on `main`. A regression test asserts that the raw-count entry point is not exposed. The canonical Gaussian/GLS calibrated-spectrum path is unchanged. This removes an out-of-contract input path; it does not add detector calibration, count conversion, or experimental validation.

### Reconciliation of the earlier T01–T20 list

The earlier list mixed confirmed code defects, incomplete integration, scientific-model boundaries and quantities that cannot be uniquely identified. It must not be treated as twenty confirmed bugs. The repository already contains subsequent direct-main work covering several items: Kappa/MJ structured-background and joint spatial-spectral paths; relativistic multi-channel GLS; Full-EM local external-input covariance; a full-forward CTS geometry-ensemble uncertainty API; response-operator preservation; and synchronized output-identifiability contracts. Remaining integration/model work is recorded explicitly in sections AJ and the 2026-11-03 implementation updates above. In particular, the full-forward geometry sampler is an API building block rather than a dashboard-managed universal covariance model, and profile-aware spatially varying Full-EM CTS remains an unimplemented physics extension. Neither can be honestly marked complete by changing metadata alone.

**Verification:** source, both canonical JSON registries, the input-contract regression test and the scientific/audit notes were updated in one direct-main commit. No local checkout was used. GitHub Actions must report success for the exact resulting commit before claiming software-test success. No experimental validation was performed or claimed.


## Implementation update — 2026-11-03: supplied spatial-coordinate covariance

The downstream measured-spectrum profile reconstruction now accepts
`profile_coordinate_covariance_m2`. The covariance is validated as finite,
symmetric, positive semidefinite and dimensionally consistent with the
channel-coordinate vector. Monte Carlo reconstruction samples the supplied
coordinate covariance jointly in the same draw loop as the profile-value
covariance; non-monotonic coordinate draws and draws that do not cover the
fixed reconstruction grid are rejected. The API applies this option to
optical profiles and species-resolved CTS flow profiles. Coordinate/value
cross-covariance is not currently accepted, so the two sources are explicitly
treated as independent. Omission retains the prior exact-coordinate
assumption. Regression tests cover successful coordinate-uncertainty
propagation and invalid covariance rejection.

This closes only the coordinate-uncertainty propagation gap for a supplied
covariance. It does not infer uncertainty in radius from the spectra, derive
flux coordinates, or perform equilibrium mapping. GitHub Actions status for
the resulting commits remains unverified; no test-pass or experimental-
validation claim is made.

## Implementation update — 2026-11-03: integrated path-resolved CTS thermal background

The Full-EM measured-spectrum inverse now accepts a path-resolved physical
background input contract in addition to the precomputed receiver-grid
background. The new builder integrates the existing non-relativistic
Maxwellian free-free emissivity and LTE radiative transfer along explicitly
supplied path profiles, requires an explicit receiver-mode ECE absorption
coefficient (or explicit null to omit ECE), converts radiance using supplied
receiver etendue and efficiency, and applies the measured frequency response
once. The inversion adds this measured-grid background to the forward spectrum.

An optional receiver-grid physical-background covariance is validated as
symmetric positive semidefinite and added to the independent model-discrepancy
covariance for GLS. The combination assumes independence and does not invent
cross-covariance. The existing fixed-spectrum input remains available and
is mutually exclusive with the path-resolved mapping.

Regression tests cover absolute-frequency construction, physical input
requirements, measured-response convolution and explicit ECE selection.
This closes the integration gap for the implemented free-free/LTE-ECE
background kernel; it does not implement a new ECE absorption solver,
relativistic/magnetized free-free emissivity, or a physical collision operator.
No experimental validation is performed or claimed.

## Direct-main implementation reconciliation — 2026-11-03

This section supersedes the preliminary open/partial labels above where later
main-branch commits changed the implementation. It does not convert declared
model boundaries into completed capabilities.

### Implemented on main

- Full-EM CTS now accepts a path-resolved free-free plus LTE thermal-ECE
  radiative-transfer background, or a precomputed receiver-grid physical
  background, with mutually exclusive inputs.
- The path-resolved route uses absolute receiver frequency
  `probe_frequency_Hz + frequency_shift_Hz`, explicitly supplied path
  profiles, ion composition, Gaunt factor, receiver-mode ECE absorption,
  incident radiance, etendue and efficiency. The measured response is applied
  once.
- A supplied positive-semidefinite receiver-grid physical-background
  covariance is included in GLS under an explicit independence assumption.
- A new main-branch regression module covers free-free density and charge
  scaling, LTE equilibrium invariance, mode-resolved ECE opacity response and
  receiver etendue/efficiency conversion.
- The optical Kappa and relativistic Maxwell-Juttner structured-background
  joint workflow, relativistic projected-EVDF route, multi-view 3-D EVDF
  inverse, generalized Landau-integral route, PSF-aware profile inverse and
  joint profile reconstruction are already registered in the main branch.
  Their documented assumptions remain binding; they are not unrestricted
  arbitrary-distribution inversions.

### Still partial or open — do not mark complete

1. **ECE opacity solver:** the Full-EM background integrator consumes a
   researcher-supplied mode-resolved ECE absorption coefficient. It does not
   calculate the magnetized kinetic ECE opacity itself.
2. **Free-free domain:** the implemented emissivity is non-relativistic and
   unmagnetized with an explicitly supplied Gaunt factor. Relativistic and
   magnetized mode-resolved free-free emission remain outside this kernel.
3. **Physical collisional CTS:** the numerical resonance damping parameter is
   not a collision operator. A conserving BGK/Mermin or other collision model
   must not be represented by inserting a damping rate into the collisionless
   dielectric response. A physically consistent collisional dynamic structure
   factor, including its fluctuation source terms and species-temperature
   treatment, remains unimplemented.
4. **Full-EM propagation:** current cold-plasma ray tracing and local Gaussian
   overlap do not implement diffraction-resolved quasi-optical propagation,
   full-vector non-paraxial propagation or general mode conversion/branch
   tracking through degeneracies.
5. **Profile-aware Full-EM CTS:** scattering-volume quadrature can vary local
   ray geometry and magnetic field, but the fitted bulk plasma parameters are
   still spatially uniform. A node-wise profile inverse with profile/geometry
   covariance remains unimplemented.
6. **Relativistic EVDF scope:** the projected relativistic inverse and
   multi-view 3-D inverse are distinct constrained routes. They do not amount
   to a universal anisotropic, relativistic, polarized, recoil-inclusive
   arbitrary-EVDF inversion.
7. **Optical background parity:** the Maxwellian channel-wise route and the
   separate joint Kappa/MJ background workflow have different contracts.
   Structured-background capability must not be claimed for every optimizer
   unless that exact route exposes and tests it.
8. **Covariance completeness:** available channel, response, external-input,
   coordinate and background covariance routes remain conditional on the
   covariance actually supplied. Unprovided cross-covariances are not
   inferred. In particular, coordinate/value cross-covariance and
   cross-channel model-discrepancy covariance are not universally supported.
9. **Geometry uncertainty integration:** the full-forward geometry ensemble
   remains a caller-supplied sampler API building block, not a universal
   dashboard-managed equilibrium/beam covariance solver.
10. **Verification:** the newly added deterministic tests are software
    regression tests only. No experimental validation is claimed. GitHub
    Actions must pass on the exact resulting main commit before reporting a
    successful CI run.

### Scientific implementation rule

Do not close any item above by relabelling a numerical damping parameter as
physical collisions, a supplied ECE opacity as an ECE solver, cold ray tracing
as diffraction-aware propagation, or a uniform-volume fit as a profile
inversion. Those changes require a complete forward model, explicit measured
input contract, parameter-identifiability treatment, numerical regression
coverage and documentation. Where a required model cannot be selected
unambiguously from the cited literature or the supplied diagnostic metadata,
the software must reject that path rather than silently invent an assumption.


## Implementation update — 2026-11-03: joint coordinate/value covariance for profile reconstruction

The downstream spectrum-inferred profile reconstruction now accepts
`profile_coordinate_value_cross_covariance`, the matrix
\(C_{rv}[i,j]=\operatorname{Cov}(r_i,v_j)\), in addition to the supplied
coordinate covariance \(C_r\) and profile-value covariance \(C_v\).
The three blocks are assembled into the joint covariance

\[
C_{joint}=\begin{bmatrix}C_r&C_{rv}\\C_{rv}^{T}&C_v\end{bmatrix}.
\]

The complete block must be finite, symmetric and positive semidefinite.
Coordinate and profile values are sampled together from this block in the
same Monte Carlo draw; the previous independent-marginal route is unchanged
when the cross-covariance is omitted. The cross block is rejected unless both
marginal covariance models are supplied (pointwise profile uncertainties are
accepted as the diagonal value covariance). The joint raw spatial-spectral
optimizer does not accept this downstream reconstruction covariance.

Regression coverage checks a valid correlated block, rejection of a
non-positive-semidefinite block, and rejection when either marginal covariance
is missing. The profile input is declared in the Python execution registry and
both synchronized JSON diagnostic registries. A cross-subanalysis contract
test also checks Analysis 1, profile reconstruction and spectral-distribution
registry wiring and output-identifiability declarations.

This closes TS-09 only for the downstream channel-profile reconstruction API.
It does not infer the cross-covariance, does not propagate it through the raw
joint spatial-spectral optimizer, and does not solve the full Full-EM CTS
profile/geometry covariance problem. The new tests have not been executed in
this environment; no CI-pass or experimental-validation claim is made.


## Implementation update — 2026-11-03: explicit Full-EM resonance-broadening semantics

The Full-EM kinetic response previously exposed `damping_rate_rad_s` as a
numerical control. The value enters the kinetic resonance denominators and is
not a conserving collision operator. To prevent downstream users, dashboard
consumers or exported results from interpreting it as a measured/physical
collision frequency, the Full-EM result now reports:

- `damping_rate_is_physical_collision_frequency = false`;
- `resonance_broadening_semantics =
  numerical_resonance_regularization_in_kinetic_denominators_only`;
- `collisional_cts_model_status = not_implemented`;
- `collision_operator = null`.

The numerical-control output retains whether the value was caller-supplied or
used the documented numerical default. The canonical and packaged diagnostic
registries expose the same result fields, and a regression test guards both
caller-supplied and defaulted paths.

**This is a semantic/API safety correction, not closure of TS-03.** No
collisional CTS spectrum is calculated. A physical implementation must
include the collision-modified susceptibility and its consistent fluctuation
source/dynamic-structure-factor treatment, with species-resolved collision
physics and conservation checks. A scalar imaginary broadening term is not a
substitute.

Relevant model-selection references:
- Chuna & Murillo, *Physical Review E* **111**, 035206 (2025),
  DOI: 10.1103/PhysRevE.111.035206 — multicomponent number- and
  momentum-conserving BGK susceptibility and associated dynamic structure
  factor; the work discusses the frequency f-sum rule and limitations of
  simpler Mermin formulations.
- Zhao et al., *Matter and Radiation at Extremes* **11**, 057401 (2026),
  DOI: 10.1063/5.0290643 — collisional BGK correction applied to collective
  Thomson-scattering spectra and EPW feature modelling.

The new regression test has been committed but was not executed in this
environment. GitHub Actions status is not available for the resulting
contents-API commits; no test-pass or experimental-validation claim is made.


## Implementation update — 2026-11-03: channel-wise structured-background parity and audit reconciliation

The canonical channel-wise non-collective Maxwellian, finite-variance Kappa and relativistic Maxwell-Juttner fitters expose the selected detector-space background basis as part of the physical forward fit. Supported bases are constant, linear, Legendre polynomial, cubic spline and researcher-supplied templates. Background coefficients are fitted jointly with the response-convolved plasma spectrum and retained in the local parameter covariance, including physical/background cross terms. For Kappa, the shape-identifiability Jacobian projects against every fitted background nuisance coefficient.

For the Kappa model, the one-dimensional profile-likelihood diagnostic is available for the linear background parameterization. For higher-order or template bases, the routine explicitly reports that the existing linear-only profile-likelihood calculation is not applicable and retains joint covariance/Jacobian identifiability diagnostics. This limitation concerns that optional profile-likelihood diagnostic, not structured-background fitting itself.

The relativistic Maxwell-Juttner channel-wise path first obtains a deterministic seed from the established relativistic fitter and then calls `fit_noncollective_kinetic_thomson_with_joint_background` with the selected background model, degree, spline knots/templates, and the supplied measurement covariance. Regression coverage in `tests/test_thomson_inference.py::test_relativistic_channel_fit_supports_joint_polynomial_detector_background` exercises polynomial background fitting and verifies that the joint covariance includes background cross terms. The Streamlit workflow and canonical registries expose the selected basis contract.

This supersedes the earlier sentence in this audit claiming that the relativistic Maxwell-Juttner channel-wise route retained only a linear baseline. That sentence was stale relative to the current implementation and test suite. The separate limitation on profile-likelihood diagnostics for structured Kappa backgrounds remains.

**Verification boundary:** source and regression coverage are present on `main`; this audit session has not executed the regression or full Analysis 1 test suite. No passing-CI or experimental-validation claim is made.


## Direct-main implementation update — 2026-11-04: Full-EM physical-background state parity (TS-11)

The Full-EM CTS inversion previously distinguished an omitted physical background from an explicit zero through the lower-level physical-background fields, but did not expose the same top-level background-state contract as the electrostatic CTS path. This left consumers liable to conflate the profiled detector-space constant offset with a physically known zero radiative background.

**Fixed directly on main:** the Full-EM result now returns `background_state`, `background_state_is_explicit`, `background_zero_assumption`, `background_state_provenance`, and `physical_background_zero`. If neither a receiver-grid physical background nor a path-resolved radiative-transfer model is supplied, the state is `uncharacterized_constant_offset`, explicit is false, the physical zero flag is null, and the provenance states that the physical background is unknown rather than assumed zero. If a physical background is supplied/modelled, it is distinguished from the separately profiled constant offset. `background_zero_assumption` remains false: an explicitly supplied zero spectrum is a measured/model input, not an assumption.

Added regression coverage for both omitted and explicitly zero receiver-grid backgrounds, including the independent profiled offset. Updated the CTS scientific note, Python runtime registry and both canonical JSON registry copies; root/package JSON registries were verified byte-for-byte equivalent after the update.

Commits:
- `afc287bfd50fae097bbd15dd15b450855c4bcf69` — Full-EM result contract.
- `63e62b8b38d4edf4a95d4615d91c6b63f4724a1a` — regression coverage.
- `157af3ef4d4e646f418a2152650e9fb3f8b49f79` — CTS scientific note.
- `cf2d2fa44f2e99e65558d150775cda9dd87871a8` — Python runtime registry.
- `a22c1456452d4ef68f4f4435420bcfc5e4a1181f` and `8e1f149fc93ec7ee9f6df4d8b7774c19300295cd` — root/package JSON registries.

**Scope:** this closes the Full-EM omission-versus-zero reporting inconsistency within TS-11. It does not calculate missing Bremsstrahlung/ECE components, infer unprovided background covariance, or claim a complete physical-background model. The remaining TS items are not marked complete by this metadata/state correction.

**Verification:** the source, tests, documentation and registries are present on `main`. GitHub combined commit status returned no status entries for the resulting registry commit; this update therefore does not claim that the regression or full test suite has passed. No experimental validation is performed or claimed.

**TS-20 traceability correction (2026-11-04):** the canonical `docs/audits/cts_ts01_ts20_implementation_status.md` ledger defines TS-20 as the cross-subanalysis contract and regression matrix. The earlier statement that its subject was unspecified was stale and is superseded. The Analysis-1 cross-contract tests cover the four public family routes (optical Thomson, CTS, spatial-profile reconstruction and descriptive spectral distribution), executable registry wiring, output-identifiability producers, CTS dispatcher outputs and dashboard exposure. The regression matrix has been extended to enforce byte-identical root/package JSON registries and exact equality between each Analysis-1 diagnostic execution contract and the runtime Python registry. TS-20 is therefore implemented at the contract/test level; it is not a substitute for closure of the physical model gaps TS-01–TS-18. Final status remains pending GitHub Actions on the exact commit.

## O. Additional physical CTS kernel added on 2026-11-04

A local 3-D magnetoionic group-velocity relation is now implemented in
`tokamak_analysis/cts_magnetoionic.py` as
`cold_plasma_mode_group_velocity`. Starting from
`M(ω,k)=ε_r(ω)−(c²/ω²)(k²I−kkᵀ)`, the simple-mode Hellmann–Feynman
derivative is
`v_g,j=−(e†(∂M/∂k_j)e)/(e†(∂M/∂ω)e)`. The code computes the k derivative
analytically and the ω derivative with a symmetric finite difference. It
returns Cartesian group velocity and the ray/wave-normal walk-off angle,
which are distinct in an anisotropic magnetized plasma. The public advanced
physics dispatcher and both formula registries expose the relation.

The validity gate is intentionally strict: collisionless, transparent,
positive-real refractive-index root, isolated simple mode. Collisional,
evanescent and degenerate branches are rejected rather than assigned a
misleading real ray velocity. Regression tests compare parallel propagation
with an independent numerical derivative of k(ω), check oblique walk-off,
and verify rejection of collisions and degeneracies. This is a physical
kernel addition, not completion of arbitrary-3-D inhomogeneous propagation,
ray integration or mode conversion; TS-04/05 remain partial.


The same local magnetoionic kernel additionally returns the Brillouin
dispersive energy density
`W=1/4[ε₀ E†∂(ωε_r)/∂ω E+μ₀ H†H]`, Poynting vector
`S=1/2 Re(E×H*)`, and energy velocity `v_energy=S/W`. The relative
difference between energy velocity and the group velocity is exposed as a
consistency diagnostic and covered by a parallel-branch regression. These
relations are evaluated only for transparent collisionless simple modes;
they are not used to manufacture an energy velocity for dissipative or
degenerate modes.


## Physical implementation update — 2026-11-04: multi-pitch gyrotropic anisotropy inversion

A physical GLS inverse for optical Thomson directional-temperature measurements
is now implemented in
`tokamak_analysis.thomson_kinetic_models.infer_gyrotropic_bi_maxwellian_temperatures`
and exposed through the advanced-physics dispatcher as
`gyrotropic_bi_maxwellian_anisotropy`. For a gyrotropic bi-Maxwellian with
zero heat flux, the one-dimensional projected variance obeys

```text
T_q = T_perp (1 - mu^2) + T_parallel mu^2
mu = b_hat dot q_hat
```

The inverse uses the full supplied covariance of directional temperatures:
`theta = (X.T C_T^-1 X)^-1 X.T C_T^-1 T_q`, with columns
`[1-mu^2, mu^2]`. It returns the two-temperature covariance and GLS residual
diagnostics only when the design matrix has rank two and both fitted
temperatures are positive. Repeated equivalent pitch angles return
`non_identifiable` and null parallel/perpendicular temperatures; no anisotropy
is fabricated from a single projection. The advanced Thomson dashboard now
exposes the model and a self-consistent input-shape example. The root and
packaged formula registries are synchronized.

This closes a bounded, two-temperature gyrotropic inference capability under
its explicit assumptions; it does **not** implement a general anisotropic
EVDF, pitch-angle uncertainty propagation, or a direct raw-spectrum
multi-view optimizer. Directional temperatures must first be obtained from
the calibrated measured-spectrum optical Thomson fit. Regression tests cover
recovery, rank-deficient non-identifiability, covariance validation and
dispatcher/dashboard/formula-registry wiring. CI on the resulting main SHA
and independent experimental validation remain unverified.
