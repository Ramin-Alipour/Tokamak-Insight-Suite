# Thomson Scattering (Analysis 1) — Comprehensive main-branch audit

**Audit date:** 2026-11-01  
**Reviewed baseline:** `main` at `c07bf416c8c54a9baaffa3e40de987c22fbcd158`  
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
57. Background parameters and their shared systematic uncertainty are not jointly sampled with the moment Monte Carlo.
58. Pointwise-noise Monte Carlo treats wavelength samples as independent; full within-channel covariance is required for correlated spectral noise.
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
101. Full-EM inversion does not propagate measured response-matrix uncertainty.
102. Full-EM inversion does not propagate the full external magnetic-field, geometry, polarization, ray or equilibrium covariance.
103. Full-EM covariance is conditional on fixed external inputs and the selected damping/quadrature/harmonic settings.
104. If no spectral noise/covariance is supplied, its unweighted-fit covariance is residual-scaled and not measurement-noise-calibrated.
105. Positive log-parameters have no default scientific bounds; researcher-supplied bounds are available where needed.
106. Gyro-harmonic and momentum/velocity quadrature convergence is an opt-in numerical assessment, not an automatic physical uncertainty estimate.
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

## J. Items deliberately not changed without a scientific decision

The most important unresolved scope question is the meaning of **“Thomson spectral distribution analysis.”** The current entry 81 is a descriptive wavelength-space moment/shape analysis, not an electron VDF inversion. Turning it into an inferred electron distribution would require selecting a parameterized VDF family (or a regularized non-parametric inverse problem), declaring the information content and identifiability conditions, and specifying what prior/regularization is scientifically acceptable. A single spectrum does not uniquely determine an arbitrary multidimensional VDF. This must not be silently changed.

Likewise, extending full-EM CTS uncertainty propagation, adding spatially regularized joint optical inversion, or adding collision/finite-orbit-width physics changes the computational scope and requires explicit model/parameter choices. These are recorded as development gaps, not falsely reported as already fixed.

**Validation status:** no experimental validation was performed or claimed. GitHub Actions software tests are separate from experimental validation; their final status must be checked against the resulting main-branch commit before calling the code changes software-test-clean.
