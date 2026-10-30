# Physics-model selection policy for the 80 diagnostic analyses

## Project-wide rule

This policy applies to all diagnostic analyses registered in `DIAGNOSTIC_REGISTRY.json`.

1. Each diagnostic must infer physical quantities from its declared measured input data, using a physically stated forward model or inversion relation appropriate to that diagnostic.
2. Select no more than 2–3 principal physical models per diagnostic. Prefer models that are widely used, physically interpretable, and supported by recent peer-reviewed primary literature (normally 2023–2026 at the time of implementation).
3. The shortlist is diagnostic-specific. Do not force one model family across unrelated diagnostics, and do not add a model merely to reach a count of three.
4. Distinguish physical forward models from numerical optimizers, statistical estimators, and machine-learning regressors. An optimizer (least squares, Bayesian sampling, regularization) is not itself a plasma-physics model.
5. State each model's assumptions, validity regime, required measured inputs and fixed instrument/device parameters, fitted parameters, outputs, uncertainty treatment, and known degeneracies/limitations.
6. Do not silently invent missing physical constants, geometry, species composition, instrument response, boundary conditions, or calibration data. If a required quantity is not part of the supplied calibrated data or documented instrument metadata, stop and request it.
7. Do not claim experimental validation unless independent experimental validation has actually been performed and documented.
8. Keep model selection and model implementation traceable to the cited original journal papers. Prefer original research papers and authoritative reviews over secondary summaries. Record DOI, journal, year, and the specific model feature supported by each source.
9. Implement only models whose assumptions match the diagnostic and requested output. Optional advanced models must be explicitly selected or triggered by a justified physical regime; never silently switch physical assumptions.
10. Apply this policy progressively to all 80 registry entries. A registry entry is not considered physically complete merely because its software pipeline executes.

## Thomson scattering (Analysis 1): initial literature-backed model shortlist

For measured, calibrated Thomson-scattering spectra, the core physical distinction is the scattering regime. The candidate forward-model families are:

- **Incoherent / non-collective Thomson scattering**: use the single-electron scattering spectral distribution for the declared electron velocity distribution; the Maxwellian form is the standard baseline when a Maxwellian electron population is physically justified. Infer electron temperature primarily from spectral width/shape; absolute density inference requires the documented scattering geometry and absolute optical/detector response or a valid normalization.
- **Collective Thomson scattering**: use the plasma dielectric-response/dynamic structure-factor formulation when the scattering parameter and experimental geometry place the measurement in the collective regime. The spectrum can contain electron-plasma-wave and ion-acoustic features; inference may depend on electron and ion distributions and composition.
- **Non-Maxwellian distribution extension**: only add when justified by the plasma regime and supported by identifiable spectral information. A super-Gaussian or other parameterized distribution is an extension to the distribution function within the appropriate scattering forward model, not a replacement for deciding whether scattering is collective or non-collective.

The first two are the primary regime-dependent model families; the third is conditional, not mandatory. The numerical fitting method (for example weighted least squares, regularized inversion, or Bayesian inference) must be documented separately from the physical scattering model.

### Recent journal references for Thomson-scattering model architecture

- M. Carole et al., “A photonic model of the High-Resolution Thomson Scattering diagnostic of WEST tokamak,” *Journal of Instrumentation* **20** (2025), P07014. DOI: [10.1088/1748-0221/20/07/P07014](https://doi.org/10.1088/1748-0221/20/07/P07014). Supports an end-to-end incoherent Thomson-scattering diagnostic forward model, including scattering spectrum, viewing volume, light collection, optical transfer, polychromators, and parameter inference.
- S. Eisenbach et al., “Machine learning analysis of high-repetition-rate two-dimensional Thomson scattering spectra from laser-produced plasmas,” *Journal of Physics D: Applied Physics* **58** (2025), 035202. DOI: [10.1088/1361-6463/ad89d6](https://doi.org/10.1088/1361-6463/ad89d6). Compares inference from spectra in both non-collective and collective regimes with conventional inversion. Machine learning is treated as an inference approach, not as a physical scattering model.
- D. Schaeffer et al., “Transfer learning for analysis of collective and non-collective Thomson scattering spectra,” *Physics of Plasmas* **33** (2026), 013903. DOI: [10.1063/5.0304922](https://doi.org/10.1063/5.0304922). Recent work explicitly addresses both scattering regimes; its neural-network method is not itself a physical forward model.
- B. C. Foo et al., “Regularized inversion of optical Thomson scattering spectra for improved determination of plasma parameter evolution,” *Review of Scientific Instruments* **97** (2026), 103535. DOI: [10.1063/5.0350497](https://doi.org/10.1063/5.0350497). Supports treating regularization as an inversion strategy for ill-conditioned/degenerate spectral fits, not as a distinct scattering theory.

## Implementation status

This file establishes the governing selection and documentation policy. It does **not** claim that all 80 diagnostic-specific model shortlists have already been researched or implemented. Each registry entry must be assessed against its own measured inputs and physics before its 2–3 model shortlist is finalized.
