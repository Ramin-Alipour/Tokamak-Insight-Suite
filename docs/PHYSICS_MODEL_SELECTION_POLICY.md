# Physics-model selection policy for the 80 diagnostic analyses

## Project-wide rule

This policy applies to all diagnostic analyses registered in `DIAGNOSTIC_REGISTRY.json`.

1. Each diagnostic must infer physical quantities from its declared measured input data, using a physically stated forward model or inversion relation appropriate for that diagnostic.
2. Select no more than 2–3 principal physical models per subanalysis. Prefer models that are widely used, physically interpretable, and supported by recent peer-reviewed primary literature (normally 2023–2026 at the time of implementation).
3. The shortlist is diagnostic-specific. Do not force one model family across unrelated diagnostics, and do not add a model merely to reach a count of three.
4. Distinguish physical forward models from numerical optimizers, statistical estimators, and machine-learning regressors. An optimizer (least squares, Bayesian sampling, regularization) is not itself a plasma-physics model.
5. State each model's assumptions, validity regime, required measured inputs and fixed instrument/device parameters, fitted parameters, outputs, uncertainty treatment, and known degeneracies/limitations.
6. Do not silently invent missing physical constants, geometry, species composition, instrument response, boundary conditions, or calibration data. If a required quantity is not part of the supplied calibrated data or documented instrument metadata, stop and request it.
7. Do not claim experimental validation unless independent experimental validation has actually been performed and documented.
8. Keep model selection and model implementation traceable to the cited original journal papers. Prefer original research papers and authoritative reviews over secondary summaries. Record DOI, journal, year, and the specific model feature supported by each source.
9. Implement only models whose assumptions match the diagnostic and requested output. Optional advanced models must be explicitly selected or triggered by a justified physical regime; never silently switch physical assumptions.
10. Apply this policy progressively to all 80 registry entries. A registry entry is not considered physically complete merely because its software pipeline executes.
11. The default input contract is measured, prepared and calibrated diagnostic data. Do not add detector calibration, raw-count correction, instrument adjustment or independent experimental validation to an analysis unless that work is separately authorized.

## Thomson scattering (Analysis 1): optical model shortlist

Analysis 1's optical Thomson workflow starts from measured, wavelength-calibrated and intensity-calibrated spectra. It has exactly three explicitly selected non-collective physical forward models:

1. **Non-collective Maxwellian** — the standard single-particle Maxwellian model for the ordinary non-relativistic regime.
2. **Non-collective Kappa** — a finite-variance parameterized non-Maxwellian electron distribution for spectra where a Kappa representation is physically justified and identifiable.
3. **Non-collective relativistic Maxwell-Juttner** — exact relativistic photon-frequency kinematics for high-temperature Maxwell-Juttner electrons within the Thomson limit.

These models are not switched automatically. The non-relativistic Maxwellian/Kappa validity limits and the relativistic model's isotropy, flow-projection and recoil limitations are stated in the Analysis 1 documentation. Optical collective dielectric models are not part of this three-model shortlist. Microwave collective Thomson scattering is a separate registered CTS subanalysis and has its own model selection.

Recent peer-reviewed sources informing the optical shortlist:

- M. Carole et al., “A photonic model of the High-Resolution Thomson Scattering diagnostic of WEST tokamak,” *Journal of Instrumentation* **20** (2025), P07014, DOI: [10.1088/1748-0221/20/07/P07014](https://doi.org/10.1088/1748-0221/20/07/P07014). Supports optical Thomson forward modelling and parameter inference.
- C. R. Skolar, W. J. Longley and L. V. Goodwin, “Computing the generalized plasma dispersion function for non-Maxwellian plasmas, with applications to Thomson scattering,” *Physics of Plasmas* **32** (2025), 043907, DOI: [10.1063/5.0261966](https://doi.org/10.1063/5.0261966). Supports generalized non-Maxwellian spectral-response calculations. TIS's current optical Kappa fit uses the analytic Student-t projection and must not be described as an implementation of the generalized plasma-dispersion-function algorithm.
- K. Sakai et al., “Monte Carlo simulation method for incoherent Thomson scattering spectra from arbitrary electron distribution functions,” *Journal of Instrumentation* **21** (2026), C01002, DOI: [10.1088/1748-0221/21/01/C01002](https://doi.org/10.1088/1748-0221/21/01/C01002). Provides comparison/reference spectra for arbitrary distributions, including relativistic Maxwellian and Kappa cases; TIS uses deterministic quadrature and does not claim to reproduce the paper's Monte Carlo algorithm.
- T. Kawate and M. Goto, “Spectroscopic Diagnostics of Non-Maxwellian Electron Velocity Distribution Function in the Large Helical Device,” *Journal of Fusion Energy* **44** (2025), 44, DOI: [10.1007/s10894-025-00516-0](https://doi.org/10.1007/s10894-025-00516-0). Supports the relevance of non-Maxwellian electron diagnostics in magnetic-confinement plasmas.
- M. Amarika et al., “ITER Core Plasma Thomson Scattering diagnostic design,” *Fusion Engineering and Design* **203** (2024), 114416, DOI: [10.1016/j.fusengdes.2024.114416](https://doi.org/10.1016/j.fusengdes.2024.114416). Documents ITER core Thomson measurements of electron temperature and density over 0.5–40 keV.

## Collective Thomson / microwave CTS subanalysis

The registered CTS subanalysis is separate from optical Analysis 1 and exposes exactly two physical forward-model choices:

1. **Unmagnetized electrostatic longitudinal CTS** — a multi-species longitudinal dielectric-response model with explicit ion composition, flow projections and measured frequency-response operator.
2. **Magnetized full-electromagnetic CTS** — a kinetic electromagnetic model with supplied magnetic field, vector geometry/polarization and species-resolved plasma inputs; optional relativistic gyrotropic populations and frequency-resolved ray/beam-overlap geometry are explicit model inputs.

The selected model is mandatory. The software never silently substitutes the electrostatic model for the magnetized full-EM model. Differential evolution, improved genetic search, local least squares and regularization are inversion strategies, not additional physical models.

Recent peer-reviewed sources informing the CTS model scope:

- J. Zhang et al., “Spectral decomposition for collective Thomson scattering based on an improved genetic algorithm,” *Review of Scientific Instruments* **95** (2024), 093542, DOI: [10.1063/5.0215734](https://doi.org/10.1063/5.0215734). Supports CTS spectral decomposition and ion-temperature sensitivity; the genetic algorithm is an optimizer, not the scattering theory.
- T. Verdier et al., “Robust monitoring of thermal and fast ions using collective Thomson scattering: Combining physics- and data-driven background estimation,” *Review of Scientific Instruments* **96** (2025), 043513, DOI: [10.1063/5.0238948](https://doi.org/10.1063/5.0238948). Supports contemporary CTS background characterization; automatic background estimation remains outside the calibrated-input contract.
- X. Fang et al., “Study on the Full Electromagnetic Model of Collective Thomson Scattering in Magnetic Confinement Fusion,” *Power Generation Technology* **46**(1) (2025), 200–208, DOI: [10.12096/j.2096-4528.pgt.24149](https://doi.org/10.12096/j.2096-4528.pgt.24149). Supports the magnetized full-electromagnetic CTS model scope.
- M. Nishiura et al., “Development and Application of Collective Thomson Scattering Diagnostics for Energetic and Bulk Ion Measurements in the Large Helical Device,” *Journal of Fusion Energy* **45** (2026), 55, DOI: [10.1007/s10894-026-00595-7](https://doi.org/10.1007/s10894-026-00595-7). Supports contemporary CTS kinetic inference for bulk and energetic ions.
- J. L. Flocken, S. B. Korsholm and J. Rasmussen, “Accuracy-driven optimisation procedure for collective Thomson scattering diagnostics for fusion plasmas,” *Fusion Engineering and Design* (2026), 115845, DOI: [10.1016/j.fusengdes.2026.115845](https://doi.org/10.1016/j.fusengdes.2026.115845). Supports current forward modelling, ray tracing and Bayesian spectral inversion.

The CTS implementation's remaining limits are stated in `docs/analyses/collective_thomson_scattering.md`; no experimental validation is claimed.
