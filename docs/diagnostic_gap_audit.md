# Audit of the 60 Registry-Only Diagnostic Workflows

**Scope:** the 60 entries in `DIAGNOSTIC_REGISTRY.json` at tag `v0.5.1` whose `implementation_status` is exactly `Registry reference only`. This is a triage audit, not a claim that the listed physics is already validated.

## Classification rule

- **A — Completable by extending existing code:** existing generic numerical functions cover a substantial part of the computation; work is mostly a thin workflow, input contract, unit checks, diagnostics and tests. This does not imply the entire diagnostic inference is complete.
- **B — Requires new algorithm/model:** a dedicated numerical or physics model is absent; a generic filter, fitter or inverse solver is not a substitute for the missing physical model.
- **C — Requires device-specific data/calibration:** the workflow depends first on machine geometry, instrument response, calibration, atomic/material data, synchronized channels or a device-specific correction. General code can be added, but meaningful physical outputs must remain gated until those inputs are supplied.

One primary group is assigned per entry based on the principal blocker. Several entries also have secondary blockers. “A” is not equivalent to “validated.”

## Summary

| Group | Count | Interpretation |
|---|---:|---|
| A — Existing code can be extended | 11 | Build wrapper/workflow and add tests around current numerical primitives. |
| B — New algorithm/model | 22 | Implement a dedicated method and validate independently. |
| C — Device data/calibration | 27 | Define a strict calibration/geometry/data contract before enabling physical inference. |
| **Total** | **60** | **All registry-only entries included exactly once.** |

## Item-by-item audit

| # | Diagnostic | Family | Primary classification | Reason and next implementation step |
|---:|---|---|---|---|
| 1 | Thomson scattering | Plasma parameters | Requires new algorithm/model | New scattering-spectrum forward model and fit are required; input response/calibration and laser metadata must be explicit. |
| 2 | Electron cyclotron emission (ECE) | Plasma parameters | Requires new algorithm/model | New frequency-to-resonance mapping and radiative-transfer/optical-depth handling; avoid assuming a universal Te mapping. |
| 3 | Charge-exchange recombination spectroscopy (CXRS) | Plasma parameters | Requires device-specific data/calibration | Needs beam-overlap/viewing geometry, instrument function, atomic rates and calibrated spectra before T_i/rotation can be inferred. |
| 4 | Molecular spectroscopy | Plasma parameters | Requires new algorithm/model | New molecular-band model, line/band database and rotational/vibrational population fitting. |
| 5 | Thomson spectral distribution analysis | Plasma parameters | Requires new algorithm/model | New collective/non-collective scattering model for non-Maxwellian distribution constraints. |
| 6 | Toroidal-field coils / magnetic probes | Magnetic and equilibrium | Requires device-specific data/calibration | Needs device-specific coil geometry, magnetic materials, transfer function and calibration constants. |
| 7 | Poloidal-field coils / magnetic probes | Magnetic and equilibrium | Requires device-specific data/calibration | Needs calibrated probe positions/orientations and coil/equilibrium geometry. |
| 8 | Equilibrium reconstruction (Grad–Shafranov / EFIT-like) | Magnetic and equilibrium | Requires new algorithm/model | Major new constrained Grad–Shafranov solver or robust external-solver interface; not derivable from a flux grid derivative alone. |
| 9 | Polarimetry / Faraday rotation | Magnetic and equilibrium | Requires device-specific data/calibration | Needs wavelength, line-of-sight geometry, polarization calibration and density/field inversion assumptions. |
| 10 | Magnetic flux / poloidal flux | Magnetic and equilibrium | Completable by extending existing code | Integrate calibrated time series; existing integration and parameter utilities can support a bounded flux-change workflow. |
| 11 | Ball-pen probe | Electrostatic probes | Requires device-specific data/calibration | Needs collector geometry/current balance and probe-specific correction validated against a reference. |
| 12 | Gundestrup / compound directional probe | Electrostatic probes | Requires device-specific data/calibration | Needs probe geometry/orientation, electrical channel map and validated sheath/collection model. |
| 13 | Ion saturation current | Electrostatic probes | Requires device-specific data/calibration | Needs effective collection area, bias regime, ion species and regime-appropriate sheath model. |
| 14 | Floating / plasma potential arrays | Electrostatic probes | Requires new algorithm/model | Add spatial geometry-aware gradient/E-field calculations and uncertainty propagation; calibration/timing inputs still required. |
| 15 | Probe-derived density fluctuations | Electrostatic probes | Requires device-specific data/calibration | Needs synchronized calibrated probe channels and a validated mapping from current/voltage to density fluctuations. |
| 16 | Visible / optical emission spectroscopy | Spectroscopy | Completable by extending existing code | Reuse line fitting and calibration functions for centroid/width/intensity; species identification and atomic interpretation remain out of scope. |
| 17 | VUV spectroscopy | Spectroscopy | Requires device-specific data/calibration | Needs calibrated VUV response, wavelength/absolute calibration and atomic collisional-radiative data. |
| 18 | Charge-exchange spectroscopy | Spectroscopy | Requires device-specific data/calibration | Needs neutral population/beam overlap, sightline geometry, instrument response and atomic rates. |
| 19 | Hydrogen Balmer / Balmer-alpha spectroscopy | Spectroscopy | Requires device-specific data/calibration | Needs wavelength/absolute calibration and excitation/recombination model to infer recycling or particle flux. |
| 20 | Spectral wavelength calibration | Spectroscopy | Completable by extending existing code | Reuse linear/polynomial calibration and uncertainty functions; add residual reporting and reference-line inputs. |
| 21 | Instrumental line-shape characterization | Spectroscopy | Completable by extending existing code | Reuse spectral fitting for reference-line width; full non-Gaussian instrument response needs additional work. |
| 22 | Laser / microwave interferometry | Density and refractive diagnostics | Requires device-specific data/calibration | Needs frequency/wavelength, path length, phase convention, fringe-jump handling and calibrated phase signal. |
| 23 | Microwave reflectometry | Density and refractive diagnostics | Requires new algorithm/model | New sweep/phase/group-delay and cutoff/ray interpretation pipeline with mode and propagation checks. |
| 24 | Collective Thomson / coherent scattering (CTS) | Density and refractive diagnostics | Requires new algorithm/model | New scattering geometry and plasma-response forward model; avoid treating spectral features as direct velocity-space measurements. |
| 25 | Refractive-index / ray-tracing diagnostics | Density and refractive diagnostics | Requires new algorithm/model | New ray/beam propagation solver and cutoff/refraction handling. |
| 26 | Fast bolometry | Radiation and imaging | Completable by extending existing code | Reuse time-series QC, filtering, PSD and event tools after response correction is supplied. |
| 27 | Soft X-ray arrays | Radiation and imaging | Requires new algorithm/model | New multi-channel response handling and geometry-aware inversion; generic tomography alone does not identify emissivity physics. |
| 28 | Hard X-ray detector | Radiation and imaging | Completable by extending existing code | Reuse X-ray spectrum summary and calibration utilities for basic counts/peak summaries; dead-time and response unfolding are not covered. |
| 29 | Visible imaging | Radiation and imaging | Requires new algorithm/model | New image ingestion, calibration transforms, segmentation and feature/boundary tracking. |
| 30 | Fast visible imaging | Radiation and imaging | Requires new algorithm/model | New frame-stack ingestion, temporal feature tracking and uncertainty-aware projected-velocity estimates. |
| 31 | IR thermography | Radiation and imaging | Requires device-specific data/calibration | Needs radiometric calibration, emissivity/reflection correction, viewing geometry and material thermal properties. |
| 32 | Bolometric tomography | Radiation and imaging | Completable by extending existing code | Reuse generic tomography matrix builder/regularized solver; registry status should be reconciled with existing functions and test coverage. |
| 33 | Neutron yield monitor | Neutron and nuclear | Requires device-specific data/calibration | Needs efficiency, dead-time, geometry, background and scattering corrections for absolute yield. |
| 34 | Neutron spectrometry | Neutron and nuclear | Requires new algorithm/model | New detector-response forward/unfolding methods with regularization and uncertainty evaluation. |
| 35 | Neutron imaging / tomography | Neutron and nuclear | Requires new algorithm/model | New geometry and response matrix construction plus sparse-view inverse solver and uncertainty diagnostics. |
| 36 | Gamma spectroscopy | Neutron and nuclear | Requires device-specific data/calibration | Needs detector energy/efficiency calibration, background, dead-time and response characterization. |
| 37 | Neutral particle analyzer (NPA) | Fast particles | Requires new algorithm/model | New neutralization/charge-exchange forward model and calibrated energy-response inversion. |
| 38 | Fast-ion D-alpha (FIDA) | Fast particles | Requires new algorithm/model | New atomic-emission and beam-geometry forward model, background separation and phase-space response. |
| 39 | Collective Thomson scattering (CTS) | Fast particles | Requires new algorithm/model | New plasma-response spectral model and fast-ion inference; distinct from generic signal processing. |
| 40 | Fast-ion loss detector (FILD) | Fast particles | Requires new algorithm/model | New scintillator response and orbit-following mapping tied to equilibrium and detector geometry. |
| 41 | Runaway-electron diagnostics | Fast particles | Requires new algorithm/model | New calibrated radiation/transport response model to distinguish proxies from population/energy estimates. |
| 42 | Pressure gauges | Gas, pressure and fueling | Requires device-specific data/calibration | Needs gauge-type calibration, gas-species sensitivity, location and operating range. |
| 43 | Residual gas analyzer (RGA) | Gas, pressure and fueling | Requires device-specific data/calibration | Needs mass sensitivity and cracking-pattern calibration to separate species. |
| 44 | Gas flow meters | Gas, pressure and fueling | Completable by extending existing code | Reuse time-series integration and unit conversion for integrated injected amount once calibration and reference conditions are supplied. |
| 45 | Neutral pressure / recycling estimates | Gas, pressure and fueling | Requires device-specific data/calibration | Needs local neutral/pressure constraints plus a validated neutral transport model; vessel pressure alone is insufficient. |
| 46 | Effective pumping speed | Gas, pressure and fueling | Completable by extending existing code | Add a thin formula wrapper around existing unit/QC utilities: S=Q/p; verify conductance and unit conventions. |
| 47 | MHD mode / Mirnov analysis | MHD and turbulence | Completable by extending existing code | Reuse multi-channel phase/mode-number and PSD tools; geometry-aware mode fit and validity checks need extension. |
| 48 | Particle flux and effective diffusion | Transport and confinement | Requires new algorithm/model | New explicit transport conventions and profile/velocity-gradient combination with uncertainty; needs co-temporal profiles. |
| 49 | Energy balance | Transport and confinement | Requires new algorithm/model | New synchronized time-resolved balance with consistent signs, stored-energy derivative, radiation and net-loss channels. |
| 50 | Neutral beam injection (NBI) power | Heating and RF | Requires device-specific data/calibration | Needs calibrated injected/reflected/attenuated beam diagnostics and deposition model to separate injected from absorbed power. |
| 51 | Electron cyclotron heating (ECRH) | Heating and RF | Requires device-specific data/calibration | Needs directional-coupler calibration, transmission losses and absorption/deposition model. |
| 52 | Ion cyclotron heating (ICRH) | Heating and RF | Requires device-specific data/calibration | Needs calibrated forward/reflected power, matching-network/antenna losses and coupling model. |
| 53 | Lower hybrid heating/current drive | Heating and RF | Requires device-specific data/calibration | Needs calibrated RF channels, launcher/plasma coupling and propagation/current-drive model. |
| 54 | Electron/ion heating power | Heating and RF | Requires new algorithm/model | No matching implementation is registered; implement a dedicated workflow and test it against an independently documented reference dataset. |
| 55 | Thermocouples / embedded sensors | Plasma-facing components | Requires device-specific data/calibration | Needs sensor calibration/dynamics and material heat-transfer model to distinguish sensor temperature from surface temperature. |
| 56 | Surface erosion / deposition | Plasma-facing components | Requires device-specific data/calibration | Needs pre/post-exposure metrology, spatial registration, material/surface metadata and uncertainty protocol. |
| 57 | Wall impurity / dust monitoring | Plasma-facing components | Requires device-specific data/calibration | Needs calibrated imaging/spectroscopy or post-mortem samples with geometry and sampling metadata. |
| 58 | Strike-point / hot-spot tracking | Plasma-facing components | Requires device-specific data/calibration | Needs camera-to-machine coordinate calibration, timing and emissivity/temperature context. |
| 59 | Thermal response / heat diffusion | Plasma-facing components | Requires device-specific data/calibration | Needs material properties, boundary conditions and validated regularized inverse heat-conduction solver. |
| 60 | Channel synchronization | Data quality and calibration | Completable by extending existing code | Reuse cross-correlation lag analysis as an initial offset estimate; add common-trigger/jitter and uncertainty reporting. |

## Recommended development order

1. **A workflows first:** wavelength calibration and residuals; instrument line-shape summary; flux integration; flow integration and pumping-speed formula; fast-bolometry time-series workflow; channel synchronization diagnostics; MHD mode fit; X-ray summary extension; bolometric tomography registry reconciliation. Each needs unit tests and explicit assumptions.
2. **Low-risk B workflows next:** image/frame ingestion and feature tracking; profile/energy-balance bookkeeping; spectral/model wrappers only after input schema is explicit. Do not silently infer calibrated physical units from arbitrary CSV columns.
3. **C input contracts in parallel:** define reusable schemas for calibration curves, detector response, line-of-sight geometry, timing alignment, atomic/material metadata, and provenance. Missing inputs must yield a clear “not computable” status, not a plausible-looking number.
4. **High-complexity physics later:** Grad–Shafranov equilibrium, Thomson/ECE/CXRS/CTS/FIDA/NPA, neutron unfolding/tomography, reflectometry/ray tracing, inverse heat conduction. These require model-specific benchmark datasets and uncertainty tests.

## Acceptance criteria for each promoted workflow

- Inputs, units, time base, calibration and geometry are explicit and validated.
- Directly measured quantities are distinguished from derived estimates and model-dependent inferences.
- Tests include synthetic unit/limit cases and, where available, real raw data paired with an independently reported result from the same data.
- Numerical results carry uncertainty/quality flags and provenance.
- If a raw dataset plus matching independent reference result cannot be found, mark the validation as **pending** and continue to the next workflow; do not label it validated.
- Update the registry status only after implementation tests pass; scientific validation is tracked separately from code execution.

## Evidence boundary

This audit is derived from the registry's declared methods, caveats, implementation function names, and the existing v0.5.1 coverage notes. It does not claim that calibration data or all required physical models are available for every tokamak. In particular, a generic numerical primitive does not validate a diagnostic-specific physical inference.
