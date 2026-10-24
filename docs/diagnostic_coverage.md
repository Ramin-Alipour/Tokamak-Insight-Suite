# Coverage map against the extracted `tokamak.pdf` reference text

The original third-party PDF is not redistributed in this repository. A limited extracted-text reference used to organize the diagnostic catalog is bundled at `docs/reference/tokamak_reference_extracted.txt`; consult the original publication and its license for authoritative details.

This map follows the source document's final diagnostic-family index. “Registry” means an explanatory catalog entry exists; “implemented workflow” means a generic analysis operation is executable in this release. It does not imply every listed observable is solved from raw detector data.

| # | Source family | TIS coverage in v0.5.1 |
|---:|---|---|
| 1 | Magnetic measurements | Generic time-series/FFT/cross-channel tools plus calibrated signal integration with optional baseline subtraction; device-specific transfer functions remain user-supplied |
| 2 | Equilibrium reconstruction | Registry only; specialized Grad–Shafranov solver not included |
| 3 | Langmuir probes | Implemented I–V fitting and scoped electron-temperature/summary estimates; regime selection and device calibration remain user-dependent |
| 4 | Ball-pen probe | Registry only; no universal calibration correction |
| 5 | Gundestrup/Mach probe | Registry only; directional/sheath model is device-specific |
| 6 | Probe arrays | Registry; generic multi-channel correlation/coherence tools |
| 7 | Thomson scattering | Calibrated-profile gradients/scale lengths and a scoped raw-spectrum Gaussian temperature fit; density requires calibrated power/geometry inputs; not a full device-specific inversion |
| 8 | Interferometry | Registry; simple high-frequency phase-to-line-density approximation |
| 9 | Polarimetry | Registry only; no full magnetized propagation inversion |
| 10 | Reflectometry | Registry only; no full cutoff/ray-tracing inversion |
| 11 | ECE/CECE | Scoped resonance mapping, calibrated brightness-temperature utilities and fluctuation PSD; no full radiative-transfer inversion |
| 12 | Doppler spectroscopy | Implemented generic Gaussian/Lorentzian/Voigt fitting and conditional Doppler estimates |
| 13 | CXRS | Scoped calibrated line fitting, ion-temperature and line-of-sight velocity estimates; atomic/beam and viewing-geometry systematics remain |
| 14 | Atomic/molecular spectroscopy | Generic line fitting and user-defined molecular-model fitting; atomic identification/population models are not bundled |
| 15 | VUV/X-ray spectroscopy | Generic line fitting and X-ray spectrum summaries; detector response and atomic models remain user-supplied |
| 16 | Bolometry | Generic regularized linear inverse solver; requires calibrated geometry matrix |
| 17 | Visible imaging | Registry only; image calibration/segmentation module not included |
| 18 | Infrared imaging | Registry only; radiometric/heat-transfer inversion not included |
| 19 | Neutron diagnostics | Registry only; response unfolding not included |
| 20 | Gamma diagnostics | Registry only; energy/efficiency calibration and unfolding not included |
| 21 | NPA | Registry only; fast-neutral forward model not included |
| 22 | FIDA/CTS | Registry only; fast-ion inversion not included |
| 23 | FILD | Registry only; orbit-following/response model not included |
| 24 | RGA/pressure gauges | Registry only; species-dependent calibration not included |
| 25 | Flowmeters/fueling | Registry; formula reference for injected particle number |
| 26 | Heating/RF diagnostics | Registry; basic power/energy formula references |
| 27 | MHD diagnostics | Implemented time-series, FFT/Welch PSD, cross-spectrum/coherence, lag and scoped Mirnov mode-number analysis; calibration and geometry remain essential |
| 28 | Turbulence diagnostics | Generic fluctuation/correlation/spectral tools and scoped fluctuation-flux estimates; cross-phase, calibration and model assumptions must be checked |
| 29 | Wall/divertor diagnostics | Registry; no device-specific thermal/erosion inversion |
| 30 | Data processing tools | Implemented QC, calibration, filtering, fitting, uncertainty, provenance and exports |

## Design rule
Where the source requires an inverse problem, machine geometry, atomic rates, sheath models, instrument response, or specialized equilibrium/orbit solvers, the program documents the requirement instead of claiming a direct measurement or inventing missing physics.

## Addendum (v0.5.1): workflows added beyond the 30-family map
| Topic | Implementation | Limits |
|---|---|---|
| ELM / transient detection | `tis.events.detect_elms` | Statistical detector; no ELM-type classification |
| Sawtooth analysis | `tis.events.detect_sawteeth` | Needs a core channel inside the inversion radius |
| Disruption analysis | `tis.events.current_quench_analysis` | Quench timing from I_p only; no thermal-quench/precursor classifier |
| Energy confinement / scalings | `tis.plasma` (IPB98(y,2), H98, Martin-08, q95, β_N, ν*) | Empirical scalings for conventional tokamaks; comparison only |
| Gap/spike handling | `tis.signals` | Interpolation attenuates content near Nyquist; Hampel despiking can remove real transients if configured poorly |\n| Noise floor / detection limit | `tis.signals.noise_floor_detection_limit` | Requires background-only samples; threshold is per sample and does not infer bandwidth, correlation, multiple-comparison effects or device calibration |
