# Coverage map against `tokamak.pdf`

This map follows the source document's final diagnostic-family index. “Registry” means an explanatory catalog entry exists; “implemented workflow” means a generic analysis operation is executable in this release. It does not imply every listed observable is solved from raw detector data.

| # | Source family | TIS coverage in v0.2.0 |
|---:|---|---|
| 1 | Magnetic measurements | Registry; general time-series/FFT/cross-channel tools |
| 2 | Equilibrium reconstruction | Registry only; specialized Grad–Shafranov solver not included |
| 3 | Langmuir probes | Registry; I–V descriptive analysis, not full regime-specific Te/ne inference |
| 4 | Ball-pen probe | Registry only; no universal calibration correction |
| 5 | Gundestrup/Mach probe | Registry only; directional/sheath model is device-specific |
| 6 | Probe arrays | Registry; generic multi-channel correlation/coherence tools |
| 7 | Thomson scattering | Registry only; no instrument-specific scattering forward model |
| 8 | Interferometry | Registry; simple high-frequency phase-to-line-density approximation |
| 9 | Polarimetry | Registry only; no full magnetized propagation inversion |
| 10 | Reflectometry | Registry only; no full cutoff/ray-tracing inversion |
| 11 | ECE/CECE | Registry only; no resonance/radiative-transfer reconstruction |
| 12 | Doppler spectroscopy | Implemented generic Gaussian/Lorentzian/Voigt fitting and conditional Doppler estimates |
| 13 | CXRS | Registry only; atomic/beam model required |
| 14 | Atomic/molecular spectroscopy | Generic line fitting; atomic identification/population models not included |
| 15 | VUV/X-ray spectroscopy | Registry; generic line fitting; detector response/atomic model not included |
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
| 27 | MHD diagnostics | Implemented generic time-series, FFT, Welch PSD, cross-spectrum, coherence and lag analysis |
| 28 | Turbulence diagnostics | Implemented generic fluctuation/correlation/spectral tools; physical flux inversion not included |
| 29 | Wall/divertor diagnostics | Registry; no device-specific thermal/erosion inversion |
| 30 | Data processing tools | Implemented QC, calibration, filtering, fitting, uncertainty, provenance and exports |

## Design rule
Where the source requires an inverse problem, machine geometry, atomic rates, sheath models, instrument response, or specialized equilibrium/orbit solvers, the program documents the requirement instead of claiming a direct measurement or inventing missing physics.
