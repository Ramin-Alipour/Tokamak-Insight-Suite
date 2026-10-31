# TIS primary analysis structure

Tokamak Insight Suite exposes **20 primary analyses** in its navigation:

- **19 physical/diagnostic families** that group related measurement systems and physics workflows.
- **1 shared instrumentation and measurement-support family** for wavelength calibration, calibration curves and uncertainty, instrumental line-shape/response characterization, noise-floor and detection-limit estimates, sampling/aliasing audits, and channel synchronization.

## Design rule

The 20 families are a user-facing organization layer, not a replacement for the scientific implementations. The existing **79 registered analysis entry points** remain intact as subanalyses/workflows. A family may contain more than one physically distinct solver; grouping never implies that their equations or assumptions are interchangeable.

The mapping is maintained in `ANALYSIS_FAMILIES.json` and mirrored in `tis/data/ANALYSIS_FAMILIES.json` for packaged installations. The registry loader verifies that family IDs are 1–20 and that every registered underlying analysis ID appears exactly once; retired ID 7 is intentionally absent.

## Primary analyses

| # | Primary analysis | Existing entry-point IDs |
|---:|---|---|
| 1 | Thomson Scattering | 1, 30, 46 |
| 2 | Langmuir & Electrostatic Probes | 2, 16–20 |
| 3 | Electron Cyclotron Emission (ECE) | 3 |
| 4 | Doppler & Charge-Exchange Spectroscopy | 4, 5, 24 |
| 5 | Molecular Spectroscopy | 6 |
| 6 | Magnetic Diagnostics & Equilibrium | 8–15, 55 |
| 7 | Emission Spectroscopy | 21, 22, 25 |
| 8 | Interferometry & Reflectometry | 28, 29, 31 |
| 9 | Bolometry & Radiation Tomography | 32, 33, 39 |
| 10 | Imaging & IR Thermography | 36–38, 75 |
| 11 | Neutron Diagnostics | 40–42 |
| 12 | Gamma-Ray Spectroscopy | 43 |
| 13 | Fast-Particle Diagnostics | 44, 45, 47, 48 |
| 14 | Gas, Pressure & Fueling Diagnostics | 49–53 |
| 15 | Fluctuation, MHD & Transient Analysis | 54, 56–60 |
| 16 | Transport, Confinement & Plasma Parameters | 61–65 |
| 17 | Heating & Power Analysis | 66–70 |
| 18 | Plasma-Facing Components & Thermal Analysis | 71–74, 76 |
| 19 | X-Ray Diagnostics | 23, 34, 35 |
| 20 | Instrumentation & Measurement Support | 26, 27, 77–80 |

## Thomson family

All Thomson-scattering-related paths are grouped under primary analysis 1, including canonical optical inference, collective/coherent-scattering and spectrum-to-spatial-profile reconstruction. Every workflow starts from measured diagnostic spectra; profile-only inputs containing previously extracted ne/Te values are not accepted. The former duplicate spectral-distribution wrapper (ID 7) has been retired.

## Instrumentation support family

Primary analysis 20 consolidates cross-diagnostic measurement-support functions into one place. It is a shared toolkit, not a twentieth physical diagnostic. These tools remain explicitly selected supporting workflows; the application does not silently calibrate, repair, resample, synchronize, or otherwise alter data as a hidden preprocessing step.

## Compatibility

The numeric IDs and callable signatures of the 79 remaining analysis entry points are preserved; ID 7 is intentionally retired and existing IDs are not renumbered. Existing registry consumers, analysis contracts, and scientific workflows can continue to refer to those IDs. The primary-family ID is a separate navigation identifier.
