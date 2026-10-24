# First-ten diagnostic audit (main)

Audit baseline reviewed against current `main` on 2026-10-24.  
Scope: first ten entries in `DIAGNOSTIC_REGISTRY.json`, implementation modules and focused tests inspected through GitHub. This is a source review, not a locally executed test run; a clean clone/test execution was unavailable in the review environment. The current canonical registry and `tests/test_catalog_contract_sync.py` supersede the stale status/count snapshot in earlier versions of this audit.

## Executive finding

The registry is not saying that all ten entries lack code. Current `main` declares Thomson scattering `complete_with_declared_scope`, ECE and CXRS `complete_with_declared_scope`, and the other seven entries in the first ten `complete` within their declared computational scope. These metadata labels do not mean independent experimental validation.

The current status-report test requires the global canonical counts to be 80 records = 8 `complete` + 3 `complete_with_declared_scope` + 69 `incomplete`. The first ten include seven `complete` and three `complete_with_declared_scope`. Do not mark Thomson incomplete solely because it lacks independent experimental validation; equally, do not describe it as independently validated.

## Per-diagnostic findings

| # | Diagnostic | Code/test evidence inspected | Audit conclusion / remaining scientific boundary |
|---|---|---|---|
| 1 | Thomson scattering | `tokamak_analysis.thomson_workflow`, `thomson_scattering`, `thomson_spectrum`, `thomson_models`; registry declares five modes and focused numerical/input-validation tests | **Complete within declared software scope; not independently experimentally validated.** The supported calibrated-profile, raw-spectrum electron-temperature fit, calibrated power-density, combined workflow, and custom-model paths are implemented and covered by focused numerical/input-contract tests. Limitations remain: non-relativistic Gaussian Maxwellian spectral approximation, known Gaussian instrument response, calibrated power/geometry for density, and no device-specific empirical validation fixture. The canonical registry therefore uses `complete_with_declared_scope`; this must not be mistaken for a claim of experimental validation. |
| 2 | Langmuir probe | `tokamak_analysis.langmuir_probe`, `tis.core`; contract and warnings inspected | **Complete within declared single-sweep model scope.** Fit-window selection, calibration, sheath assumptions and uncertainty limitations remain user responsibilities; do not imply automatic experimental validation. |
| 3 | ECE | `tokamak_analysis/ece_analysis.py`, `tests/test_ece_analysis.py` | **Complete with declared scope.** Resonance mapping uses a simplified circular toroidal-field model; brightness temperature needs calibrated spectral radiance and only approximates electron temperature when optically thick/thermalized. No EFIT-grade localization. |
| 4 | CXRS | `tokamak_analysis/cxrs.py`, `tests/test_cxrs.py` | **Complete with declared scope.** Isolated calibrated Gaussian line fitting can estimate line-of-sight velocity and ion temperature. No impurity-density/atomic-kinetics, beam attenuation, line deblending, multi-view inversion or independent device validation. |
| 5 | Ion temperature from Doppler spectroscopy | `tokamak_analysis/ion_temperature.py`, `tokamak_analysis/doppler_temperature.py`; registry declares fit diagnostics and uncertainty outputs | **Complete within isolated-line workflow.** Instrumental Gaussian width is removed in quadrature; unresolved/non-thermal broadening, Zeeman/Stark effects, flow and line identification require external review. Do not confuse the separate profile-summary function with spectral inference. |
| 6 | Molecular spectroscopy | `tokamak_analysis/molecular_spectroscopy.py`, `tests/test_molecular_spectroscopy.py` | **Complete for exploratory feature extraction.** Peak detection, integration, supplied-reference matching and caller-supplied model fitting do not constitute automatic species identification or rotational/vibrational population/temperature inversion. No built-in molecular database is claimed. |
| 7 | Thomson spectral distribution | `tokamak_analysis/thomson_distribution.py`, `tests/test_thomson_distribution.py` | **Complete for model-independent measured-spectrum characterization.** Moments and Gaussian fit describe the measured wavelength distribution; they do not uniquely infer an electron velocity distribution, density or temperature. Instrument width and background assumptions matter. |
| 8 | Toroidal-field coils / magnetic probes | `tokamak_analysis/toroidal_field_coils.py`, `tests/test_toroidal_field_coils.py` | **Complete for calibrated signal conversion and explicitly calibrated pickup-voltage integration.** Pickup integration needs validated sensitivity/transfer function, baseline and initial-field condition; drift and correlated uncertainty are not fully modeled. |
| 9 | Poloidal-field coils / magnetic probes | `tokamak_analysis/poloidal_field_probes.py`, `tests/test_poloidal_field_probes.py` | **Complete for calibrated digitized signals.** Derivative is temporal, not spatial; it does not infer poloidal flux, PF current, boundary or equilibrium. Sampling jitter is reported but does not replace analog anti-alias/bandwidth validation. |
| 10 | Mirnov / pickup coils | `tokamak_analysis/mirnov_analysis.py`, `tests/test_mirnov_analysis.py` | **Complete for declared numerical scope.** PSD, ideal geometry-based dB/dt, baseline-explicit integration and coherence-weighted mode candidates exist. Mode number is alias-prone and depends on phase/polarity/geometry/bandwidth; transfer calibration and independent validation remain external. |

## Follow-up work required before calling the first ten scientifically validated

1. Run the full test suite and locked dependency environment against the exact main SHA; this review did not execute tests locally.
2. Add independently sourced reference datasets and tolerance budgets for each diagnostic, not only synthetic round-trip tests.
3. For Thomson, validate the forward model against calibrated experimental spectra and document calibration/response uncertainty before changing its completeness flag.
4. For ECE, test resonance localization against a supplied equilibrium and validate radiance/optical-depth assumptions.
5. For CXRS and Doppler Ti, test blended lines, non-thermal broadening and instrument line-shape uncertainty against reference data.
6. For probes/Mirnov, add calibrated transfer-function, polarity, phase, drift and aliasing reference cases representative of real acquisition chains.
7. Keep computational scope and experimental validation as separate fields in reports. A `complete` label must never be presented as proof of experimental validation.

## Change policy

This audit does not promote any completeness flag. Such a change without reference-data evidence would misrepresent scientific maturity. The next code changes should address a reproduced numerical/contract defect and add a regression test for it.
