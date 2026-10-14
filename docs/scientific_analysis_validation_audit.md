# Scientific analysis validation audit

Audit date: 2026-10-14  
Repository revision audited: `856b3147f6a096fece94c4ef02b541c4984e42`  
CI: [Continuous validation](https://github.com/Ramin-Alipour/Tokamak-Insight-Suite/actions/runs/38610080962) and [Streamlit smoke test](https://github.com/Ramin-Alipour/Tokamak-Insight-Suite/actions/runs/38610080800).

## Validation levels (do not conflate them)

- **A — Independent analytic/golden test:** a known equation, limit, sign, unit, or independently computed numerical answer is checked.
- **S — Synthetic recovery/invariant test:** known synthetic inputs are recovered, or a mathematical invariant is checked.
- **P — Publication comparison:** output is compared to a published number/table/figure. Digitized figure values carry digitization uncertainty.
- **U — Not independently validated as a complete experimental diagnostic:** requires device-specific calibration, geometry, raw signals, atomic/response data, uncertainty model, or an independent reference.

Passing A or S validates the tested computation under its assumptions; it does not establish experimental validity. A catalog formula marked implemented or documented is not automatically a validated workflow.

## Analysis-family audit

| Analysis family | Implemented workflow(s) | Evidence currently in automated tests | Remaining scientific boundary |
|---|---|---|---|
| Signal ingestion, calibration, quality, timebase | CSV/table ingestion, hashing, calibration, missing-data and sampling checks | A/S: calibration, units/header parsing, timestamp quantization, gap handling, byte/row limits, deterministic hash | Device metadata and calibration provenance still have to be supplied and checked for each dataset |
| Time-series integration and resampling | Integration, timebase preparation, resampling | A/S: linear integral, baseline-offset removal, irregular sampling/gap rejection, jitter resampling | Real sensor bandwidth, drift, offset and uncertainty are dataset-specific |
| Filtering and spectral response | Butterworth filtering, filter-impact analysis, anti-alias decimation | A/S: passband recovery, invalid cutoff rejection, out-of-band suppression, transfer-response impact | Must justify filter/window/cutoff against the diagnostic bandwidth; filtering can change amplitudes and phases |
| FFT, Welch PSD, spectrogram, autocorrelation | Frequency, power, time-frequency and periodicity analysis | A/S: known-tone recovery, Nyquist amplitude, PSD/variance consistency, spectrogram and autocorrelation checks | Finite-record/window uncertainty and nonstationarity require per-shot assessment |
| Cross-spectrum, coherence, lag, mode number | Cross-analysis and multichannel phase methods | A/S: lag sign, coherence bounds, phase winding, spatial Nyquist and sensor-angle checks | Channel synchronization, probe geometry, phase calibration and mode aliasing remain experimental constraints |
| SVD and wavelet MHD decomposition | SVD modes, Morlet time-frequency localization | S: rank-one recovery, known-tone localization, invalid-support rejection | Synthetic recovery is not proof that a given measured component is an MHD eigenmode |
| Doppler spectroscopy | Gaussian/Lorentzian/Voigt fitting, wavelength calibration, instrument-width correction, Doppler velocity and temperature | A/S: fit recovery, uncertainty checks, calibration and unresolved-width rejection; P: 15-row table-to-formula reconstruction added in this audit | Published tables are not raw CCD spectra; blends, calibration covariance, line identification and instrumental profile need independent evidence |
| Spectral peak identification and line ratios | Peak detection/matching, response-corrected ratios | A/S: sub-sample center recovery, ambiguity/no-match handling, response-ratio golden values | Atomic data and response curves are required before interpreting a line ratio as Te, ne or impurity abundance |
| Langmuir and triple probes | I–V summary, electron-temperature slope, floating potential, optional density/ion-fit paths; idealized triple-probe estimate | A/S: synthetic exponential slope, zero-crossing interpolation, invalid/insufficient-range and missing-geometry checks | Sheath model, collection area, ion species, circuit topology and real probe calibration are not automatically known |
| Electric fields and E×B drift/transport | Potential gradients, drift quantities, fluctuation covariance/flux | A/S: explicit gradient/sign/spacing, E×B numerical golden tests, common finite masks, mean-product vs covariance separation | Sign conventions, magnetic-field orientation and probe spatial alignment must match the experiment |
| Mach number and sound speed | Ion-sound speed and signed Mach number | A: SI equation and numerical golden values; P: unresolved compound-probe figure consistency comparison | Probe article’s exact temperature convention and paired time series remain unresolved; do not force agreement |
| Heat-flux profiles | Exponential and Eich-like fits, profile integration | S: synthetic recovery and uncertainty outputs; profile integral tests | Real target geometry, surface mapping, background and heat-flux calibration are required |
| X-ray and spectral moments | X-ray spectrum summary, spectral moments and background handling | A/S: counts, centroid, uncertainty, signed-net-bin and bin-representation checks; P: five-point figure-derived mean vs paper table | Figure digitization is provisional; not raw detector events or independent energy calibration |
| Neutron-rate estimate | Count-rate/efficiency/geometric correction | A: factor correction and invalid-factor tests | Efficiency, solid angle, attenuation, dead time and detector calibration must be measured for a real setup |
| Tomography / Abel reconstruction / radiated power | Geometry path matrix, regularized inversion, Abel reconstruction, axisymmetric power | A/S: geometry path lengths, known-profile recovery, rank/conditioning, uncertainty/noise sensitivity, emissivity constraints | Inverse solution depends on geometry, regularization, coverage and noise; compare with an independent reconstruction before claiming experimental validation |
| Equilibrium-grid derivatives | Numerical derivatives of flux and derived fields | A/S: quadratic-flux golden case and invalid-input checks | Not a Grad–Shafranov equilibrium reconstruction and not validated against a magnetic equilibrium code |
| Derived global plasma quantities | Pressure, confinement-time and related estimates | A: equation/units and invalid-domain checks | Input definitions (stored energy, loss power, volume averages, sign convention) must be consistent with the device |
| Reflectometry/ECR and formula utilities | O-mode cutoff density, ECR field, ratios and related utilities | A: independent SI equations and numeric golden values; X-mode intentionally rejects unsupported shortcut | These are formula-level calculations, not complete diagnostic inversions |
| Publication/reference comparisons | Scalar and time-series comparison with explicit uncertainty/tolerance | A/S: explicit tolerance/uncertainty policy and no arbitrary pass status | No criterion means UNASSESSED; agreement with a digitized figure is not independent raw-data validation |

## Publication benchmarks currently available

1. **IR-T1 Doppler impurity table:** 15 published line-width/temperature rows are compared after explicitly converting Gaussian FWHM to sigma. This validates table-to-formula conversion within published rounding. It is not a raw-spectrum refit.
2. **IR-T1 compound probe:** digitized averages from the Mach and velocity plots do not close under both illustrative sound-speed assumptions. This remains an investigation flag, not a declared paper/software error.
3. **IR-T1 hard X-ray:** five provisionally digitized event energies reproduce a mean near 657.69 keV, versus 660.20 keV from the article table; the relative difference is about 0.38%. This is a figure-to-table check, not raw detector validation.

The above are not yet six-paper end-to-end experimental validations. Publication benchmarks must be added per article and per reported analysis before claiming that every result in all six papers has been independently reproduced.

## CI result at this revision

Continuous validation on Python 3.10, 3.11 and 3.12 passed; each job ran the automated suite, static checks, packaged CLI/dashboard smoke checks and HTTP health check. On Python 3.11 the current suite reported **276 passed** and **64% total line coverage**; `tis/core.py` had **83% line coverage**. Coverage measures executed code, not physical correctness. Several UI pages and data-source paths have low coverage and need dedicated integration fixtures.

## Required interpretation

- A/S pass: the specific equation or synthetic behaviour tested passed.
- P pass: the specified published table/figure comparison passed under its stated extraction/tolerance assumptions.
- U/unresolved: do not advertise the workflow as experimentally validated.
- No raw data, uncertainty, calibration or reproducible reference means no claim of full experimental validation.
