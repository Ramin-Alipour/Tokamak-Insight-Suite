# External scientific validation results — run 38619292500

Date: 2026-10-14
Repository commit: b8f7514f7aebefce42e43f686396a3c903911db0
Workflow: https://github.com/Ramin-Alipour/Tokamak-Insight-Suite/actions/runs/38619292500
Artifact: https://github.com/Ramin-Alipour/Tokamak-Insight-Suite/actions/runs/38619292500/artifacts/11922500024
Rule: no publications authored by Ramin Alipour were used as validation references in this external-data campaign.

## Results actually executed on GitHub Actions

### MAST shot 29790 — OMAHA Mirnov signal
Source: UKAEA FAIR-MAST Level-1, https://mastapp.site/level1-data.html
Object: https://s3.echo.stfc.ac.uk/mast/level1/shots/29790.zarr
Signal: `xmo/omaha_1lz`
Source classification in manifest: Raw; source quality: Not Checked; units: not supplied.
- Original samples: 1,400,000 at 1,999,999.999998 Hz.
- Exported samples: 70,000; stride 20; polyphase FIR Kaiser-window anti-alias filtering applied before decimation.
- Export interval: -0.100000 to 0.599990 s; effective sampling rate 99,999.9999999 Hz.
- Exported signal min/max: -2.8042306 / 2.2185223 (source units not supplied).
- Mean: -8.99515e-4; standard deviation: 0.1348171.
- Welch PSD peak excluding DC: 19,836.425781 Hz.
- Executed and passed: finite/monotonic timebase checks, CSV ingestion, FFT, Welch PSD, Butterworth digital filtering and finite-output checks.
- Interpretation: **PASS for data ingestion and algorithm execution on real measured signal; UNASSESSED for physical-frequency agreement** because no independent published/archived reference value for the same channel/window and processing is yet tied to this signal. The 19.836 kHz peak is an observed software result, not yet a validated physical result.

### MAST shot 30421 — plasma current and density traces
- Group `amc/plasma_current`: 30,000 samples, estimated sampling rate 4,999.170 Hz; FFT and Welch PSD executed successfully.
- Group `ane/density`: 32,768 samples, estimated sampling rate 25,003.303 Hz; FFT and Welch PSD executed successfully.
- Inventory checks: amc 46 variables / 44 one-dimensional traces; ane 7 variables / 3 one-dimensional traces.
- Interpretation: **PASS for ingestion and numerical execution; UNASSESSED for quantitative scientific agreement** because a matching independent reference trace/value and processing metadata were not established.

### MAST shot 30421 — existing EFIT equilibrium grid
- Snapshot at t = 0.200000 s; exported grid shape 65 × 65; finite fraction 1.000.
- Interpretation: **PASS for retrieval/shape/finite-value checks only.** This does not validate equilibrium reconstruction, flux derivatives, magnetic topology or agreement with an independent equilibrium solver.

## Family-by-family status

| TIS family | Current external evidence | Status |
|---|---|---|
| CSV ingestion, timebase, finite/missing data | Real MAST traces from shots 29790 and 30421 | PASS for tested cases |
| FFT and Welch PSD | Executed on MAST Mirnov, plasma-current and density traces | PASS for execution; physical reference comparison UNASSESSED |
| Digital filtering | Executed on real MAST Mirnov trace after sampling-rate handling | PASS for execution; physical transfer/phase/amplitude agreement UNASSESSED |
| SVD, wavelet, cross-spectrum, coherence, mode number | No external matched reference output executed in this run | UNASSESSED |
| Doppler spectroscopy, line fitting and line ratios | No independent non-user raw numerical spectrum with calibrated wavelength/instrument response executed | BLOCKED |
| Langmuir/triple probes and Mach number | No matched external raw I–V curve or paired probe channels with geometry/calibration executed | BLOCKED |
| E-field, E×B, fluctuation flux/Reynolds stress | No synchronized, calibrated, geometrically matched multichannel reference executed | BLOCKED |
| Heat-flux profile fitting | No matching calibrated target profile/reference executed | BLOCKED |
| X-ray spectrum and neutron-rate estimate | No external raw detector spectrum/counts with response/efficiency metadata executed | BLOCKED |
| Tomography, Abel inversion and radiated power | No independent projection dataset plus geometry/reference reconstruction executed | BLOCKED |
| Equilibrium-grid derivatives | EFIT grid retrieved, but derivative comparison with independent solver not run | UNASSESSED |
| Global derived parameters and formula utilities | Existing analytic/unit tests only; not an external experimental validation | UNASSESSED |
| Reflectometry/ECR formula utilities | Formula-level analytic tests only; no raw diagnostic inversion | UNASSESSED |

## What this run does and does not establish

It establishes that TIS can retrieve and ingest selected open MAST measurements and execute several signal-processing routines on them in GitHub Actions. It does **not** establish that all analyses are scientifically valid, and it does not establish agreement with publication results for every family. The only honest current outcome is partial real-data integration plus a set of explicitly unassessed/blocked physical comparisons.

## Next required external comparisons

1. Obtain the MAST `ama27643.nc` catalogued mode-frequency/amplitude product and the corresponding raw signal/provenance; compare only after establishing whether the signal is raw or already processed and matching time windows/units.
2. Reproduce the GOLEM 35755/37074 Mirnov-position notebook from its exact source signals and vacuum-shot correction; compare the reconstructed R/Z traces with its notebook output.
3. For spectroscopy, probes, heat flux, X-ray, tomography and neutron rate, find independent public raw numeric datasets plus calibration/geometry and an independently reported result. Until then keep these families BLOCKED rather than substituting synthetic data or user-authored papers.
