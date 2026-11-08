# Public tokamak experimental data sources

This catalog is a practical starting point for testing Tokamak Insight Suite (TIS) against real experimental data. It is **not** a claim that every tokamak's full archive is public: access, licensing, signal coverage, calibration, and metadata vary by facility and dataset.

## Sources with public access routes

| Source | What is available | Access route | Important caveats |
|---|---|---|---|
| **UKAEA FAIR-MAST** | Shot-level Zarr groups covering magnetics/Mirnov coils, plasma currents, interferometry, Thomson scattering, EFIT-derived equilibrium, cameras, and other diagnostics | [Catalog/API](https://mastapp.site/); shot metadata API: `https://mastapp.site/json/shots/{shot}`; [Level-1 examples](https://mastapp.site/level1-data.html) | Level-1 can be raw or processed depending on group. Check each group's `signal_type`, `quality`, units, and metadata. Some groups are large. |
| **GOLEM tokamak** | Public per-shot time traces and diagnostic products; examples include loop voltage, magnetic field/current traces, spectroscopy, camera and hard-X-ray data where recorded | [Shot database](https://golem.fjfi.cvut.cz/); [data/API formats](https://golem.fjfi.cvut.cz/wiki/SW/pygolem/pygolem) | Available diagnostics differ by shot. Prefer the shot's Data page and its explicit download links; do not assume every signal exists for every discharge. |
| **TokaMark / FAIR-MAST benchmark** | Harmonized benchmark built from 39 signals across 11,573 MAST shots, including magnetics, kinetic profiles, radiation, actuators, and EFIT-derived quantities | [Dataset](https://huggingface.co/datasets/UKAEA-IBM-STFC/tokamark-v1) | The full archive is about 565 GB and distributed in batch archives. It is a curated benchmark, not a substitute for each original raw signal. Start with the included sample/benchmark code before downloading large batches. |
| **VEST / VAFT** | Public HSDS/IMAS-style per-shot data and a Python interface mapping diagnostics such as magnetics, Thomson scattering, UV spectroscopy, and charge exchange | [VAFT repository](https://github.com/VEST-Tokamak/vaft); [data access docs](https://vest-tokamak.github.io/vaft/workflows/data-access-imas/) | Some diagnostics are file-backed or require optional access/configuration. Check module documentation and the specific shot's available IDSs. |

## Integration strategy for TIS

1. Keep a **source adapter** for each archive instead of pretending all archives share one schema.
2. Preserve the original shot number, source URL, diagnostic/group, signal name, units, calibration status, quality flags, timestamps, and retrieval time in a provenance manifest.
3. Never silently infer calibration or units. Raw voltage, calibrated magnetic field, reconstructed equilibrium, and derived profiles must remain distinguishable.
4. Validate each data path independently: download/decoding, timebase and missing-value checks, import into TIS, algorithm-level checks, and—where possible—comparison with published/reference results.
5. Use small, reproducible samples in CI. Large archives should be opt-in and should not be committed to Git.

## Current validation boundary

The current automated real-data test targets one FAIR-MAST Mirnov time series. Passing that test demonstrates that a real signal can traverse the tested import and signal-processing path; it does **not** validate every diagnostic module or every data source. Separate integration tests are required for spectroscopy, Thomson profiles, equilibrium grids, imaging, probe I–V curves, bolometry/tomography, and other specialized data formats.
