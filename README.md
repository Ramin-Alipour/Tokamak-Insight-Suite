# Tokamak Insight Suite (TIS)

**Python toolkit for tokamak plasma diagnostics, data processing, and physics-oriented analysis.** Current source version: **v0.5.1**. See `docs/architecture/` for execution contracts, the canonical measurement model and reproducible project persistence.

![Dashboard design reference](docs/design_reference_dashboard.png)

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# Optional extras: pip install '.[test,excel,data]'
streamlit run app.py
```

On Windows, `run_windows.bat` is also provided. On Linux/macOS, use `bash run_unix.sh`.

## Command line

```bash
tis qc data.csv                     # column audit + SHA-256
tis fft data.csv --signal b1 --time t
tis events data.csv --kind elm --time t --signal d_alpha
tis events data.csv --kind quench --time t --signal ip
tis scalings --ip-A 1e6 --bt-T 1 --r-m 0.9 --a-m 0.55 --kappa 1.8 --ne-line-avg-m3 4e19 --p-loss-W 2e6
```

## Run with Docker

```bash
docker build -t tokamak-insight-suite .
docker run --rm -p 8501:8501 tokamak-insight-suite
```

Then open `http://localhost:8501`.

## Real experimental data integration test

TIS includes a reproducible fetcher for **public raw MAST data** from the UKAEA FAIR-MAST archive. The reference case is shot **29790**, Mirnov-coil source group `xmo`, signal `omaha_1lz`. The public catalog documents the signal as raw and shows 1,400,000 time samples. The script exports a downsampled CSV plus a JSON provenance/quality manifest; it does not claim absolute magnetic calibration.

Source documentation: https://mastapp.site/level1-data.html  
Archive overview and access: https://www.ukaea.org/service/fair-mast/

Run locally (requires network access):
```bash
python -m pip install -r requirements-data.txt
python tools/fetch_fair_mast_sample.py --output-dir artifacts/mast-real-data
```

A GitHub Actions integration workflow now exercises multiple real diagnostics, not just one trace: shot 29790 `xmo/omaha_1lz` (Mirnov pickup signal), plus shot 30421 `amc/plasma_current` (plasma/coil-current source) and `ane/density` (CO₂ interferometer). It also exports a selected `efm/plasma_current_rz` grid from the archive’s existing EFIT reconstruction at a requested time. In the dashboard, the Public Data Access page can inventory variables available in a shot/group, mark which numeric one-dimensional traces the current importer supports, fetch time-series data, and retrieve/plot an existing EFIT grid. It checks finite samples, strictly increasing time, anti-alias-filtered downsampling, CSV export, FFT, Welch PSD, and digital filtering for the Mirnov case; FFT/Welch plus signal-quality metrics for the current and interferometer traces; and the dimensions, coordinates, finite fraction, and metadata of the EFIT grid. Successful runs publish CSV files, JSON provenance manifests, and a summary in the downloadable artifact `mast-real-data-multidiagnostic-validation`. Run it manually from **Actions → Real MAST data integration test → Run workflow**. Network access to the public archive is required, so raw data are not bundled in Git.

This validates a real diagnostic time-series path only. Anti-aliasing is applied before downsampling; the exported sample-rate and filter method are recorded in the provenance manifest. It does not validate all TIS algorithms, diagnostic calibration, or absolute physical units. The source catalog labels data quality as “Not Checked”; inspect the source metadata and limitations before using the data for scientific conclusions.

Public real-data archives and access notes: [Public Tokamak Data Sources](docs/public_tokamak_data_sources.md).

**External empirical validation protocol:** [raw-data + independent-reference acceptance criteria, GOLEM candidates, and FAIR-MAST comparison requirements](docs/external_empirical_validation_protocol.md). Candidate datasets are not labeled validated until rawness, reference equivalence, numerical discrepancy, and reproducibility are all documented.

## Workspace

- **Data Import & QC:** CSV, TXT, Excel and Parquet import, column inspection, missing-data checks, signal quality metrics, SHA-256 provenance and exportable QC reports.
- **Time Series & MHD:** filtering, FFT, configurable Welch PSD, cross-correlation/coherence, channel-derived coherence-weighted mode-number estimates, calibrated time-series integration with optional baseline subtraction, time-frequency spectrograms, and autocorrelation/correlation-time estimates.
- **Spectral Analysis:** line fitting, spectral calibration, Doppler temperature/velocity estimates, spectral moments and line-ratio utilities.
- **Probe Analysis:** Langmuir I–V analysis, probe summaries, triple-probe estimates and fluctuation/particle-flux utilities.
- **Plasma Parameters:** parameter estimates from explicitly selected measurements and assumptions.
- **Equilibrium & Reconstruction:** derivatives of an input flux grid and profile analysis; Public Data Access can fetch and plot an existing FAIR-MAST EFIT `plasma_current_rz` grid without claiming to reconstruct equilibrium.
- **Tomography & Bolometry:** uncertainty-weighted regularized tomography with optional L-curve λ selection, Abel reconstruction, axisymmetric radiated-power integration, and Eich-like or simple-exponential target heat-flux fitting.
- **Advanced Analysis:** X-ray spectrum summaries, neutron-rate estimates, heat-flux fits and related routines.
- **Events & Transients:** ELM detection and repetition statistics, sawtooth crash detection (period, amplitude), plasma-current-quench analysis (ITER 80 %→20 % convention). Statistical detectors; always confirm against the raw trace.
- **Confinement & Scalings:** Greenwald fraction, q95 (ITER formula), β_t/β_p/β_N, ν*ₑ, Spitzer resistivity, IPB98(y,2) τ_E and H98, Martin-08 L–H threshold. Empirical comparison tools, not device-validated predictions.
- **Signal conditioning:** explicit gap policies, Hampel despiking, drift-corrected integration. Every repair is reported.
- **Diagnostic Catalog:** `DIAGNOSTIC_REGISTRY.json` and `FORMULA_REGISTRY.json`, with supporting catalog modules.
- **Thomson Scattering (analysis 1/80):** [inputs, model, outputs, assumptions, limitations, and completion gate](docs/analyses/thomson_scattering.md).
- **Reports & Export:** data and analysis exports with provenance metadata.

## Tests and CI

Install test tools with `pip install -e '.[test]'`; data archive support is optional via `pip install -e '.[data]'`. Excel support (`.xlsx` and legacy `.xls`) is included in the base dependencies.

The `Continuous validation` workflow tests Python 3.10–3.12, compiles the source, runs critical Ruff checks, executes numerical tests with coverage, and uses Streamlit AppTest to exercise Home and the default Probe page. An HTTP health check supplements these runtime tests; it is not treated as proof that every page works. Results are recorded in [GitHub Actions](https://github.com/Ramin-Alipour/Tokamak-Insight-Suite/actions).

## Scientific scope and limitations

This is a research analysis framework, **not** a tokamak control, interlock, or safety system. Calculations are only meaningful when the user supplies appropriate units, calibration, geometry, instrument response, uncertainty assumptions, and valid measurement conditions. Mode-number fits sort sensors by angle and report the spatial alias limit; phase-only data cannot identify a mode above that limit without a physical prior. Timestamp gaps, invalid Langmuir temperature estimates, and negative background-subtracted spectral bins are retained; when their signed moments are physically undefined, the software reports an explicit invalid-status rather than fabricating a centroid or width. Some modules provide generic estimates or analysis building blocks rather than complete device-specific diagnostic inversions. In particular, flux-grid derivatives are not an EFIT/Grad–Shafranov equilibrium solver. Do not treat a formula catalog entry as proof of a validated end-to-end implementation.

The bundled reference notes are in `docs/reference/tokamak_reference_extracted.txt`. The original third-party PDF is intentionally not redistributed; users should consult the original source and its applicable license. Read [Scientific Scope](docs/scientific_scope.md), [Diagnostic Coverage](docs/diagnostic_coverage.md), and [Release Notes](RELEASE_NOTES.md) before interpreting research results.
