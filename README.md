# Tokamak Insight Suite (TIS)

**Python toolkit for tokamak plasma diagnostics, data processing, and physics-oriented analysis.** Current source release: **v0.3.0**.

![Dashboard design reference](docs/design_reference_dashboard.png)

## Run locally

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

On Windows, `run_windows.bat` is also provided. On Linux/macOS, use `bash run_unix.sh`.

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

A separate GitHub Actions workflow runs the fetch and checks real-data CSV ingestion, finite/time-ordered samples, FFT, Welch PSD, and filtering. Run it manually from **Actions → Real MAST data integration test → Run workflow**; successful runs publish the CSV and manifest as a downloadable workflow artifact. Network access to the public archive is required, so the raw data are not bundled in Git.

This validates a real diagnostic time-series path only. It does not validate all TIS algorithms, diagnostic calibration, or absolute physical units. The source catalog labels data quality as “Not Checked”; inspect the source metadata and limitations before using the data for scientific conclusions.

## Workspace

- **Data Import & QC:** CSV, TXT, Excel and Parquet import, column inspection, missing-data checks, signal quality metrics, SHA-256 provenance and exportable QC reports.
- **Time Series & MHD:** filtering, FFT, Welch PSD, cross-correlation/coherence and phase-based mode-number estimates.
- **Spectral Analysis:** line fitting, spectral calibration, Doppler temperature/velocity estimates, spectral moments and line-ratio utilities.
- **Probe Analysis:** Langmuir I–V analysis, probe summaries, triple-probe estimates and fluctuation/particle-flux utilities.
- **Plasma Parameters:** parameter estimates from explicitly selected measurements and assumptions.
- **Equilibrium & Reconstruction:** derivatives of an input flux grid and profile analysis.
- **Tomography & Bolometry:** regularized tomography, Abel reconstruction and axisymmetric radiated-power integration.
- **Advanced Analysis:** X-ray spectrum summaries, neutron-rate estimates, heat-flux fits and related routines.
- **Diagnostic Catalog:** `DIAGNOSTIC_REGISTRY.json` and `FORMULA_REGISTRY.json`, with supporting catalog modules.
- **Reports & Export:** data and analysis exports with provenance metadata.

## Tests and CI

The automated GitHub Actions workflow installs the dependencies and runs the test suite on repository updates. The suite includes **66 test cases** (including parameterized input-validation cases). The current result is recorded in [GitHub Actions](https://github.com/Ramin-Alipour/Tokamak-Insight-Suite/actions).

## Scientific scope and limitations

This is a research analysis framework, **not** a tokamak control, interlock, or safety system. Calculations are only meaningful when the user supplies appropriate units, calibration, geometry, instrument response, uncertainty assumptions, and valid measurement conditions. Some modules provide generic estimates or analysis building blocks rather than complete device-specific diagnostic inversions. In particular, flux-grid derivatives are not an EFIT/Grad–Shafranov equilibrium solver. Do not treat a formula catalog entry as proof of a validated end-to-end implementation.

The bundled source reference is `docs/reference/tokamak_reference_extracted.txt`, extracted from the 24-page `tokamak.pdf` held in the user's Library. The original PDF is not bundled in this repository. Read [Scientific Scope](docs/scientific_scope.md), [Diagnostic Coverage](docs/diagnostic_coverage.md), and [Release Notes](RELEASE_NOTES.md) before interpreting research results.
