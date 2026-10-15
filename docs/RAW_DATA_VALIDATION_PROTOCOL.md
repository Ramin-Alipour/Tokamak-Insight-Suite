# Raw-Data Scientific Validation Protocol

## Mandatory acceptance rule

From this point forward, a diagnostic/analysis feature is not considered independently validated merely because it passes a formula test or fits synthetic data. A qualifying benchmark must provide:

1. **Accessible experimental input**: the rawest available measured data, with units, timebase/wavelength/voltage axes, instrument metadata, and calibration details where published.
2. **An independent reference result**: a value, fitted curve, spectrum, table, or figure reported by the authors or supplied analysis pipeline.
3. **A reproducible comparison**: a scripted download or documented source URL, immutable shot/dataset identifier, file checksum where practical, exact preprocessing and fitting settings, output metrics, tolerances justified by uncertainty/resolution, and saved machine-readable results.
4. **Provenance classification**: explicitly label inputs as raw, calibrated, level-1, or already analysed. Never call processed data raw.
5. **Honest limits**: if the input or reference result is unavailable, classify the feature as synthetic/unit-tested or not independently validated; do not imply experimental validation.

## Candidate public sources to screen first

### 1. GOLEM tokamak — shot 27546

- Shot data page: https://golem.fjfi.cvut.cz/shots/27546/Data.php
- The shot catalogue lists spectrometer-related products including residual spectra, temperature, and other derived outputs. Inspect the actual files and their metadata before selecting a benchmark.
- Priority use: test whether a spectrum/line-analysis workflow can be run from downloadable measured spectra and compared with the supplied temperature or spectral-analysis reference.
- Acceptance gate: verify that a measured spectrum file is directly downloadable, identify whether it is raw or processed, and establish compatible calibration/line-identification information. A catalogue entry alone does not pass this gate.

### 2. FAIR-MAST public diagnostic archive

- Overview and access: https://www.ukaea.org/service/fair-mast/
- Data catalogue and level-1 examples: https://mastapp.site/level1-data.html
- Example shot: 30421, described in the catalogue's level-1 tutorial.
- The archive exposes experimental diagnostic signals in Zarr through a public API/S3 endpoint. The catalogue distinguishes level-1 signals from higher-level products; preserve that distinction in each benchmark.
- Priority use: reproducible time-series, FFT/PSD, cross-signal, event-timing, and equilibrium-grid checks where an independently reported or archive-supplied reference exists.
- Acceptance gate: record the shot, signal names, units, sample rate, time window, archive level, reference source, and exact comparison metrics. A successful download or plausible plot alone is not validation.

## Required benchmark record

Every benchmark added to the repository must record:

- benchmark ID and diagnostic/analysis feature;
- paper/data citation and stable URL;
- device, shot/pulse, diagnostic, data level, and file names;
- raw-data and reference-result availability;
- units, calibration, uncertainty/resolution, and preprocessing;
- independent implementation and software version;
- comparison metrics, acceptance threshold, observed error, and pass/fail;
- limitations and whether the benchmark can run offline in CI.

## Validation status vocabulary

- **independently-reproduced**: measured input and independent reference result were both obtained and quantitatively compared.
- **partially-reproduced**: some published numbers/derived tables were reconstructed, but raw measurements or a direct refit were unavailable.
- **processed-data-check**: data were already calibrated/processed by the source archive.
- **synthetic-benchmark**: algorithm checked only against generated data.
- **unit-tested**: formula or software behavior tested without an experimental reference.
- **not-validated**: insufficient data or reference evidence.

Do not upgrade a status without adding the source data/reference and a repeatable test.