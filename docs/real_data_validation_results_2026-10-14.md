# Real-data validation results — 2026-10-14

Repository: `Ramin-Alipour/Tokamak-Insight-Suite`
Source run: https://github.com/Ramin-Alipour/Tokamak-Insight-Suite/actions/runs/38617475025
Run commit: `eb49c5965d4a15f4ce8154993944e691c87af21a`
Artifact: `mast-real-data-multidiagnostic-validation` (1,990,886 bytes; SHA-256 `6b77e8b4f76fc0a971c15922548184291e17532344cf97bb2b3e0ab8f2f3bd53`)

## Scope and verdict

This run **succeeded as a data-ingestion and generic numerical-pipeline integration test**. It is not a pass for every scientific diagnostic and it does not establish agreement with a publication. The public-data archive labels the source quality `Not Checked`; no independent calibrated reference time trace for the exact same signals was included in this run.

## Independently rechecked from the downloaded artifact

The artifact ZIP was downloaded and parsed separately. Sampling intervals, finite-value counts, descriptive statistics and Welch spectra were recomputed from the exported CSVs using SciPy.

| Dataset | Samples | Sampling rate | Range | Independent Welch peak | Status |
|---|---:|---:|---:|---:|---|
| MAST shot 29790, raw signal `xmo/omaha_1lz` | 70,000 exported from 1,400,000 | 100,000 Hz after documented anti-alias/polyphase downsampling | -2.80423 to 2.21852 (archive units unspecified) | 19,836.43 Hz | PASS for ingestion/timebase/PSD calculation only |
| MAST shot 30421, `amc/plasma_current` | 30,000 | 4,999.17 Hz | -40.8065 to 649.009 kA | 0.61025 Hz | PASS for ingestion and generic numerical checks only |
| MAST shot 30421, `ane/density` | 32,768 | 25,003.30 Hz | -7.02992e18 to 1.56886e20 m^-2 | 3.05216 Hz | Imported, but **physical interpretation not validated** |

The exported time arrays were finite and strictly increasing in all three CSVs. The independently computed sampling rates match the artifact manifests. The Welch dominant frequency for shot 29790 independently reproduces the manifest's 19,836.4258 Hz value to the displayed precision. This is a cross-implementation consistency check, not a publication benchmark; both calculations use the same exported samples.

## Scientific issues detected

1. **MAST 29790 signal metadata:** source group `xmo`, signal `omaha_1lz`; the export has no physical units and source quality is `Not Checked`. Do not label the resulting peak as a validated MHD mode without checking the diagnostic identity, calibration, signal conditioning, and an independent analysis/reference for the same shot and time window.
2. **MAST 30421 interferometer `density`:** its archive units are m^-2, i.e. a line-integrated quantity, not automatically a local electron density in m^-3. It contains negative values and a very large baseline/fluctuation relative to the mean. The generic pipeline only verifies finite samples, timebase and spectrum; it does not perform phase/fringe unwrapping, vibration correction, baseline validation, or convert line-integrated density into local density. Treat this signal as unvalidated and do not interpret its mean as physical density.
3. **MAST 30421 plasma current:** the archive is labelled analysed and units are kA. Import and generic spectrum calculation pass, but there is no paired publication/reference trace or documented quality flag comparison in this run. The 0.61025 Hz Welch peak should not be treated as a physical oscillation until the time interval, plasma phase, trend/DC contribution, and archive signal definition are checked.
4. **EFIT snapshot:** the workflow exports an existing archive grid at 0.2 s (65 × 65; finite fraction 1.0). This is a data-export check, not a Grad–Shafranov solve or equilibrium reconstruction validation. Coordinates, units, quality flags and current-density semantics still need confirmation against the archive documentation.

## Next validation gates

- Identify a documented, physically calibrated signal and an independent analysis result for the same MAST shot, channel and time interval.
- Compare spectral peak(s), PSD normalization/integrated variance and filter response against a separately implemented reference, with the selected segment/window/units explicitly recorded.
- Validate interferometer processing only after confirming the archive's signal definition and the required phase/fringe/vibration corrections.
- Validate equilibrium-derived fields against an independent equilibrium code/reference, not by checking that a grid exports.
- For each other TIS diagnostic family (Doppler fitting, Langmuir/triple probe, SVD/wavelet, heat flux, X-ray, tomography, neutron-rate and transport), use a diagnostic-specific raw dataset and a matched published or independently computed result. Generic MAST ingestion is not a substitute.

## Honest overall result

- Real-data retrieval and basic numerical processing: **PASS** for the three exported time series in this artifact.
- Independent recomputation of MAST 29790 Welch peak: **PASS** as a software cross-check (19,836.43 Hz).
- Publication-level or calibrated physical validation of these three signals: **NOT ESTABLISHED**.
- Full scientific validation of all analysis families: **INCOMPLETE**.


## Independent IR-T1 publication-table calculation (Doppler spectroscopy)

The open full-text of Alipour, Ghoranneviss & Salar Elahi, *AIP Advances* 7, 115303 (2017), DOI https://doi.org/10.1063/1.4990869, exposes Tables V–VII. I independently recalculated all 15 Table-VI Doppler widths using
`Delta_lambda_D = sqrt(Delta_lambda_measured^2 - Delta_lambda_resolution^2)`
and then recalculated all 15 Table-VII ion temperatures using the paper's printed equation
`Ti[eV] = M * 1.68216831e8 * (Delta_lambda_D/lambda0)^2`
with wavelengths and widths in nm and masses in amu.

- All 15 widths reproduce the published Doppler-width column to its rounding precision; largest absolute discrepancy from the printed value is about `6.98e-6 nm`.
- All 15 temperatures reproduce the published table within rounding. The largest relative difference is about `0.18%` (Fe I; the input width and published temperature are rounded).
- Ti I recalculates to `3.65884 eV`, which rounds to the published `3.66 eV`.

**Verdict: PASS — publication-table arithmetic/formula reconstruction, 15/15 rows.** This validates the conversion from the published rounded widths to the published rounded temperatures. It does **not** validate Gaussian fitting of raw CCD spectra, wavelength calibration, instrumental line-spread function, blends, or the physical assumption that the residual width is entirely Doppler broadening. The paper itself notes overlapping lines and decomposes total broadening into several terms, so those assumptions must remain explicit.

Source table evidence: https://www.researchgate.net/publication/320839921_First_investigation_on_plasma_impurities_of_the_IR-T1_Tokamak (Tables V–VII); publisher DOI above.
