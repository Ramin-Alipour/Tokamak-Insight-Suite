# External-data scientific validation campaign

**Rule:** no papers authored by Ramin Alipour are eligible as validation references for this campaign.

## Dataset candidates with raw and analyzed products

### MAST shot 27643 — central-column toroidal Mirnov analysis
- Catalog record: https://opendata.ukaea.uk/mast-data/?ccfe-ta=493&ccfe-tb=9040&ccfe-tc=1158204
- Raw/processed object: `ama27643.nc`, NetCDF4/HDF5.
- Signals include `AMA_N=ODD SIGNAL`, `AMA_N=ODD AMPLITUDE`, `AMA_N=ODD FREQUENCY`, `AMA_N=2 SIGNAL`, `AMA_N=2 AMPLITUDE`, `AMA_N=2 FREQUENCY`.
- Suitable comparison: independently recompute spectrum/peak frequency and compare with catalogued analyzed amplitude/frequency only after verifying units, timebase, filtering, and analysis-window metadata.
- Status: candidate confirmed in catalog; binary file not yet downloaded/analysed in this execution environment.

### MAST shot 29790 — high-frequency Mirnov OMAHA
- Catalog: https://mastapp.site/level1-data.html
- Object: `xmo029790.nc`, raw CDF; signal group includes OMAHA channels.
- Suitable comparison: timebase/quality validation, FFT/Welch PSD, spectral peak stability; requires a matching published or archived analyzed output for the same channels and window.
- Status: raw object identified in catalog; reference-output equivalence still needs confirmation.

### GOLEM shot 30622 — broad diagnostics and analysis products
- Data page: https://golem.fjfi.cvut.cz/shots/30622/Data.php
- Python/data-access guide: https://golem.fjfi.cvut.cz/wiki/Handling/DataAccess/DataProcessing
- Listed products include plasma current, toroidal field, interferometer-derived density, probe electron temperature/floating potential, spectrometer spectra and calibrated spectra, impurity/temperature products, HXR, and Mirnov channels.
- Suitable comparisons: signal ingestion/timebase; current and field trends; spectral fitting only if the numerical spectrum and wavelength calibration can be retrieved; probe-derived quantities only with diagnostic assumptions and matching metadata.
- Status: data and analyzed products are listed publicly; files not downloaded/analysed in this execution environment.

### GOLEM shot 35755 — Mirnov plasma-position notebook
- Analysis notebook: https://golem.fjfi.cvut.cz/shots/35755/Diagnostics/LimiterMirnovCoils/analysis.html
- Describes raw Mirnov loading, toroidal-field subtraction, integration and position reconstruction; gives code/data URL patterns and a vacuum-shot reference.
- Suitable comparison: reproduce the published/notebook algorithm on the exact shot and compare the resulting position trace with the notebook's reference output.
- Status: analysis workflow identified; exact numerical rerun not yet executed here.

### TCV L–H transition database
- Dataset: https://zenodo.org/records/14996664
- Includes HDF5 data and a Python notebook to reproduce specific figures.
- Suitable comparison: reproduce only the included paper figures/derived quantities that map to implemented TIS workflows.
- Status: candidate; not a raw diagnostic benchmark for all TIS analysis families.

## Required validation record per case

Record source URL, shot, diagnostic/channel, raw-vs-derived status, file SHA-256, timebase and units, calibration/response, selected interval, preprocessing settings, software version/commit, independent reference value/trace, TIS result, absolute/relative discrepancy, uncertainty, and status.

Statuses: `AGREES`, `DISAGREES`, `UNASSESSED`, `BLOCKED`.

**Do not call a workflow experimentally validated** merely because synthetic/golden tests pass, a source lists a derived quantity, or an analyzed file is mistaken for raw data. MAST `ama` files may themselves contain analyzed products; inspect provenance before treating any signal as raw.
