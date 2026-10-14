# Tokamak Insight Suite (TIS)

Python-based tokamak plasma diagnostics and data-analysis toolkit, version 0.3.0.

## Run locally

Python 3.10 or newer. Install dependencies with `pip install -r requirements.txt`, then launch the interface using `streamlit run app.py`.

The project includes data-quality checks, time-series and spectral analysis, Langmuir probe routines, calibration, tomography helpers, diagnostic/formula registries, and provenance-aware exports.

Scientific outputs depend on instrument calibration, geometry, uncertainty, and explicit model assumptions. This is a research analysis tool, not a tokamak control or protection system. The v0.3.0 source package passed 39 tests in its build environment; real-device validation remains necessary.
