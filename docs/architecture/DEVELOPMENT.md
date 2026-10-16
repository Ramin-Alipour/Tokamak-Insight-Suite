# Developer guide

## Separation of concerns

- `DIAGNOSTIC_REGISTRY.json`: canonical diagnostic metadata and execution contracts.
- `FORMULA_REGISTRY.json`: formula reference and explicit implementation-link status.
- `tis/contracts.py`: typed interpretation and validation of diagnostic contracts.
- `tis/data_model.py`: measurement metadata, unit conversion and quality state.
- `tis/project.py`: persistent projects and reproducible analysis-run records.
- `tis/signal_processing.py`: filtering, FFT, Welch PSD, correlation and spectrograms.
- `tis/spectral_analysis.py`: spectral line fitting and fit-quality metrics.
- `tis/tomography.py`: response-matrix tomography, regularization selection and Abel inversion.
- `tis/uncertainty.py`: correlated first-order covariance propagation and seeded Monte Carlo propagation.
- `tis/ui/catalog_page.py` and `tis/ui/project_page.py`: separated Streamlit pages.
- `tokamak_analysis/`: public numerical analysis functions.
- `tis/core.py` and `app.py`: compatibility entry points during incremental refactoring; new UI pages and numerical domains should be added in their dedicated modules first.

## Adding or promoting an analysis

1. Define the physical quantity, accepted input shape, units and timebase semantics.
2. Declare required preprocessing and model assumptions.
3. Implement the smallest numerical function that can be tested independently.
4. Define outputs, units, uncertainty behavior, quality flags and failure modes.
5. Add known-answer, dimensional, boundary, invalid-input and API-contract tests.
6. Add registry function links only after the linked function exists.
7. Promote execution level only after the required contract fields are documented.
8. Distinguish algorithmic tests from comparisons against measured experimental data.

## Test commands

```bash
python -m pip install -e ".[test]"
python -m pytest tests/test_contracts.py tests/test_data_model.py tests/test_project.py
python -m pytest
```

The full suite requires the declared project dependencies, including Streamlit for UI smoke tests. A skipped or uncollected UI test is not a successful UI test.
