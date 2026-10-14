# Architecture and module boundaries

The application keeps numerical algorithms independent from Streamlit and now routes the two largest analysis workspaces through dedicated page renderers:

- `tis/core.py`: numerical routines, explicit validation and scientific assumptions.
- `tis/catalog.py`: diagnostic/formula metadata and callable-workflow mapping.
- `tis/data_sources.py`: public archive access and source provenance.
- `tis/cli.py`: batch CLI and installed dashboard launcher.
- `tis/ui/time_series_page.py`: time-series/MHD, filter-impact audit, resampling, spectrogram, autocorrelation and integration UI.
- `tis/ui/spectral_page.py`: wavelength calibration, spectral fit/residuals, end-to-end Doppler, line-ratio and energy-spectrum UI.
- `app.py`: app setup, session state, common UI services and page routing.

Page renderers receive shared services as arguments rather than importing the dashboard module or reaching into its globals. This enables focused AppTest coverage and reduces coupling. Remaining workspace branches are still in `app.py` and should be extracted as their contracts are stabilized; this release does not claim that every UI page has been fully modularized.
