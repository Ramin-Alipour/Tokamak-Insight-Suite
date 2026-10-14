# Changelog

## 0.4.0 — 2026-10-14

### Correctness and scientific safeguards
- Preserve channel names containing `#`; expose CSV/TXT delimiter, decimal, header and skipped-row parsing options.
- Accept expected microsecond timestamp quantization while retaining detection of likely dropped-sample gaps.
- Require an explicit sampling rate when no time coordinate is supplied.
- Avoid defaulting probe ion mass or interferometer phase sign to an assumed physical convention.
- Use coherence-aware frequency selection and a segment-count-derived coherence significance threshold for channel mode fitting.
- Exclude zero-weight sensors from phase fitting and search integer spatial modes.
- Normalize inverse-problem response matrices and default to gradient regularization; add a geometry-derived straight-ray 2-D response-matrix builder.
- Preserve signed background-subtracted spectral bins and distinguish counts-per-bin from spectral density.
- Add fit covariance checks, standard errors and parameter-bound warnings to heat-flux fitting.
- Interpolate profile half-height crossings and explicitly decline to infer a pedestal without a model/window definition.
- Convert non-finite values to JSON null in exports.

### Application, security and maintenance
- Bound cache entries, plot with min/max decimation, and use per-call Matplotlib figures.
- Record import settings and clear stale source manifests when a new file is imported.
- Make data-import and analysis provenance more explicit.
- Bind public URL fetches to validated global IPs, validate redirects and disable environment proxies.
- Limit input size/row counts, validate XLSX archive expansion and run the container as a non-root user.
- Add AppTest dashboard smoke tests and CI coverage across Python 3.10, 3.11 and 3.12.
- Centralize dependency declarations in `pyproject.toml`; add citation, contribution and security guidance.

### Scope limitations
This release does not claim a Grad–Shafranov equilibrium solver, a collisional-radiative atomic model, device-specific calibration, or validation of every catalog entry. Geometry matrices, atomic data, diagnostic response and uncertainty models remain explicit user/device inputs where applicable.
