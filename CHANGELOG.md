# Changelog

## Unreleased — 2026-10-30

### Thomson Scattering / Analysis 1
- Added full-covariance joint generalized least-squares inference for non-collective Maxwellian and Kappa spectra, including cross-channel physical-parameter covariance.
- Added residual-based model-discrepancy covariance diagnostics and stricter joint-fit acceptance / physical-validity checks.
- Added Monte Carlo refitting for supplied Collective ion-temperature, ion/electron-flow, laser-wavelength and instrument-response uncertainties.
- Routed Advanced Analysis to the canonical Thomson inference interface and converted the legacy single-channel Gaussian API into a compatibility adapter.
- Added explicit physical-parameter mapper contracts for custom spectral models.
- Extended the Thomson dashboard to accept channel-major spectral covariance and Collective external-parameter uncertainty inputs.

### Microwave Collective Thomson Scattering
- Added covariance-aware CTS residual whitening, response/external-parameter Monte Carlo propagation and separate model-discrepancy covariance reporting.
- Added differential-evolution, adaptive real-coded genetic and regularized inversion modes. The genetic implementation is informed by Zhang et al. (2024), not an exact reproduction of the paper-specific fitness function.
- Added tabulated 1-D fast-ion susceptibility and regularized non-negative tabulated VDF inversion with softmax parameterization, second-difference regularization and propagated VDF-node covariance.
- Added a dedicated microwave CTS dashboard workflow and scientific-scope documentation. The implementation remains an unmagnetized electrostatic longitudinal model, not a full magnetized electromagnetic CTS simulator.

### Tests and scientific traceability
- Added synthetic tests for joint channel covariance, Kappa joint inference, Collective external uncertainty, CTS genetic/regularized inversion, tabulated fast-ion VDFs, response uncertainty and custom-model physical contracts.
- Expanded FORMULA_REGISTRY.json and DIAGNOSTIC_REGISTRY.json with explicit equations, units, uncertainty models, assumptions, limitations and scientific references.
- No independent experimental validation is claimed; all new regression coverage is mathematical/software testing with controlled synthetic data.

## 0.5.0 — 2026-10-15

### Bugs fixed
- **pandas ≥ 3 crash:** the "drop units row" import option used `pd.to_numeric(errors="ignore")`, which pandas 3 removed; replaced by a loss-free coercion.
- **Text encodings:** CSV/TXT import failed on Windows-1252/Latin-1 files (e.g. `Température`, `µs`). Import now tries UTF-8 (BOM-tolerant) → Windows-1252 → Latin-1, accepts an explicit encoding, and records the encoding actually used in provenance.
- **Mode-number sign:** the fitted `n` had the opposite sign to the usual wave convention cos(ωt − nθ) and this was undocumented. Both fitting functions now take an explicit `convention`, report it in the result, and the UI defaults to the usual convention (the library default is unchanged for backward compatibility).
- **Langmuir analysis:** electron temperature now carries a statistical standard error; an `ion_positive` current convention is supported (previously failed with an unhelpful "needs 4 points" error); a positive ion-saturation current now raises a sign warning.
- **Analysis history** was re-appended on every Streamlit rerun (duplicates); now de-duplicated by result hash and logged with its SHA-256.
- **Registry drift:** `DIAGNOSTIC_REGISTRY.json` was stale (duplicate NPA entry, no implementation status) and disagreed with `catalog.py`. Regenerated and guarded by a test.
- **Docker image** did not install the optional data dependencies, so the Public Data Access page could not work in the container; it now installs `.[data]`.
- CI packaging workflow no longer references stale v0.3.0/v0.4.0 file names.
- Removed a duplicated AppTest module.

### New capabilities
- **Events & Transients page / `tis.events`:** ELM detection (rolling-median baseline, MAD threshold + prominence), sawtooth crash/period/amplitude detection (with spurious-peak rejection), current-quench analysis with the ITER 80 %→20 % convention and runaway-plateau detection; interval statistics.
- **Confinement & Scalings page / `tis.plasma`:** Greenwald fraction, q*, ITER-formula q95, β_t/β_p/β_N, collisionality ν*ₑ, Coulomb logarithm, Spitzer resistivity, gyro-radii/Alfvén speed, IPB98(y,2) τ_E and H98, Martin-08 L–H threshold, triple product. Checked against ITER design values in tests.
- **Gap handling / `tis.signals`:** sampling audit, explicit gap policies (strict / longest segment / interpolate short gaps), uniform resampling that never invents data inside long gaps, Hampel despiking, polynomial detrending.
- **Linear drift correction** for time-series integration (pre- and post-shot quiet windows).
- **Self-contained HTML report** with provenance, results and figures (all text escaped).
- `tis events` and `tis scalings` CLI commands; loaded `z_m`/`R=…_m` grid tables can be used on the Equilibrium page (with an explicit "this is ψ" confirmation).

### Known limitations (unchanged or still open)
See README → "Scientific scope and limitations" and RELEASE_NOTES.

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
