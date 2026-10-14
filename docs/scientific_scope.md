# Scientific scope and validation policy

## Classification
- **M — measured:** detector signal or quantity available from a calibrated instrument.
- **D — derived:** computed from measured/calibrated data using an explicit relation.
- **I — inferred:** requires an inverse problem, physical model, atomic data, geometry, or additional assumptions.

A quantity can pass through more than one class (for example M→D/I). Catalog labels are guidance, not a substitute for the detailed measurement model.

## Software behavior
- Original imported values are not overwritten; processed results are separate outputs.
- The input file's SHA-256, name, byte count, columns and import timestamp can be exported with the analysis record.
- FFT and digital filtering require a sufficiently uniform sample interval; the core rejects visibly irregular sampling rather than silently pretending it is uniform.
- The spectral fitting code fits line-shape models, but a fit is not atomic line identification. Line blends, instrument function, wavelength calibration and background must be assessed separately.
- Doppler temperature is offered only as a conditional estimate for a corrected Gaussian width. It is not valid when other broadening or nonthermal populations dominate.
- Probe I–V fitting can conditionally estimate electron temperature and ion density only when the user selects fit windows and supplies the required collection geometry/species inputs. These are first-pass model-dependent estimates, not device-validated measurements; sheath expansion, current-sign convention, uncertainty and instrument response remain explicit limitations.
- Tomography requires a supplied response/geometry matrix. The code reports residual and regularization but does not establish that the matrix is physically correct.
- Greenwald density is a conventional empirical reference; ohmic power is an approximation; energy confinement requires a consistent loss-power definition.

## Validation levels
1. **Unit-tested numerical routines:** tested in this release.
2. **Synthetic-data demonstrations:** bundled for app smoke testing; never represented as experimental tokamak data.
3. **Device-specific validation:** not claimed. Requires independent benchmark datasets, machine calibration, documented conventions and expert review.
4. **Operational/control use:** explicitly out of scope.


## Registry inference and validation metadata
Each formula reference carries an `inference_class` separate from `validation_status`:
- `derived_quantity`: an explicit relation applied to correctly calibrated inputs;
- `model_dependent_estimate`: an empirical scaling or result conditional on a physical/engineering model;
- `inverse_problem_or_model_inference`: an inverse or reconstruction problem requiring geometry, regularization or a forward model;
- `reference_only`: documented knowledge without a mapped executable end-to-end implementation.

The validation field currently states that a complete diagnostic workflow has **not** been independently validated. Unit tests are evidence about tested numerical behavior, not independent experimental validation. The dashboard displays both fields. Diagnostic and formula IDs are stable within the checked-in registry ordering, and duplicate diagnostic names are rejected by regression tests.

## Input safety and provenance
The table importer applies a byte cap before parsing, bounds CSV row materialization, checks XLSX archive expansion and workbook dimensions, checks legacy XLS dimensions where the optional reader is installed, and checks Parquet row counts, row groups and uncompressed metadata size. Numeric coercions are recorded in the DataFrame import report and surfaced in the QC page. These checks reduce accidental/resource-exhaustion risk; they are not a sandbox for hostile documents.

## Algorithm-specific cautions
- Filter impact output includes the forward-backward filter response and raw/filtered RMS and dominant FFT peak. These are diagnostics for user review, not automatic proof that a cutoff is physically appropriate.
- Spectral fits report residual diagnostics and plot residuals. The reference line must be selected by the user; a line-shape fit does not identify atomic species or automatically correct all blends, detector response, Zeeman/fine structure or background errors.
- Tomography reports response-matrix coverage and, when matrix dimensions are manageable, SVD rank/condition diagnostics. Optional Monte Carlo perturbation propagates only supplied independent Gaussian measurement uncertainties; geometry/model error and regularization-choice uncertainty remain excluded.
- Channel-derived mode-number analysis reports phase residuals and leave-one-sensor-out sensitivity when enough sensors are available. This is not a confidence interval and cannot eliminate spatial aliasing.
- Explicit time-base resampling uses linear interpolation only for records without large gaps, refuses implicit downsampling, and exports method metadata. It cannot reconstruct missing samples.
- SVD of synchronized channels is data-driven: sensor order, channel units/scaling and time alignment affect the modes; singular-vector signs are arbitrary, and an SVD component index is not automatically a physical mode number.
- Morlet CWT requires a sufficiently uniform time base and a frequency grid below Nyquist. Frequencies whose wavelet support is too long for the record are rejected; edge-affected coefficients are flagged, and CWT power is relative rather than a calibrated PSD.
- E×B transport uses explicit radial/poloidal velocity sign conventions. The mean instantaneous product and the fluctuation covariance are both reported because they differ by mean-product terms; use the same definition as the target paper.

- The end-to-end Doppler workflow requires analyst-supplied line identification/rest wavelength, ion mass, wavelength references, and instrument width or a justified negligible-width assumption. It propagates fit and supplied-input uncertainties, but does not include wavelength-calibration covariance, response/background uncertainty, atomic line blends, Zeeman/fine structure or other model systematics.
