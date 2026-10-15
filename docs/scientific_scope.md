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
