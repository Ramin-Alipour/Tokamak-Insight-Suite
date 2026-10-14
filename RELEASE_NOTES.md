# Release notes — v0.4.0

## Scientific correctness and security fixes
- Added direct cross-spectral phase extraction from synchronized multi-sensor channels with coherence weighting; manual phase fitting remains available as an advanced path.
- Welch PSD now exposes segment length, overlap and window controls. Spectral fitting supports absorption as well as emission lines and optional intensity-uncertainty weighting.
- Added an Eich-like exponential-convolved-with-Gaussian target heat-flux model with covariance-derived parameter uncertainties; the simpler exponential remains available.
- Tomography supports measurement-uncertainty weighting and an optional discrete L-curve curvature heuristic for λ selection, with the full curve and a cautionary note exported. A 2-D straight-ray path-length matrix can be built from line-of-sight endpoints and pixel edges; detector response and equilibrium mapping remain outside this geometry model.
- Cached repeated time-series preparation, filtering and spectral calculations in Streamlit; min/max plot decimation preserves narrow spikes, uploads are keyed to file identity, and remote data are not overwritten on rerun.
- FAIR-MAST signal reduction now uses polyphase FIR anti-alias filtering; source/output sample rates and filter metadata are included in the provenance manifest. Irregular/gapped timebases are rejected before decimation.
- Mode-number fitting sorts sensors by angle, excludes zero-weight sensors before circular-phase fitting, recalculates the spatial alias limit from active sensor geometry, and searches integer modes using circular residuals. Phase-only data cannot distinguish modes that have already aliased.
- Time-series preparation detects likely missing-sample gaps while allowing timestamp quantization at high sample rates; the sample rate is estimated from the full-record mean interval after gap checks.
- Langmuir ion-density estimation now raises an explicit error if no valid positive electron temperature was measured/fitted, instead of silently substituting Te=0.
- Plasma parameter calculations accept signed plasma current and toroidal field; Greenwald density uses |Ip|, while the ohmic-power sign follows Vloop·Ip.
- Spectral moments preserve negative background-subtracted bins rather than clipping noise; when signed net area or variance makes physical moments undefined, the result carries an explicit invalid-status instead of crashing the report.
- Cross-channel dashboard analysis now uses a common timestamp-validity mask and identical filter settings; lag and cross-phase sign conventions are labeled.
- Interferometer phase conversion documents that its output is line-integrated electron density (m⁻²) and requires an explicit instrument-calibrated sign convention; no sign is assumed by default.
- Public URL import blocks non-global destinations, pins validated DNS addresses to sockets, validates redirect targets and disables environment proxies to mitigate SSRF/DNS rebinding.
- Package metadata now uses the English README, optional dependencies are declared in pyproject.toml, and the runtime, dashboard, and package version are aligned to 0.4.0.
- Voigt fits require sufficient samples, reject singular covariance and report approximate FWHM and integrated area. Regression tests cover these behavior changes; CI must pass before release.

## Scientific scope and limitations
- These are research analysis workflows requiring calibrated inputs, correct units, appropriate geometry, documented uncertainty assumptions, and valid measurement conditions.
- The supplied-flux-grid equilibrium page computes derivatives and magnetic-field components; it does not solve the Grad–Shafranov equation or replace EFIT.
- Phase-only mode fitting cannot recover the true mode number beyond the spatial Nyquist limit without independent information or a physical prior.
- Negative background-subtracted spectral bins require a justified noise/background treatment; they are not silently clipped.
- CI compilation, unit tests, and dashboard health checks do not validate every interactive workflow or every physical diagnostic model.
