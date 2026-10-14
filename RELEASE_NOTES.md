# Release notes — v0.4.0

## Scientific correctness and security fixes
- Added direct cross-spectral phase extraction from synchronized multi-sensor channels with coherence weighting; manual phase fitting remains available as an advanced path.
- Welch PSD now exposes segment length, overlap and window controls. Spectral fitting supports absorption as well as emission lines and optional intensity-uncertainty weighting.
- Added an Eich-like exponential-convolved-with-Gaussian target heat-flux model with covariance-derived parameter uncertainties; the simpler exponential remains available.
- Tomography supports measurement-uncertainty weighting and an optional discrete L-curve curvature heuristic for λ selection, with the full curve and a cautionary note exported.
- Cached repeated time-series preparation, filtering and spectral calculations in Streamlit; plots of very long arrays are bounded to 20,000 display points.
- FAIR-MAST signal reduction now uses polyphase FIR anti-alias filtering; source/output sample rates and filter metadata are included in the provenance manifest. Irregular/gapped timebases are rejected before decimation.
- Mode-number fitting sorts sensors by angle, applies weights in the sorted order, reports a spatial alias limit, and rejects fits at the ambiguous spatial Nyquist boundary. Phase-only data cannot distinguish modes that have already aliased.
- Time-series preparation detects isolated dropped-sample gaps, not only global interval jitter.
- Langmuir ion-density estimation now raises an explicit error if no valid positive electron temperature was measured/fitted, instead of silently substituting Te=0.
- Plasma parameter calculations accept signed plasma current and toroidal field; Greenwald density uses |Ip|, while the ohmic-power sign follows Vloop·Ip.
- Spectral moments reject negative background-subtracted bins rather than clipping noise into a positive-looking spectrum.
- Cross-channel dashboard analysis now uses a common timestamp-validity mask and identical filter settings; lag and cross-phase sign conventions are labeled.
- Interferometer phase conversion documents that its output is line-integrated electron density (m⁻²) and exposes an explicit sign-convention argument.
- Public URL import blocks private/non-global destinations and validates redirect targets to mitigate SSRF.
- Package metadata now uses the English README, optional dependencies are declared in pyproject.toml, and the dashboard/package version is aligned to 0.4.0.
- Regression tests cover the defects above; final pass/fail status is determined by the current CI run.

## Scientific scope and limitations
- These are research analysis workflows requiring calibrated inputs, correct units, appropriate geometry, documented uncertainty assumptions, and valid measurement conditions.
- The supplied-flux-grid equilibrium page computes derivatives and magnetic-field components; it does not solve the Grad–Shafranov equation or replace EFIT.
- Phase-only mode fitting cannot recover the true mode number beyond the spatial Nyquist limit without independent information or a physical prior.
- Negative background-subtracted spectral bins require a justified noise/background treatment; they are not silently clipped.
- CI compilation, unit tests, and dashboard health checks do not validate every interactive workflow or every physical diagnostic model.
