# Release notes — v0.3.0

## Dashboard and workflow expansion
- Redesigned the Streamlit control surface as a dark navy/blue workspace with left navigation, project/shot context, persistent data upload, quick tools, dashboard preview, and dedicated analysis pages, following the saved Tokamak Insight Suite dashboard reference.
- Added a persistent single-session dataset workflow and data integrity hash.
- Added Langmuir I–V analysis with user-selected ion-saturation and electron-retarding fit ranges, floating-potential interpolation, numerical dI/dV proxy, Te estimate from the log-linear region after ion-current baseline subtraction, and optional model-based ion-density estimate.
- Added multi-sensor phase-to-mode-number fit, covariance-based particle/heat-flux estimates, target heat-flux exponential fit, profile gradient descriptors, spectrum moments, neutron-rate conversion with explicit efficiency/geometry factors, supplied-flux-grid magnetic-field derivatives, axisymmetric Abel inversion, and axisymmetric emissivity-to-radiated-power integration.
- Expanded dashboard navigation to include data QC, time series/MHD, spectroscopy, probe analysis, plasma parameters, equilibrium-grid tools, tomography/bolometry, advanced analysis, diagnostic catalog, and provenance/export.
- Added automated tests for the new analysis functions.
- Added O-mode reflectometry cutoff-density calculation, ECE cyclotron-resonance field estimate, and instrument-response-corrected spectral line ratios.
- Automated test suite: 39 passed in the build environment.

## Scope and limitations
- These are research analysis workflows that require calibrated input data, correct units, appropriate geometry, and documented assumptions. They do not automatically validate an experimental diagnostic.
- The supplied-flux-grid equilibrium page computes derivatives and magnetic-field components; it does not solve the Grad–Shafranov equation or replace EFIT.
- Abel inversion assumes axisymmetry and simplified chord geometry. General tomography needs the real response matrix.
- Atomic collisional-radiative models, full ray tracing, validated fast-ion orbit following, and device-specific equilibrium reconstruction remain separate specialist solvers.
- Streamlit is not installed in the build environment, so the GUI could be syntax-compiled but not launched in a browser during this build. Install dependencies with `pip install -r requirements.txt` before launch.
