# Release notes — v0.3.0

## Post-release engineering corrections
- Corrected FFT behavior so `detrend=False` preserves the DC component instead of always removing the mean.
- Filter cutoffs explicitly set to zero, negative, or non-finite values are now rejected rather than silently treated as omitted.
- Spectral calibration now validates polynomial degree and matching input-array lengths before fitting.
- Axisymmetric radiated-power integration rejects negative emissivity grids instead of returning a potentially nonphysical total.
- Added regression tests for DC handling, invalid filter/calibration inputs, reflectometry cutoff scaling, electron-cyclotron resonance, analytic equilibrium derivatives, radiated-power integration, and synthetic heat-flux decay recovery.
- Added calibrated time-series integration with optional baseline subtraction, CSV export, and a dashboard workflow for derivative-type diagnostics; sensor calibration and initial conditions remain explicit user inputs.
- Added short-time power spectrogram analysis for transient/frequency-evolving signals, including window validation, a dashboard heatmap and CSV matrix export.
- Added a continuous integration workflow to compile sources, run the test suite, and check the Streamlit HTTP health endpoint on pushes and pull requests. These checks must pass in GitHub Actions before the changes are considered verified.


## Dashboard and workflow expansion
- Redesigned the Streamlit control surface as a dark navy/blue workspace with left navigation, project/shot context, persistent data upload, quick tools, dashboard preview, and dedicated analysis pages, following the saved Tokamak Insight Suite dashboard reference.
- Added a persistent single-session dataset workflow and data integrity hash.
- Added Langmuir I–V analysis with user-selected ion-saturation and electron-retarding fit ranges, floating-potential interpolation, numerical dI/dV proxy, Te estimate from the log-linear region after ion-current baseline subtraction, and optional model-based ion-density estimate.
- Added multi-sensor phase-to-mode-number fit, covariance-based particle/heat-flux estimates, target heat-flux exponential fit, profile gradient descriptors, spectrum moments, neutron-rate conversion with explicit efficiency/geometry factors, supplied-flux-grid magnetic-field derivatives, axisymmetric Abel inversion, and axisymmetric emissivity-to-radiated-power integration.
- Expanded dashboard navigation to include data QC, time series/MHD, spectroscopy, probe analysis, plasma parameters, equilibrium-grid tools, tomography/bolometry, advanced analysis, diagnostic catalog, and provenance/export.
- Added automated tests for the new analysis functions.
- Added O-mode reflectometry cutoff-density calculation, ECE cyclotron-resonance field estimate, and instrument-response-corrected spectral line ratios.
- Automated test suite: 66 test cases, including numerical regression and invalid-input checks; current CI status is recorded in GitHub Actions.

## Scope and limitations
- These are research analysis workflows that require calibrated input data, correct units, appropriate geometry, and documented assumptions. They do not automatically validate an experimental diagnostic.
- The supplied-flux-grid equilibrium page computes derivatives and magnetic-field components; it does not solve the Grad–Shafranov equation or replace EFIT.
- Abel inversion assumes axisymmetry and simplified chord geometry. General tomography needs the real response matrix.
- Atomic collisional-radiative models, full ray tracing, validated fast-ion orbit following, and device-specific equilibrium reconstruction remain separate specialist solvers.
- CI installs the declared dependencies, compiles the Python sources, runs the full test suite, and performs a Streamlit HTTP health smoke test. This checks application startup, not every interactive page or scientific workflow.
