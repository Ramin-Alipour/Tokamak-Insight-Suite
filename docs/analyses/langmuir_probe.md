# Analysis 2 — Langmuir probe

## Scope
This workflow analyzes a calibrated, quasi-static, single-probe I–V sweep. It reports cleaned and voltage-sorted samples, descriptive current tails, the floating-potential crossing(s), a maximum-slope plasma-potential proxy, and optional fit-window-based electron temperature and ion-branch metrics.

## Inputs and preprocessing
- `voltage` (V) and `current` (A): paired one-dimensional arrays.
- At least eight valid samples and eight distinct voltage values are required. Non-finite pairs are discarded; duplicate voltages are aggregated by averaging current; pairs are sorted together by voltage.
- `current_convention`: `electron_positive` (default) or `ion_positive`. The latter is sign-flipped before analysis.
- Optional fit intervals must have two finite, strictly increasing voltage bounds within the measured sweep. The electron-retarding fit requires at least four selected samples and at least three samples above the initial low-voltage-tail baseline estimate. The ion-branch fit requires at least three selected samples. Intervals are selected and justified by the analyst; the software does not automatically establish the physical regime.

## Physics and outputs
For an electron-retarding branch approximated by a Maxwellian exponential plus a fitted current offset,
\[
I(V)=I_0+A\exp\!\left[s(V-V_{\rm ref})\right],\qquad
T_e[\mathrm{eV}]=\frac{1}{s[\mathrm{V}^{-1}]}.
\]

The implementation performs a joint nonlinear fit of the current offset, positive amplitude, and positive exponential slope over the selected interval. It does not perform an unweighted log-linear fit after subtracting a fixed baseline. The reported covariance-based standard error is conditional on this three-parameter model and the selected interval.

The maximum of numerical `dI/dV` is reported only as a plasma-potential proxy. The optional ion-branch fit is a linear current-versus-voltage characterization; its median current is reported as a selected-window estimate, not a geometry-derived density or particle flux.

The floating-potential result includes all detected zero crossings. A zero-current plateau is represented by the midpoint of the plateau, not the first adjacent zero pair. `floating_potential_V` is populated only for exactly one unique crossing; no crossing or multiple crossings produces `null` with an explicit status. The standalone `floating_potential()` helper follows the same rule, ignores non-finite pairs, averages currents at duplicate voltages, and rejects malformed array shapes rather than flattening them.

## Validation and limitations
Focused regression tests cover crossing interpolation, zero plateaus, ambiguous crossings across both the main analysis and helper APIs, malformed helper inputs, fit-range validation, current polarity, duplicate voltages, and non-finite sample handling. Fit-window choice, probe calibration, RF pickup, magnetic-field effects, collisions, secondary emission, sheath expansion, surface condition, and sweep direction remain experiment-specific. The reported temperature regression error is statistical only; calibration and model uncertainty are not propagated. Do not interpret a numerical proxy or model estimate as independently validated experimental truth.


## Auditable quality and fit diagnostics

The result includes an `input_quality` object with original pair count, finite pairs retained, non-finite pairs discarded, duplicate-voltage pairs averaged, processed unique-voltage count, voltage span, and normalized current convention. When an electron-retarding fit is enabled, the output also includes fit residual RMS (A), normalized RMSE, R-squared, fit-point count, and a status explicitly instructing the analyst to review the residuals and R-squared. These diagnostics do not certify that the selected voltage interval is physically valid and do not replace calibration or systematic uncertainty analysis.

Regression tests: `tests/test_langmuir_qc_contract.py`. These are software regression checks only; they are not experimental validation.


## Geometry-independent scope

This analysis accepts calibrated voltage and current samples only. It does not infer ion density or particle flux because those require additional probe-collection geometry and a declared sheath/ion model. No tokamak dimensions, converter geometry, or device-specific geometry are inferred. The ion-branch fit is retained only as a current-versus-voltage characterization.
