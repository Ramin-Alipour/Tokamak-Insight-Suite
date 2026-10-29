# Analysis 2 — Langmuir probe

## Scope
This workflow analyzes a calibrated, quasi-static, single-probe I–V sweep. It reports cleaned and voltage-sorted samples, descriptive current tails, the floating-potential crossing(s), a maximum-slope plasma-potential proxy, and optional fit-window-based electron temperature and ion-branch metrics.

## Inputs and preprocessing
- `voltage` (V) and `current` (A): paired one-dimensional arrays.
- At least eight valid samples and eight distinct voltage values are required. Non-finite pairs are discarded; duplicate voltages are aggregated by averaging current; pairs are sorted together by voltage.
- `current_convention`: `electron_positive` (default) or `ion_positive`. The latter is sign-flipped before analysis.
- Returned audit arrays: `processed_voltage_V`, `processed_current_A`, and `dIdV_A_per_V` expose the exact sorted, duplicate-aggregated, polarity-normalized samples and derivative used downstream. They are outputs of the calculation, not additional inferred measurements.
- Optional fit intervals must have two finite, strictly increasing voltage bounds within the measured sweep. The electron-retarding fit requires at least four selected samples and at least three samples above the initial low-voltage-tail baseline estimate. The ion-branch fit requires at least three selected samples. Intervals are selected and justified by the analyst; the software does not automatically establish the physical regime.

## Physics and outputs
For an electron-retarding branch approximated by a Maxwellian exponential plus a fitted current offset,
\[
I(V)=I_0+A\exp\!\left[s(V-V_{\rm ref})\right],\qquad
T_e[\mathrm{eV}]=\frac{1}{s[\mathrm{V}^{-1}]}.
\]

The implementation performs a joint nonlinear fit of the current offset, positive amplitude, and positive exponential slope over the selected interval. It does not perform an unweighted log-linear fit after subtracting a fixed baseline. The reported covariance-based standard error is conditional on this three-parameter model and the selected interval.

The maximum of numerical `dI/dV` is reported only as a plasma-potential proxy. The optional ion-branch fit is a linear current-versus-voltage characterization and reports slope, intercept, R-squared, residual RMS, normalized RMSE, point count, and an explicit fit-review status. Its median current is a selected-window estimate, not a geometry-derived density or particle flux. Fit diagnostics describe numerical agreement only; they do not certify that the selected interval is physically an ion-saturation regime.

The floating-potential result includes all detected zero crossings. A zero-current plateau is represented by the midpoint of the plateau, not the first adjacent zero pair. `floating_potential_V` is populated only for exactly one unique crossing; no crossing or multiple crossings produces `null` with an explicit status. The standalone `floating_potential()` helper follows the same rule, ignores non-finite pairs, averages currents at duplicate voltages, and rejects malformed array shapes rather than flattening them.

## Validation and limitations
Focused regression tests cover crossing interpolation, zero plateaus, ambiguous crossings across both the main analysis and helper APIs, malformed helper inputs, fit-range validation, current polarity, duplicate voltages, and non-finite sample handling. Fit-window choice, probe calibration, RF pickup, magnetic-field effects, collisions, secondary emission, sheath expansion, surface condition, and sweep direction remain experiment-specific. The reported temperature regression error is statistical only; calibration and model uncertainty are not propagated. Do not interpret a numerical proxy or model estimate as independently validated experimental truth.


## Auditable quality and fit diagnostics

The result includes an `input_quality` object with original pair count, finite pairs retained, non-finite pairs discarded, duplicate-voltage pairs averaged, processed unique-voltage count, voltage span, and normalized current convention. When an electron-retarding fit is enabled, the output also includes fit residual RMS (A), normalized RMSE, R-squared, fit-point count, and a status explicitly instructing the analyst to review the residuals and R-squared. These diagnostics do not certify that the selected voltage interval is physically valid and do not replace calibration or systematic uncertainty analysis.

Regression tests: `tests/test_langmuir_qc_contract.py`. These are software regression checks only; they are not experimental validation.


## Geometry-independent scope and optional density estimate

The core analysis accepts already-calibrated voltage and current samples and does not infer instrument calibration or tokamak geometry. An optional ion-density estimate is available only when the user supplies both analyst-selected electron-retarding and ion-branch fit windows, a positive effective collection area, a positive assumed ion mass, a positive coefficient from an explicitly selected sheath/Bohm model, and a non-empty model label. The estimate uses

\[
 n_i = \frac{|I_{i,\mathrm{sat}}|}{C_s e A_p\sqrt{eT_e/m_i}},
\]

where \(T_e\) is in eV, \(I_{i,\mathrm{sat}}\) is the negative ion-saturation current in the normalized electron-positive convention, and the supplied coefficient must match the stated probe geometry and plasma regime. This is a model-based estimate under a single-dominant-ion-species assumption, not a direct or experimentally validated measurement. No density is produced from descriptive tail fallbacks. Automatic uncertainty propagation for density and its model inputs is not implemented. Particle flux remains outside this analysis.


## Fit-window review and interpretation status

The analysis reports numerical screening separately from physical interpretation. An electron fit is marked as a numerical review pass only when its R-squared is at least 0.98 and normalized RMSE is at most 0.1; the ion-branch fit uses the same numerical thresholds. These are transparent screening thresholds, not universal physics acceptance criteria. The electron branch must still be checked for an appropriate exponential retarding regime. For an ion branch, the optional relative-current-change indicator is a flatness candidate only; it cannot certify ion saturation. Outputs explicitly retain `physical_regime_confirmed: false`.

The maximum-`dI/dV` voltage is labelled as a proxy-only plasma-potential indicator. It is not automatically promoted to a validated plasma potential. Tail-derived ion current remains descriptive only and is never sufficient for density inversion.

## Optional first-order density uncertainty propagation

For density inversion, callers may supply the absolute standard uncertainty of the selected ion current and relative standard uncertainties for collection area, ion mass, sheath coefficient and electron temperature. The code combines the fitted statistical temperature uncertainty with the supplied relative temperature component, then propagates the supplied terms in quadrature to a first-order density uncertainty. The output lists each propagated component and clearly reports that the result is partial: unprovided calibration covariance, correlated errors, model discrepancy, fit-window sensitivity and other systematic effects are not inferred. With no uncertainty inputs, density uncertainty fields are null and the status states that propagation was not performed.

## Declared scope

This workflow analyzes a single calibrated Langmuir-probe I–V sweep. It does not claim full support for multi-probe arrays, triple probes, probe-system geometry reconstruction or all electrostatic-probe configurations. Other configurations require their own explicitly defined models and execution contracts.
