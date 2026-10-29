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

where \(T_e\) is in eV, \(I_{i,\mathrm{sat}}\) is the negative ion-saturation current in the normalized electron-positive convention, and the supplied coefficient must match the stated probe geometry and plasma regime. This is a model-based estimate under a single-dominant-ion-species assumption, not a direct or experimentally validated measurement. No density is produced from descriptive tail fallbacks. Optional first-order uncertainty propagation for density is implemented only for uncertainty components explicitly supplied by the caller. It is partial, not a complete propagation of calibration covariance, correlated errors, fit-window sensitivity, model discrepancy, or other systematic effects. Particle flux remains outside this analysis.


## Fit-window review and interpretation status

The analysis reports numerical screening separately from physical interpretation. An electron fit is marked as a numerical review pass only when its R-squared is at least 0.98 and normalized RMSE is at most 0.1; the ion-branch fit uses the same numerical thresholds. These are transparent screening thresholds, not universal physics acceptance criteria. The electron branch must still be checked for an appropriate exponential retarding regime. For an ion branch, the optional relative-current-change indicator is a flatness candidate only; it cannot certify ion saturation. Outputs explicitly retain `physical_regime_confirmed: false`.

The maximum-`dI/dV` voltage is labelled as a proxy-only plasma-potential indicator. It is not automatically promoted to a validated plasma potential. Tail-derived ion current remains descriptive only and is never sufficient for density inversion.

## Optional first-order density uncertainty propagation

For density inversion, callers may supply the absolute standard uncertainty of the selected ion current and relative standard uncertainties for collection area, ion mass, sheath coefficient and electron temperature. The code combines the fitted statistical temperature uncertainty with the supplied relative temperature component, then propagates the supplied terms in quadrature to a first-order density uncertainty. The output lists each propagated component and clearly reports that the result is partial: unprovided calibration covariance, correlated errors, model discrepancy, fit-window sensitivity and other systematic effects are not inferred. With no uncertainty inputs, density uncertainty fields are null and the status states that propagation was not performed.

## Declared scope

This workflow analyzes a single calibrated Langmuir-probe I–V sweep. It does not claim full support for multi-probe arrays, triple probes, probe-system geometry reconstruction or all electrostatic-probe configurations. Other configurations require their own explicitly defined models and execution contracts.


## Canonical entry point and compatibility contract

The canonical calculation pipeline is `tis.probe_analysis.langmuir_iv_analysis`. The public diagnostic wrapper is `tokamak_analysis.langmuir_probe.analyze_langmuir_probe`. The generic `tokamak_analysis.probe_analysis.analyze_probe_voltage_current` entry point delegates to that same wrapper and adds legacy output aliases only; it does not maintain a separate Langmuir fitting implementation. All three entry points therefore share the same minimum of eight valid samples and eight distinct voltages, polarity normalization, fit calculations, optional density guards, and canonical output fields. Legacy names are compatibility aliases and should not define new result contracts.

## Software completion criterion

The registry label `complete_with_declared_scope` means that the implemented software contract and regression tests cover the explicitly declared calibrated single-probe I–V scope. It does not claim independent experimental validation, full systematic uncertainty propagation, automatic physical-regime certification, or support for every probe geometry. Those exclusions are intentional scope boundaries, not claims of experimental validation. Contract parity is tested across the canonical and generic public entry points; optional estimates remain absent or null when their required fit windows and model inputs are not supplied.


## Multi-channel and multi-probe sweeps
The public helper `analyze_langmuir_probe_array` accepts either a shared voltage vector with a two-dimensional (channels, samples) current array, matching two-dimensional voltage/current arrays, or a mapping of channel names to calibrated voltage/current pairs. Each channel is analyzed independently with the same canonical calculations and explicit options. Results preserve channel labels and report the channel count. This adds batch execution, not spatial reconstruction, probe geometry inference, cross-channel correlations, or an assertion that all channels share the same physical regime.

## Plasma-potential proxy contract
`plasma_potential_method` is `maximum_numerical_dIdV_proxy`; `plasma_potential_estimate_V` uses the maximum numerical first derivative. The maximum-second-derivative location remains a separate candidate field and is not silently treated as the primary estimate. Both are numerical indicators, not a physical certification. Derivatives are calculated on the supplied calibrated samples; no smoothing is silently applied.

## Model and uncertainty boundaries
The current fit remains the declared single-Maxwellian exponential-plus-offset model. The software does not fit non-Maxwellian or multi-temperature distributions. Statistical covariance is conditional on that model and selected window; uncertainty remains partial when calibration covariance, model discrepancy, correlations, or fit-window sensitivity are not explicitly quantified. These are not claimed as solved by multi-channel batching.


## Advanced calibrated-data extensions

The canonical single-sweep pipeline remains the default and retains its original output contract. Additional explicit functions are available from `tokamak_analysis.langmuir_probe`:

- `fit_langmuir_electron_models(..., model="single_maxwellian")` fits the existing single-exponential-plus-offset model independently and returns fit quality.
- `fit_langmuir_electron_models(..., model="two_temperature")` fits a two-exponential mixture and reports both component temperatures. This is an empirical model, not proof of a bi-Maxwellian electron distribution; ill-conditioning and component non-identifiability must be reviewed.
- `analyze_langmuir_probe_series` accepts calibrated sweeps shaped (time, channel, sample) and processes each sweep independently. It does not infer temporal trends or cross-channel physics.
- `estimate_multispecies_ion_density` supports an explicitly supplied species mixture (mass, fraction, sheath coefficient). Fractions must sum to one. It assumes a common electron temperature and effective collection area, and reports a model estimate without physical-regime certification.
- `langmuir_fit_window_sensitivity` re-fits nearby windows and reports the spread in fitted temperature. This is a sensitivity diagnostic, not a statistical confidence interval.

All of these functions require calibrated input and explicit physical assumptions. No calibration is inferred, no experimental validation is claimed, and no automated output is promoted to a confirmed physical measurement.

## Plasma-potential derivative handling

The canonical output keeps the maximum-`dI/dV` proxy and maximum-`d²I/dV²` candidate separate. The canonical derivative is not silently smoothed because smoothing changes the effective method and can shift extrema. Users who need a noise-robust derivative estimate should compare a declared smoothing/window method and retain its parameters and sensitivity; neither derivative candidate alone certifies plasma potential.

## Uncertainty scope

The canonical covariance estimate is conditional on the selected fit model and voltage window. The advanced window-sensitivity helper quantifies the observed temperature spread across the caller-specified alternative windows, but does not combine that spread into a formal standard uncertainty. Model discrepancy, correlated errors, calibration covariance and other systematic effects must not be silently treated as zero. Multispecies density currently returns no uncertainty estimate unless a separate uncertainty model is supplied; its output explicitly marks propagation as not performed.

## Regression tests

The advanced contracts are covered by `tests/test_langmuir_advanced.py`. These tests check software behavior only; they are not empirical validation against a tokamak or probe experiment.


Additional explicit helpers extend the declared calculation scope without changing the canonical single-sweep default:

- `fit_langmuir_stretched_exponential` provides an empirical curve-shape alternative. Its shape exponent is not a kinetic kappa parameter and does not identify a unique non-Maxwellian distribution.
- `estimate_smoothed_plasma_potential_candidates` reports raw-gradient and Savitzky–Golay-smoothed derivative candidates separately, with the smoothing window and polynomial order exposed. Neither candidate is automatically certified as the plasma potential.
- `estimate_electron_density_from_saturation` estimates electron density from a positive electron-saturation current using the stated Maxwellian collection relation and user-supplied effective area.
- `estimate_ion_particle_flux` computes a Bohm-like ion particle flux from supplied density, electron temperature, ion mass, and explicit sheath factor.

These helpers do not infer area, species composition, sheath coefficients, or uncertainty inputs. Electron-density and particle-flux estimates are model-dependent and report no uncertainty propagation unless a separate uncertainty model is supplied. Their presence means these quantities can be calculated under explicit assumptions; it does not mean the physical regime is confirmed.


## Explicit resolution of the Analysis 2 contract findings

- **Electron density is an advanced, separate model helper, not a field silently inferred by the canonical single-sweep pipeline.** The public module exports `estimate_electron_density_from_saturation`; it requires a positive supplied electron-saturation current, fitted/supplied electron temperature, and effective collection area. It uses the explicitly documented Maxwellian collection relation and always marks `physical_regime_confirmed: false`. This avoids implying that the canonical pipeline derives electron density from the ion-tail fallback or from an unverified branch.
- **Plasma potential remains a candidate/proxy.** `plasma_potential_estimate_V` is the maximum-numerical-`dI/dV` proxy; the maximum-`d²I/dV²` candidate remains separate. Neither is a certified physical measurement.
- **Ion saturation remains explicitly qualified.** Without `ion_fit_range`, the low-voltage-tail median is descriptive only. A selected-window estimate is still not physical-regime certification, and descriptive fallback values are not accepted for density inversion.
- **Uncertainty is partial by contract.** The canonical electron-temperature and ion-density uncertainty outputs include fit covariance and explicitly supplied components only. Missing calibration covariance, correlated errors, model discrepancy, and other unprovided systematic effects are not assumed to be zero or automatically estimated. The electron-density helper and ion-particle-flux helper explicitly report that uncertainty propagation was not performed.
- **Array and series APIs are batch processors, not spatial/temporal inference.** Each channel and each time/channel sweep is passed through the canonical single-sweep calculation independently. The returned scope flags explicitly state that cross-channel inference and temporal-trend analysis were not performed.

These are software-contract and interpretation safeguards for already-calibrated inputs. They do not add calibration, empirical validation, or automatic physical-regime selection.
