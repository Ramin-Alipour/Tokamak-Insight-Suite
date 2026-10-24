# Analysis 2 — Langmuir probe

## Scope
This workflow analyzes a calibrated, quasi-static, single-probe I–V sweep. It reports cleaned and voltage-sorted samples, descriptive current tails, the floating-potential crossing(s), a maximum-slope plasma-potential proxy, and optional fit-window-based electron temperature, ion-branch metrics, and sheath-model ion-density estimate.

## Inputs and preprocessing
- `voltage` (V) and `current` (A): paired one-dimensional arrays.
- At least eight valid samples and eight distinct voltage values are required. Non-finite pairs are discarded; duplicate voltages are aggregated by averaging current; pairs are sorted together by voltage.
- `current_convention`: `electron_positive` (default) or `ion_positive`. The latter is sign-flipped before analysis.
- Optional fit intervals must have two finite, strictly increasing voltage bounds. Intervals are selected by the analyst, must overlap the measured sweep, and must contain enough samples for the requested fit.

## Physics and outputs
For an electron-retarding Maxwellian branch after subtracting an estimated ion-current baseline,
\[
I_e(V)=I_{e,\mathrm{sat}}\exp\left(\frac{V-V_p}{T_e[\mathrm{eV}]}\right),\qquad
T_e[\mathrm{eV}]=\left(\frac{d\ln I_e}{dV}\right)^{-1}.
\]

The maximum of numerical `dI/dV` is reported only as a plasma-potential proxy. The optional ion-density model is
\[
n_i \approx \frac{|I_{i,\mathrm{sat}}|}{\alpha e A c_s},\qquad
c_s=\sqrt{\frac{e(\gamma_i T_i+T_e)}{m_i}},
\]
where `A` is the effective collection area, `α` is the selected sheath factor, temperatures are in eV, and `m_i` is ion mass. This is a model-dependent estimate, not a universal probe inversion.

The floating-potential result includes all detected zero crossings. A zero-current plateau is represented by its voltage midpoint. `floating_potential_V` is populated only for exactly one unique crossing; no crossing or multiple crossings produces `null` with an explicit status.

## Validation and limitations
Focused regression tests cover crossing interpolation, zero plateaus, ambiguous crossings, fit-range validation, current polarity, duplicate voltages, and non-finite sample handling. Fit-window choice, probe calibration, RF pickup, magnetic-field effects, collisions, secondary emission, sheath expansion, surface condition, and sweep direction remain experiment-specific. The reported temperature regression error is statistical only; calibration and model uncertainty are not propagated. Do not interpret a numerical proxy or model estimate as independently validated experimental truth.


## Auditable quality and fit diagnostics

The result includes an `input_quality` object with original pair count, finite pairs retained, non-finite pairs discarded, duplicate-voltage pairs averaged, processed unique-voltage count, voltage span, and normalized current convention. When an electron-retarding fit is enabled, the output also includes fit residual RMS (A), normalized RMSE, R-squared, fit-point count, and a status explicitly instructing the analyst to review the residuals and R-squared. These diagnostics do not certify that the selected voltage interval is physically valid and do not replace calibration or systematic uncertainty analysis.

Regression tests: `tests/test_langmuir_qc_contract.py`.
