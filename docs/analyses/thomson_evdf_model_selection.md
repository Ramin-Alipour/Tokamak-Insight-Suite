# Projected Thomson EVDF: parametric model-selection uncertainty

The non-parametric inverse reconstructs a normalized one-dimensional marginal
along the scattering vector. Its optional parametric comparison fits three
explicit projected families to the same response-forward-modelled spectrum:

- Maxwellian (Gaussian projected velocity density),
- finite-variance Student-t/Kappa projection,
- super-Gaussian projection.

For each converged candidate, the solver evaluates the same residual likelihood
used by its fit and computes AIC. The relative Akaike support is

\[
\Delta_i = \mathrm{AIC}_i - \min_j \mathrm{AIC}_j,\qquad
w_i = \frac{\exp(-\Delta_i/2)}{\sum_j\exp(-\Delta_j/2)}.
\]

For a scalar moment \(g_i\) reported by candidate \(i\), the model-averaged
estimate and variance are

\[
\bar g = \sum_i w_i g_i,\qquad
\mathrm{Var}(g) =
\underbrace{\sum_i w_i(g_i-\bar g)^2}_{\text{between-model}}
+
\underbrace{\sum_i w_i\,\mathrm{Var}_i(g)}_{\text{within-model}}.
\]

The implementation reports the between-model and within-model terms separately.
The within-model term is included only when every contributing candidate has a
finite local parameter covariance; otherwise the total model-averaged standard
uncertainty is withheld rather than silently treating a missing covariance as
zero.

These weights are **relative support within the three declared candidates**, not
Bayesian posterior model probabilities. They do not account for unrepresented
EVDF families, uncertainty in the candidate set, or regularization selection.
Regularization sensitivity remains a separate diagnostic. All quantities remain
conditional on the supplied geometry, measured response, background, velocity
grid and measurement/discrepancy/systematic covariance.

The output is returned by `reconstruct_thomson_evdf` as
`parametric_model_selection_uncertainty`, forwarded by Analysis 81 through
`projected_evdf_inference`, and rendered in the Advanced Thomson Physics
Dashboard. It does not promote a Gaussian width to electron temperature or
claim a three-dimensional EVDF from a single view.
