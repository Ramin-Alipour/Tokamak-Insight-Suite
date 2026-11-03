# Thomson inference closure: joint nuisance, profile and recoil contracts

This implementation adds lower-level, measured-spectrum inference primitives to
`tokamak_analysis.thomson_inference_closure` and public wrappers in
`tokamak_analysis.thomson_workflow`.

## Joint signal/background fit

`fit_thomson_spectrum_with_structured_background` fits the supplied physical
forward model and a detector-space background basis in one optimizer. The
returned covariance includes physical/background cross terms. Supply either
pointwise measurement uncertainty or a full positive-definite spectral
covariance. If neither is supplied, the fit is unweighted and any covariance is
residual-scaled; it is not measurement-noise uncertainty.

The routine does not decide which background basis is physically appropriate.
The caller supplies a measured/calibrated background design matrix. This avoids
silently treating a fitted continuum as plasma emission. The current
channel-by-channel Kappa and relativistic Maxwell-Juttner inference paths in
`thomson_inference.py` are not silently rewritten by this helper; callers must
pass the selected Kappa/MJ forward model explicitly. Full native Kappa/MJ
background integration across every UI/API path remains an integration task.

## Joint ne(r), Te(r), Ti(r) inversion

`reconstruct_joint_ne_te_ti_profiles` fits all supplied measured channel
spectra together and returns a joint profile covariance when the *data-only*
Jacobian has full column rank. It separately reports the rank of the
regularized Jacobian. A smooth solution whose data-only Jacobian is rank
deficient is labelled `non_identifiable_regularized_solution`; regularization
does not create measurement information.

The caller supplies the channel-resolved physical forward model. In
particular, Ti is not identifiable unless the supplied spectral model contains
an ion-sensitive feature. This primitive does not infer equilibrium mapping,
flux surfaces, PSF, channel cross-talk or ray geometry.

## Recoil and cross-section provenance

`thomson_compton_recoil_contract` records whether the relativistic model uses
elastic polarized Thomson scattering or single-scattering Compton kinematics
with polarized Klein-Nishina weighting. The recoil screen is an explicit
software threshold, not a universal plasma validity boundary. Multiple
scattering and unrestricted relativistic EVDF inversion are outside this
contract.

## Existing related implementations

- `cts_geometry_uncertainty.py` already performs full-forward Monte Carlo
  geometry refits. Each sampler draw must regenerate equilibrium, ray paths,
  beam overlap and scattering-volume quadrature; do not replace this with
  adding a final covariance matrix.
- `thomson_gap_closure.py` contains a relativistic 1-D projected EVDF inverse.
  It is not a general 3-D EVDF reconstruction.
- `thomson_population_mixture.py` contains a two-population MJ parametric
  approximation. It is not the generalized arbitrary-VDF algorithm of
  Skolar et al. (2025).
- The canonical Thomson UI and registry already expose the Compton-recoil
  selection for the relativistic MJ path. The recoil helper documents its
  cross-section semantics for programmatic callers.

## References

- Carole et al., *Journal of Instrumentation* **20** (2025) P07014,
  DOI: [10.1088/1748-0221/20/07/P07014](https://doi.org/10.1088/1748-0221/20/07/P07014).
- Skolar, Longley & Goodwin, *Physics of Plasmas* **32** (2025) 043907,
  DOI: [10.1063/5.0261966](https://doi.org/10.1063/5.0261966).
- Sakai et al., *Journal of Instrumentation* **21** (2026) C01002,
  DOI: [10.1088/1748-0221/21/01/C01002](https://doi.org/10.1088/1748-0221/21/01/C01002).
