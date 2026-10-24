# Thomson Scattering (Analysis 1 of 81)

## Completion status

**Implementation status: complete within the explicitly defined computational scope.** The five declared workflow modes are implemented, documented, exposed through the applicable dashboard/API paths, and covered by automated numerical and input-validation tests. This status does not claim independent device-specific experimental validation or support for collective/relativistic inversion, arbitrary non-Maxwellian reconstruction, or non-Gaussian instrument-response inversion. The supported calibrated-profile, raw-spectrum temperature-fit, calibrated single-band power-density, and combined calibrated spectrum-to-temperature/density paths have callable implementations, documented contracts, dashboard entry points, and focused numerical/validation tests. The user-defined spectral-model extension is a Python API, not an arbitrary-code dashboard editor.

The scope is intentionally bounded: no density inference from uncalibrated amplitude; no collective, relativistic, non-Maxwellian, or non-Gaussian instrument-response inversion; no automatic stray-light correction, full spectral-response correction, or complete calibration/model-discrepancy uncertainty budget. Users must verify instrument calibration, spectral-band coverage, optical geometry, time-gate compatibility, and residuals against the experiment.

## Supported implementation paths

1. **Calibrated profile analysis** (tokamak_analysis.thomson_workflow.analyze_thomson_profile): computes profile extrema, line averages, optional approximate volume averages, radial gradients, normalized gradients, and local scale lengths from already-calibrated electron-density and electron-temperature profiles.
2. **Raw spectral-shape temperature fit** (tokamak_analysis.thomson_spectrum.fit_thomson_spectrum): fits a Gaussian Doppler profile plus a linear local background, deconvolves a known Gaussian instrument width, and estimates electron temperature for a non-collective, non-relativistic Maxwellian model.

The single-band helper density_from_thomson_scattered_power estimates density from calibrated collected power and specified geometry. The spectral fit can optionally integrate its Gaussian component to collected power only when the signal is explicitly declared to be calibrated spectral power density in W/nm. The spectral-fit amplitude itself is **not** a density measurement. The power-based density helper also supports first-order propagation of optional independent 1-sigma uncertainties for measured power, incident power, interaction length, scattering angle, collection solid angle, and optical efficiency. The dashboard can pass the integrated spectral power and its fit uncertainty directly into this density calculation. The dashboard exposes this as a separate calibrated scattered-power density workflow, including optional uncertainty inputs.

## Raw spectrum input contract

| Input | Unit | Requirement |
|---|---|---|
| wavelength_nm | nm | One-dimensional, finite, strictly increasing calibrated wavelength axis; at least 7 samples |
| signal | detector units by default; W/nm when calibrated-power mode is enabled | One-dimensional finite measured signal, same length as wavelength |
| laser_wavelength_nm | nm | Positive finite laser wavelength |
| scattering_angle_rad | rad | In (0, pi] |
| instrument_sigma_nm | nm | Positive known Gaussian instrumental standard deviation |
| signal_uncertainty | same units as signal | Optional positive finite 1-sigma uncertainty for every signal sample |

The fitted model is a Gaussian Doppler-broadened profile plus a linear background. Instrumental broadening is removed in quadrature:

~~~text
sigma_physical = sqrt(sigma_fit^2 - sigma_instrument^2)
~~~

For the non-relativistic, non-collective Maxwellian approximation:

~~~text
T_e[eV] = (sigma_physical / lambda_0)^2
          * (m_e c^2 / 4) / sin(scattering_angle / 2)^2
~~~

The module rejects a fit when the fitted width does not exceed the supplied instrumental width. That case is unresolved by the stated model and must not be reported as a temperature. It also rejects wavelength windows that do not cover at least ±3 fitted standard deviations around the line center.

## Raw-spectrum outputs

- Electron temperature and local covariance-based fit uncertainty (eV)
- Fitted center wavelength and measured, instrumental, and deconvolved widths (nm)
- Fitted spectrum and residual arrays
- Residual sum of squares; chi-square and reduced chi-square only when per-sample uncertainties are supplied
- R-squared and fit covariance
- Optional integrated collected power and its fit uncertainty when the signal is calibrated in W/nm
- Explicit model assumptions and limitations

The reported temperature uncertainty does not include uncertainty in laser wavelength, scattering angle, or instrument width. A covariance-based fit uncertainty is not a complete measurement uncertainty budget.

## Profile input contract

The profile workflow expects calibrated radius_m (m), density_m3 (m^-3), and temperature_ev (eV) arrays with equal lengths and at least two finite samples. Radius values must be unique; density, temperature, and supplied pointwise uncertainties must be non-negative. The arrays are sorted by radius together. Optional volume averages use circular, large-aspect-ratio weighting and require a non-negative minor-radius coordinate; they are not an equilibrium reconstruction.

Pointwise uncertainties are validated and propagated through finite-difference gradients, normalized gradients, local scale lengths, and line/approximate volume averages assuming independent samples. Peak-location uncertainty is not estimated.

## Example

~~~python
import numpy as np
from tokamak_analysis.thomson_spectrum import fit_thomson_spectrum

wavelength_nm = ...  # calibrated wavelength axis
signal = ...         # measured spectrum
fit = fit_thomson_spectrum(
    wavelength_nm,
    signal,
    laser_wavelength_nm=532.0,
    scattering_angle_rad=np.pi / 2,
    instrument_sigma_nm=0.05,
)
print(fit["electron_temperature_eV"], fit["electron_temperature_uncertainty_eV"])
~~~

## End-to-end calibrated spectrum-to-density workflow

The callable `tokamak_analysis.thomson_workflow.analyze_thomson_spectrum_and_density` composes the spectral fit and calibrated scattered-power density estimate into one reproducible Python entry point.

Required inputs include a calibrated wavelength axis, spectral power density in W/nm, laser wavelength, scattering angle, known Gaussian instrument sigma, incident laser power, effective interaction length, collection solid angle, and total optical efficiency. Optional independent 1-sigma uncertainties can be supplied for the spectral samples and optical geometry.

The function returns electron temperature from the Gaussian spectral width and electron density from the integrated fitted Gaussian power using the classical unpolarized differential Thomson cross-section. Temperature uncertainty is fit-only. Density uncertainty is returned when at least one supported input uncertainty is supplied and uses first-order independent-error propagation.

This workflow is only appropriate when the measured spectrum is absolutely calibrated in W/nm over a band compatible with the integrated-power model, the plasma is optically thin, the signal is dominated by non-collective non-relativistic Maxwellian Thomson scattering, and the instrument response is adequately Gaussian. It does not automatically correct stray light, wavelength-dependent throughput, spectral-band truncation, polarization-specific collection geometry, calibration covariance, or model discrepancy. Results require an independent uncertainty budget and experiment-specific review.


## User-defined spectral model extension

Users can register an experiment-specific or research-specific spectral model in Python without changing the built-in Gaussian workflow. The extension is an explicit in-process registry: users provide a callable of the form `model(wavelength_nm, *parameters)`, parameter names, initial values, optional lower/upper bounds, and a description. The generic fitter returns fitted parameters, their covariance-based uncertainties, fitted spectrum, residuals, and (when signal uncertainties are supplied) chi-square diagnostics.

```python
import numpy as np
from tokamak_analysis.thomson_models import (
    register_spectral_model,
    fit_custom_spectral_model,
)

def my_spectrum(wavelength_nm, amplitude, center_nm, sigma_nm, background):
    return amplitude * np.exp(
        -0.5 * ((wavelength_nm - center_nm) / sigma_nm) ** 2
    ) + background

register_spectral_model(
    "my_experiment_model",
    my_spectrum,
    parameter_names=("amplitude", "center_nm", "sigma_nm", "background"),
    initial_values=(100.0, 532.0, 0.2, 1.0),
    bounds=((0.0, 500.0, 1e-6, -np.inf),
            (np.inf, 560.0, np.inf, np.inf)),
    description="Example model; replace with a validated physical model",
)

fit = fit_custom_spectral_model(
    wavelength_nm,
    measured_signal,
    model_name="my_experiment_model",
    signal_uncertainty=signal_sigma,  # optional
)
print(fit["parameters"])
```

The built-in registry is process-local, not isolated per web-app user: registered models remain available to code running in that Python process until unregistered or the process exits. In a multi-user service, do not use this global registry for private per-user model state; use isolated processes or a dedicated per-session registry layer. Register the model during application setup or from the user's own Python module. The software does not automatically load arbitrary files or execute downloaded plugins. The generic fitter does not automatically infer electron temperature, density, or other physical quantities from custom parameters; the model author must define and validate that interpretation, its instrument-response treatment, and its uncertainty budget. Registration is not evidence that a model is physically valid. The built-in Thomson workflow remains unchanged and continues to be the default supported model.

## Scientific limitations

- Assumes a non-collective, non-relativistic Maxwellian electron distribution.
- Assumes a Gaussian instrument line-spread function with known standard deviation.
- Uses a locally linear background; it does not perform full stray-light or spectral-response calibration.
- Does not fit collective spectra, relativistic/non-Maxwellian spectra, or non-Gaussian instrument responses.
- Does not infer density from uncalibrated spectral amplitude. Absolute density needs calibrated spectral power density or collected power, laser power, interaction length, differential cross-section, collection solid angle, optical efficiency, and appropriate spectral-band treatment.
- Synthetic tests establish numerical recovery for the stated model only; they do not constitute independent experimental validation.
- A provisional MAST shot 28830 cross-diagnostic check compared archived Thomson-derived edge density and independent interferometer line density with rounded published Table II values. Relative differences were −3.77% and −1.74%, respectively, within a predeclared 10% tolerance. The paper's exact averaging interval is unspecified, so this is not a raw-spectrum or full spectrum-to-density validation.

## Completion gate

The profile, raw-spectrum temperature-fit, calibrated scattered-power density, and combined calibrated spectrum-to-temperature/density paths have callable implementations, documented contracts, dashboard access, and focused numerical/validation tests. The software implementation is complete within that bounded scope, while the overall registry conservatively remains marked incomplete pending independent experimental validation and any broader physics extensions. The user-defined spectral-model extension is available through the documented Python API and is not presented as a dashboard workflow. This status does **not** mean complete Thomson-scattering physics or independent experimental validation. The following remain explicitly out of scope and must not be implied by the UI, registry, or reports: collective-scattering inversion, relativistic/non-Maxwellian spectral models, non-Gaussian instrument-response deconvolution, automatic stray-light and wavelength-dependent throughput correction, spectral-band completeness correction, and a full calibration/model-discrepancy uncertainty budget. Absolute density is supported only through the stated calibrated single-band power model and only when its calibration and geometry assumptions are satisfied.
