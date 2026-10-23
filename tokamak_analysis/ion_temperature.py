"""Ion-temperature profile analysis from already inferred measurements."""
import numpy as np
def analyze_ion_temperature(temperature_eV, *, coordinate=None, uncertainty_eV=None):
    ti=np.asarray(temperature_eV,dtype=float)
    if ti.ndim!=1 or len(ti)<2 or not np.all(np.isfinite(ti)) or np.any(ti<0): raise ValueError("Ion temperature must be finite, non-negative 1-D data")
    out={"temperature_i_eV":ti,"mean_eV":float(np.mean(ti)),"median_eV":float(np.median(ti)),"peak_eV":float(np.max(ti)),"peak_index":int(np.argmax(ti)),"standard_deviation_eV":float(np.std(ti,ddof=1))}
    if coordinate is not None:
        x=np.asarray(coordinate,dtype=float)
        if x.shape!=ti.shape or not np.all(np.isfinite(x)) or np.any(np.diff(x)<=0): raise ValueError("coordinate must match and increase strictly")
        out["coordinate"]=x; out["gradient_eV_per_coordinate"]=np.gradient(ti,x,edge_order=2 if len(ti)>2 else 1)
    if uncertainty_eV is not None:
        u=np.asarray(uncertainty_eV,dtype=float)
        if u.shape!=ti.shape or not np.all(np.isfinite(u)) or np.any(u<0): raise ValueError("Uncertainties must be finite, non-negative, and match data")
        out["uncertainty_eV"]=u
    out["caveat"]="Input must be a measured or independently inferred Ti profile; this does not fit spectral line widths."
    return out

def ion_temperature_from_pressure_density(ion_pressure_Pa, ion_density_m3):
    """Ideal-gas ion temperature in eV: Ti=pi/(ni*e)."""
    pi,ni=np.broadcast_arrays(np.asarray(ion_pressure_Pa,dtype=float),np.asarray(ion_density_m3,dtype=float))
    if not np.all(np.isfinite(pi)) or not np.all(np.isfinite(ni)) or np.any(pi<0) or np.any(ni<=0):
        raise ValueError("Ion pressure must be non-negative and ion density positive; inputs finite")
    ti=pi/(ni*1.602176634e-19)
    return {"ion_temperature_eV":ti,"ion_pressure_Pa":pi,"ion_density_m3":ni,
            "equation":"Ti[eV]=pi[Pa]/(ni[m^-3]*e[C])",
            "assumption":"Single-temperature ideal ion gas; ion species and density definition must be consistent."}


def ion_temperature_from_doppler_spectrum(
    wavelength_nm,
    intensity,
    *,
    rest_wavelength_nm,
    ion_mass_amu,
    instrumental_fwhm_nm,
    instrumental_fwhm_uncertainty_nm=None,
    intensity_uncertainty=None,
):
    """Fit an isolated calibrated Gaussian emission line and infer Ti.

    The fitted Gaussian FWHM is deconvolved from a known Gaussian instrument
    FWHM in quadrature. Optional independent 1-sigma uncertainties for the
    instrument width and intensity samples are propagated locally. This is an
    estimate, not an automatic validation that all line broadening is thermal.
    """
    from scipy.optimize import curve_fit

    from tis.doppler_temperature import doppler_temperature_eV

    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(intensity, dtype=float)
    rest = float(rest_wavelength_nm)
    mass = float(ion_mass_amu)
    inst = float(instrumental_fwhm_nm)
    if (x.ndim != 1 or y.shape != x.shape or x.size < 7
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))
            or np.any(np.diff(x) <= 0)):
        raise ValueError("wavelength_nm and intensity must be finite, matching 1-D arrays with at least 7 increasing wavelengths")
    if not np.isfinite(rest) or rest <= 0 or not np.isfinite(mass) or mass <= 0:
        raise ValueError("rest wavelength and ion mass must be finite and positive")
    if not np.isfinite(inst) or inst < 0:
        raise ValueError("instrumental_fwhm_nm must be finite and non-negative")
    if instrumental_fwhm_uncertainty_nm is not None:
        uinst = float(instrumental_fwhm_uncertainty_nm)
        if not np.isfinite(uinst) or uinst < 0:
            raise ValueError("instrumental width uncertainty must be finite and non-negative")
    else:
        uinst = None
    if intensity_uncertainty is not None:
        uy = np.asarray(intensity_uncertainty, dtype=float)
        if uy.shape != y.shape or not np.all(np.isfinite(uy)) or np.any(uy <= 0):
            raise ValueError("intensity_uncertainty must be finite, positive, and match intensity")
    else:
        uy = None

    baseline0 = float(np.percentile(y, 10))
    amplitude0 = float(np.max(y) - baseline0)
    if amplitude0 <= 0:
        raise ValueError("spectrum has no positive emission-line contrast")
    spacing = float(np.median(np.diff(x)))
    sigma0 = max(spacing, (x[-1] - x[0]) / 12.0)
    def gaussian(z, amplitude, center, sigma, baseline, slope):
        return baseline + slope * (z - rest) + amplitude * np.exp(-0.5 * ((z-center)/sigma)**2)
    p0 = [amplitude0, float(x[np.argmax(y)]), sigma0, baseline0, 0.0]
    lower = [0.0, float(x[0]), spacing * 1e-3, -np.inf, -np.inf]
    upper = [np.inf, float(x[-1]), float(x[-1]-x[0]), np.inf, np.inf]
    popt, pcov = curve_fit(
        gaussian, x, y, p0=p0, bounds=(lower, upper),
        sigma=uy, absolute_sigma=uy is not None, maxfev=50000,
    )
    fitted = gaussian(x, *popt)
    residual = y - fitted
    sigma_nm = float(popt[2])
    fwhm_nm = 2.354820045 * sigma_nm
    fwhm_unc = None
    if pcov.shape == (5, 5) and np.isfinite(pcov[2, 2]) and pcov[2, 2] >= 0:
        fwhm_unc = 2.354820045 * float(np.sqrt(pcov[2, 2]))
    kwargs = {}
    if fwhm_unc is not None and uinst is not None:
        kwargs = {
            "observed_fwhm_uncertainty_nm": fwhm_unc,
            "instrumental_fwhm_uncertainty_nm": uinst,
        }
    result = doppler_temperature_eV(
        rest, fwhm_nm, mass, instrumental_fwhm_nm=inst, **kwargs
    )
    variance_total = float(np.sum((y - np.mean(y))**2))
    r_squared = (float(1.0 - np.sum(residual**2)/variance_total)
                 if variance_total > 0 else float("nan"))
    return {
        "ion_temperature_eV": result.temperature_eV,
        "ion_temperature_uncertainty_eV": result.uncertainty_eV,
        "observed_fwhm_nm": fwhm_nm,
        "observed_fwhm_uncertainty_nm": fwhm_unc,
        "thermal_doppler_fwhm_nm": result.doppler_fwhm_nm,
        "instrumental_fwhm_nm": inst,
        "line_center_nm": float(popt[1]),
        "line_shift_nm": float(popt[1] - rest),
        "ion_mass_amu": mass,
        "rest_wavelength_nm": rest,
        "fit_r_squared": r_squared,
        "fit_residuals": residual,
        "fitted_intensity": fitted,
        "fit_parameters": {"amplitude": float(popt[0]), "center_nm": float(popt[1]),
                            "sigma_nm": sigma_nm, "baseline": float(popt[3]),
                            "baseline_slope_per_nm": float(popt[4])},
        "fit_covariance": pcov,
        "status": result.status,
        "assumptions": list(result.assumptions) + [
            "Single isolated Gaussian line with a locally linear background",
            "Uncertainty is local fit/covariance propagation and excludes wavelength, mass, calibration, and model-discrepancy uncertainty",
        ],
        "caveat": "Review blends, opacity, unresolved fine structure, Stark/Zeeman broadening, rotation and non-thermal ion distributions before interpreting the result as physical Ti.",
    }
