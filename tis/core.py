"""Numerical routines for Tokamak Insight Suite. Inputs and assumptions are explicit."""
from __future__ import annotations
import hashlib
import io
import json
import math
from typing import Any
import numpy as np
import pandas as pd
from scipy import signal, optimize, integrate, linalg
from .signal_processing import (butterworth_filter, fft_spectrum, welch_psd, autocorrelation_analysis, time_frequency_spectrogram, cross_analysis, cross_correlation_lag)
from .spectral_analysis import fit_spectral_line
from scipy.special import voigt_profile, erfcx, erfc

E_CHARGE = 1.602176634e-19
K_B = 1.380649e-23
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
M_E = 9.1093837139e-31
C_LIGHT = 299792458.0
AMU = 1.66053906660e-27


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_table(data: bytes, filename: str, sheet_name: str | int = 0, *,
               delimiter: str | None = None, decimal: str = ".", skiprows: int = 0,
               header: int | None | str = "infer", comment: str | None = None,
               encoding: str | None = None,
               max_bytes: int = 100_000_000, max_rows: int = 10_000_000,
               max_uncompressed_bytes: int = 500_000_000) -> pd.DataFrame:
    """Load tabular data with explicit parsing options and bounded input size.

    '#' remains an ordinary character by default, preserving headers such as Ch#1.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError("data must be bytes.")
    if len(data) > max_bytes:
        raise ValueError(f"Input exceeds the configured {max_bytes:,}-byte import limit.")
    if decimal not in {".", ","}:
        raise ValueError("decimal must be '.' or ','.")
    if isinstance(skiprows, bool) or not isinstance(skiprows, int) or skiprows < 0:
        raise ValueError("skiprows must be a nonnegative integer.")
    name = filename.lower()
    if name.endswith((".csv", ".txt")):
        kwargs = {"decimal": decimal, "skiprows": skiprows, "header": header}
        if comment is not None:
            kwargs["comment"] = comment
        # Instrument exports are frequently Windows-1252/Latin-1 (e.g. 'Température',
        # 'µs'). Try strict UTF-8 (BOM-tolerant) first, then fall back, and record
        # which encoding was actually used so the choice is part of provenance.
        candidates = [encoding] if encoding else ["utf-8-sig", "cp1252", "latin-1"]
        frame = None; used_encoding = None; last_exc = None
        for enc in candidates:
            try:
                frame = pd.read_csv(io.BytesIO(data), sep=delimiter or None, encoding=enc,
                                    engine="python" if delimiter is None else "c", **kwargs)
                used_encoding = enc
                break
            except UnicodeDecodeError as exc:
                last_exc = exc
        if frame is None:
            raise ValueError(f"Could not decode the text file with encoding(s) {candidates}: {last_exc}")
        frame.attrs["encoding"] = used_encoding
    elif name.endswith((".xlsx", ".xls")):
        if name.endswith(".xlsx"):
            import zipfile
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    members=archive.infolist()
                    expanded=sum(member.file_size for member in members)
                    if len(members)>2000 or expanded>max_uncompressed_bytes:
                        raise ValueError("Excel archive exceeds the safe member-count or uncompressed-size limit.")
                    if any(member.compress_size>0 and member.file_size/member.compress_size>1000 for member in members):
                        raise ValueError("Excel archive contains an extreme compression ratio and was rejected.")
            except zipfile.BadZipFile as exc:
                raise ValueError("The XLSX file is not a valid ZIP-based Excel workbook.") from exc
        frame = pd.read_excel(io.BytesIO(data), sheet_name=sheet_name, skiprows=skiprows, header=header)
    elif name.endswith((".parquet", ".pq")):
        import pyarrow.parquet as pq
        try:
            parquet_file=pq.ParquetFile(io.BytesIO(data))
            if parquet_file.metadata.num_rows>max_rows or parquet_file.metadata.num_row_groups>10000:
                raise ValueError("Parquet file exceeds the configured row or row-group limit.")
        except (ValueError, OSError) as exc:
            if isinstance(exc, ValueError): raise
            raise ValueError("Could not safely inspect Parquet metadata.") from exc
        frame = pd.read_parquet(io.BytesIO(data))
    else:
        raise ValueError("Supported formats: CSV, TXT, XLSX/XLS and Parquet.")
    if isinstance(frame, dict):
        raise ValueError("Select one worksheet at a time.")
    if frame.empty:
        raise ValueError("The imported table is empty.")
    if len(frame)>max_rows:
        raise ValueError(f"Table exceeds the configured {max_rows:,}-row import limit.")
    frame.columns = [str(col).strip() or f"column_{i}" for i, col in enumerate(frame.columns)]
    # Recover mostly numeric instrument columns containing a unit row or a few
    # malformed cells. Those cells become NaN and remain visible to QC.
    for col in frame.columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            candidate = pd.to_numeric(frame[col], errors="coerce")
            nonempty = int(frame[col].notna().sum())
            numeric_count = int(candidate.notna().sum())
            if nonempty >= 2 and numeric_count >= 2 and numeric_count / nonempty >= 0.67:
                frame[col] = candidate
    return frame


def signal_quality(values: Any) -> dict[str, Any]:
    x = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(x)
    v = x[finite]
    if not len(v):
        return {'n_total': int(len(x)), 'n_finite': 0, 'n_missing_or_nonfinite': int(len(x)), 'error': 'No finite numeric samples.'}
    return {
        'n_total': int(len(x)), 'n_finite': int(len(v)), 'n_missing_or_nonfinite': int(len(x)-len(v)),
        'mean': float(np.mean(v)), 'std_sample': float(np.std(v, ddof=1)) if len(v)>1 else 0.0,
        'median': float(np.median(v)), 'min': float(np.min(v)), 'max': float(np.max(v)),
        'rms_including_dc': float(np.sqrt(np.mean(v*v))), 'peak_to_peak': float(np.ptp(v)),
        'relative_rms_fluctuation': float(np.std(v)/abs(np.mean(v))) if abs(np.mean(v))>np.finfo(float).eps else None,
        'duplicate_fraction': float((len(v)-len(np.unique(v)))/len(v)),
    }


def linear_calibration(values: Any, slope: float, intercept: float = 0.0) -> np.ndarray:
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError('Calibration slope and intercept must be finite.')
    return np.asarray(values, dtype=float) * slope + intercept


def polynomial_calibration(values: Any, coefficients: list[float]) -> np.ndarray:
    if not coefficients or not np.all(np.isfinite(coefficients)):
        raise ValueError('Provide finite polynomial coefficients, highest order first.')
    return np.polyval(np.asarray(coefficients, dtype=float), np.asarray(values, dtype=float))



def integrate_time_series(
    time_s: Any,
    values: Any,
    scale: float = 1.0,
    initial_value: float = 0.0,
    baseline_start_s: float | None = None,
    baseline_end_s: float | None = None,
    drift_start_s: float | None = None,
    drift_end_s: float | None = None,
) -> dict[str, Any]:
    """Integrate a calibrated time trace by the trapezoidal rule, optionally subtracting a measured baseline.

    The scale converts input units to the derivative/integrand units required by the user.
    This utility does not infer a sensor transfer function or physical calibration.

    Optional linear drift correction: when both a pre-shot baseline window
    (baseline_start_s..baseline_end_s) and a post-shot window (drift_start_s..drift_end_s)
    are given, the offset is assumed to drift linearly between the centres of the two
    windows and the interpolated offset is subtracted before integration. This is the
    standard remedy for integrator/pickup drift when the true signal is ~0 in both windows.
    """
    t = np.asarray(time_s, dtype=float).ravel()
    y = np.asarray(values, dtype=float).ravel()
    if len(t) != len(y) or len(t) < 2:
        raise ValueError('Time and signal must have equal lengths and at least two samples.')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError('Time and signal must contain only finite values.')
    if np.any(np.diff(t) <= 0):
        raise ValueError('Time values must be strictly increasing with no duplicates.')
    if not np.isfinite(scale) or not np.isfinite(initial_value):
        raise ValueError('Scale and initial value must be finite.')
    if (baseline_start_s is None) != (baseline_end_s is None):
        raise ValueError('Specify both baseline start and end, or neither.')
    if (drift_start_s is None) != (drift_end_s is None):
        raise ValueError('Specify both drift-window start and end, or neither.')
    if drift_start_s is not None and baseline_start_s is None:
        raise ValueError('Linear drift correction needs the pre-shot baseline window as well.')
    calibrated = y * float(scale)
    baseline = 0.0
    drift_rate = None
    if baseline_start_s is not None:
        if not np.isfinite(baseline_start_s) or not np.isfinite(baseline_end_s) or baseline_start_s >= baseline_end_s:
            raise ValueError('Baseline interval must be finite with start < end.')
        mask = (t >= baseline_start_s) & (t <= baseline_end_s)
        if np.count_nonzero(mask) < 2:
            raise ValueError('Baseline interval must include at least two samples.')
        baseline = float(np.mean(calibrated[mask]))
        if drift_start_s is not None:
            if not np.isfinite(drift_start_s) or not np.isfinite(drift_end_s) or drift_start_s >= drift_end_s:
                raise ValueError('Drift window must be finite with start < end.')
            mask2 = (t >= drift_start_s) & (t <= drift_end_s)
            if np.count_nonzero(mask2) < 2:
                raise ValueError('Drift window must include at least two samples.')
            t1 = float(np.mean(t[mask])); t2 = float(np.mean(t[mask2]))
            if t2 <= t1:
                raise ValueError('The post-shot drift window must lie after the baseline window.')
            b2 = float(np.mean(calibrated[mask2]))
            drift_rate = (b2 - baseline) / (t2 - t1)
            calibrated = calibrated - (baseline + drift_rate * (t - t1))
        else:
            calibrated = calibrated - baseline
    integrated = float(initial_value) + integrate.cumulative_trapezoid(calibrated, t, initial=0.0)
    return {
        'time_s': t,
        'calibrated_signal': calibrated,
        'integrated_signal': integrated,
        'scale': float(scale),
        'baseline_offset_in_calibrated_units': baseline,
        'linear_drift_rate_calibrated_units_per_s': drift_rate,
        'initial_value': float(initial_value),
        'n_samples': int(len(t)),
        'method': 'cumulative trapezoidal integration',
        'note': 'The result is physically meaningful only when the input signal, scale, baseline interval, sensor response and initial condition are appropriate for the diagnostic.',
    }


def prepare_time_series(time: Any, values: Any, sample_rate: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(values, dtype=float).ravel()
    if time is None:
        if sample_rate is None or not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError('A positive sample rate is required when no time column is supplied.')
        t = np.arange(len(y), dtype=float) / sample_rate
    else:
        t = np.asarray(time, dtype=float).ravel()
        if len(t) != len(y):
            raise ValueError('Time and signal columns must have the same length.')
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if len(t) < 8:
        raise ValueError('At least 8 finite samples are required for time-series analysis.')
    order = np.argsort(t, kind='stable')
    t, y = t[order], y[order]
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError('Time values must be strictly increasing after sorting; duplicate timestamps need handling.')
    med_dt = float(np.median(dt))
    if not np.isfinite(med_dt) or med_dt <= 0:
        raise ValueError('Could not infer a valid sampling interval.')
    # Detect isolated dropped samples as well as globally jittery clocks.
    jitter = float(np.std(dt) / np.mean(dt)) if len(dt)>1 else 0.0
    max_gap_ratio = float(np.max(dt) / med_dt)
    # Quantized clocks can alternate adjacent intervals; reject likely gaps,
    # not harmless timestamp rounding at high sample rates.
    outlier_fraction = float(np.mean(dt > 1.5 * med_dt))
    if max_gap_ratio > 1.6 or outlier_fraction > 0.01:
        raise ValueError(
            "A likely dropped-sample gap was detected "
            f"(std/mean={jitter:.3g}, max_dt/median_dt={max_gap_ratio:.3g}, "
            f"intervals >1.5× median={outlier_fraction:.2%}). "
            "Repair/resample explicitly before FFT/filtering."
        )
    # The median is robust for detecting gaps, but can be biased by one full
    # clock-quantization tick when intervals alternate (e.g. 2 and 3 µs).
    # After the gap check, use the full-record mean interval for the sample rate.
    mean_dt=float(np.mean(dt))
    if not np.isfinite(mean_dt) or mean_dt<=0:
        raise ValueError('Could not estimate a valid mean sampling interval.')
    return t, y, 1.0/mean_dt































def doppler_temperature_eV(sigma_lambda: float, lambda0: float, mass_amu: float) -> float:
    if not np.all(np.isfinite([sigma_lambda, lambda0, mass_amu])) or sigma_lambda < 0 or lambda0 <= 0 or mass_amu <= 0:
        raise ValueError('Width must be nonnegative; wavelength and mass must be positive; all inputs must be finite.')
    return float((mass_amu*AMU*C_LIGHT**2/E_CHARGE)*(sigma_lambda/lambda0)**2)



def doppler_temperature_from_fwhm_eV(
    observed_fwhm: float, lambda0: float, mass_amu: float,
    instrumental_fwhm: float = 0.0,
    sigma_observed_fwhm: float = 0.0,
    sigma_instrumental_fwhm: float = 0.0,
) -> dict[str, float | str]:
    """Calculate ion temperature from FWHM values in the same wavelength unit.

    Gaussian line profiles are assumed. Instrument broadening is removed in
    quadrature in FWHM space (equivalent to doing so in Gaussian sigma space).
    A nonzero instrumental width must be measured independently. The returned
    uncertainty propagates independent width uncertainties only and excludes
    calibration and model systematics.
    """
    vals = [observed_fwhm, lambda0, mass_amu, instrumental_fwhm,
            sigma_observed_fwhm, sigma_instrumental_fwhm]
    if not np.all(np.isfinite(vals)):
        raise ValueError("All FWHM, wavelength, mass and uncertainty inputs must be finite.")
    if observed_fwhm <= 0 or lambda0 <= 0 or mass_amu <= 0 or instrumental_fwhm < 0:
        raise ValueError("Observed FWHM, wavelength and mass must be positive; instrument FWHM must be nonnegative.")
    if sigma_observed_fwhm < 0 or sigma_instrumental_fwhm < 0:
        raise ValueError("FWHM uncertainties must be nonnegative.")
    if observed_fwhm <= instrumental_fwhm:
        raise ValueError("Observed FWHM does not exceed instrumental FWHM; Doppler width is unresolved.")
    doppler_fwhm = float(np.sqrt(observed_fwhm**2 - instrumental_fwhm**2))
    gaussian_sigma = doppler_fwhm / np.sqrt(8.0 * np.log(2.0))
    sigma_width_unc = np.sqrt(
        (observed_fwhm / doppler_fwhm * sigma_observed_fwhm)**2
        + (instrumental_fwhm / doppler_fwhm * sigma_instrumental_fwhm)**2
    ) / np.sqrt(8.0 * np.log(2.0))
    thermal = doppler_temperature_with_uncertainty_eV(
        gaussian_sigma, lambda0, mass_amu, sigma_width_unc
    )
    return {
        "temperature_eV": thermal["temperature_eV"],
        "standard_uncertainty_eV": thermal["standard_uncertainty_eV"],
        "observed_fwhm": float(observed_fwhm),
        "instrumental_fwhm": float(instrumental_fwhm),
        "doppler_fwhm": doppler_fwhm,
        "doppler_gaussian_sigma": float(gaussian_sigma),
        "relative_standard_uncertainty": thermal["relative_standard_uncertainty"],
        "method": "Gaussian FWHM quadrature deconvolution followed by Doppler temperature calculation",
        "warning": "Uncertainty excludes wavelength calibration, instrument line-spread model, blends, Stark/Zeeman broadening and other systematics.",
    }


def doppler_temperature_with_uncertainty_eV(
    sigma_lambda: float, lambda0: float, mass_amu: float,
    sigma_sigma_lambda: float = 0.0, sigma_lambda0: float = 0.0,
    sigma_mass_amu: float = 0.0,
) -> dict[str, float | str]:
    """Estimate Doppler temperature and first-order 1-sigma uncertainty.

    Inputs and their uncertainties must use the same wavelength units. Independent
    uncertainties are propagated in quadrature; model/systematic errors are not included.
    """
    vals=[sigma_lambda,lambda0,mass_amu,sigma_sigma_lambda,sigma_lambda0,sigma_mass_amu]
    if not np.all(np.isfinite(vals)) or sigma_lambda < 0 or lambda0 <= 0 or mass_amu <= 0:
        raise ValueError("Width must be nonnegative; wavelength and mass positive; inputs finite.")
    if min(sigma_sigma_lambda,sigma_lambda0,sigma_mass_amu) < 0:
        raise ValueError("Input uncertainties must be finite and nonnegative.")
    temperature=doppler_temperature_eV(sigma_lambda,lambda0,mass_amu)
    variance_terms=[]
    if sigma_lambda > 0:
        variance_terms.append((2.0*sigma_sigma_lambda/sigma_lambda)**2)
    elif sigma_sigma_lambda > 0:
        raise ValueError("First-order uncertainty propagation is undefined at zero Doppler width; use a profile-likelihood or Monte Carlo interval.")
    variance_terms.append((2.0*sigma_lambda0/lambda0)**2)
    variance_terms.append((sigma_mass_amu/mass_amu)**2)
    relative=float(np.sqrt(sum(variance_terms)))
    return {"temperature_eV":temperature,"standard_uncertainty_eV":float(temperature*relative),
            "relative_standard_uncertainty":relative,
            "uncertainty_method":"First-order independent-error propagation; excludes model/systematic uncertainty."}


def doppler_velocity_with_uncertainty_m_s(
    observed_lambda: float, lambda0: float,
    sigma_observed_lambda: float = 0.0, sigma_lambda0: float = 0.0,
) -> dict[str, float | str]:
    """Calculate nonrelativistic line-of-sight velocity and propagated 1-sigma uncertainty.

    Wavelengths and their uncertainties must use the same units. Independent measurement
    uncertainties are propagated in quadrature; systematic wavelength-calibration errors
    are not included unless explicitly represented in the supplied uncertainties.
    """
    vals=[observed_lambda,lambda0,sigma_observed_lambda,sigma_lambda0]
    if not np.all(np.isfinite(vals)) or observed_lambda <= 0 or lambda0 <= 0:
        raise ValueError("Observed and rest wavelengths must be positive and all inputs finite.")
    if sigma_observed_lambda < 0 or sigma_lambda0 < 0:
        raise ValueError("Wavelength uncertainties must be nonnegative.")
    velocity=float(C_LIGHT*(observed_lambda/lambda0-1.0))
    uncertainty=float(C_LIGHT*np.sqrt((sigma_observed_lambda/lambda0)**2+
                                     (observed_lambda*sigma_lambda0/lambda0**2)**2))
    return {"line_of_sight_velocity_m_s":velocity,
            "standard_uncertainty_m_s":uncertainty,
            "uncertainty_method":"First-order independent-error propagation; nonrelativistic Doppler approximation."}


def doppler_velocity_m_s(delta_lambda: float, lambda0: float) -> float:
    if not np.all(np.isfinite([delta_lambda, lambda0])) or lambda0 <= 0:
        raise ValueError('Wavelength shift must be finite and rest wavelength positive.')
    return float(C_LIGHT*delta_lambda/lambda0)


def linear_phase_density(phase_rad: Any, frequency_hz: float, sign_convention: int | None = None) -> np.ndarray:
    """Return line-integrated electron density ∫n_e dl in m⁻², not local density.

    sign_convention must be +1 or -1 and reflects the instrument's documented phase
    convention: Δφ = sign_convention * r_e * wavelength * ∫n_e dl. No default sign
    is assumed because the convention is instrument- and reference-phase-dependent.
    """
    phase=np.asarray(phase_rad,float)
    if not np.isfinite(frequency_hz) or frequency_hz<=0 or not np.all(np.isfinite(phase)):
        raise ValueError('Frequency must be positive and finite; phase values must be finite.')
    if sign_convention not in (-1,1):
        raise ValueError('Select sign_convention=+1 or -1 from the instrument calibration; no default is assumed.')
    r_e=2.8179403262e-15
    wavelength=C_LIGHT/frequency_hz
    return phase/(sign_convention*r_e*wavelength)


def derive_parameters(ne_m3: float, te_eV: float, ti_eV: float, ip_A: float, vloop_V: float,
                      major_radius_m: float, minor_radius_m: float, bt_T: float,
                      stored_energy_J: float | None = None, loss_power_W: float | None = None,
                      ni_m3: float | None = None, ion_mass_amu: float = 2.0) -> dict[str, float | None]:
    vals=[ne_m3,te_eV,ti_eV,ip_A,vloop_V,major_radius_m,minor_radius_m,bt_T,ion_mass_amu]
    if not np.all(np.isfinite(vals)) or min(ne_m3,te_eV,ti_eV,major_radius_m,minor_radius_m)<0 or major_radius_m==0 or minor_radius_m==0 or ion_mass_amu<=0:
        raise ValueError('Inputs must be finite; densities/temperatures nonnegative, radii and ion mass positive. Ip and Bt are signed quantities.')
    if ni_m3 is not None and (not np.isfinite(ni_m3) or ni_m3 < 0):
        raise ValueError('Ion density must be finite and nonnegative.')
    if stored_energy_J is not None and (not np.isfinite(stored_energy_J) or stored_energy_J < 0):
        raise ValueError('Stored energy must be finite and nonnegative when supplied.')
    if loss_power_W is not None and (not np.isfinite(loss_power_W) or loss_power_W < 0):
        raise ValueError('Loss power must be finite and nonnegative when supplied.')
    ni=ne_m3 if ni_m3 is None else ni_m3
    pe=ne_m3*te_eV*E_CHARGE; pi=ni*ti_eV*E_CHARGE; p=pe+pi
    fpe=float(np.sqrt(ne_m3*E_CHARGE**2/(EPS0*M_E))/(2*np.pi)) if ne_m3>0 else 0.0
    debye=float(np.sqrt(EPS0*te_eV*E_CHARGE/(ne_m3*E_CHARGE**2))) if ne_m3>0 and te_eV>0 else None
    ng=abs(ip_A)/1e6/(np.pi*minor_radius_m**2)
    p_ohm=vloop_V*ip_A
    beta_t=p/(bt_T**2/(2*MU0)) if abs(bt_T)>0 else None
    tau=stored_energy_J/loss_power_W if stored_energy_J is not None and loss_power_W is not None and loss_power_W>0 else None
    cs=float(np.sqrt(E_CHARGE*(te_eV+ti_eV)/ (ion_mass_amu*AMU))) if te_eV+ti_eV>0 else 0.0
    return {'electron_pressure_Pa':float(pe),'ion_pressure_Pa':float(pi),'total_thermal_pressure_Pa':float(p),
            'electron_plasma_frequency_Hz':fpe,'debye_length_m':debye,'greenwald_density_1e20_m-3':float(ng),
            'ohmic_power_approx_W':float(p_ohm),'toroidal_beta_fraction':float(beta_t) if beta_t is not None else None,
            'energy_confinement_time_s':float(tau) if tau is not None else None,'ion_sound_speed_approx_m_s':cs,
            'assumption_ion_density_equals_electron_density':bool(ni_m3 is None),
            'beta_definition_note':'Pointwise pressure-based estimate from supplied n_e, T_e and T_i; not a volume-averaged experimental beta.',
            'energy_confinement_note':'W/P_loss approximation only; dW/dt, transient energy balance, impurity radiation and Zeff are not included.'}


def uncertainty_linear(y: Any, slope: float, intercept: float, sigma_y: float = 0.0,
                       sigma_slope: float = 0.0, sigma_intercept: float = 0.0) -> tuple[np.ndarray,np.ndarray]:
    if not np.all(np.isfinite([slope, intercept, sigma_y, sigma_slope, sigma_intercept])) or min(sigma_y, sigma_slope, sigma_intercept) < 0:
        raise ValueError('Calibration coefficients and uncertainties must be finite; uncertainties must be nonnegative.')
    x=np.asarray(y,float)
    if not np.all(np.isfinite(x)):
        raise ValueError('Input values must be finite; clean missing/nonfinite samples before calibration.')
    cal=slope*x+intercept
    var=(slope*sigma_y)**2+(x*sigma_slope)**2+sigma_intercept**2
    return cal,np.sqrt(np.maximum(var,0))


def build_tomography_matrix_2d(ray_start_xz: Any, ray_end_xz: Any,
                               x_edges_m: Any, z_edges_m: Any) -> np.ndarray:
    """Build a 2-D line-integral response matrix from ray segments and rectangular pixel edges.

    Each row is one ray; columns are flattened as iz * nx + ix. Entries are path
    lengths through each pixel in metres. Ray start/end coordinates are (R or x, z)
    in metres. Segments are clipped to the grid; pixels outside the grid contribute 0.
    """
    starts=np.asarray(ray_start_xz,dtype=float)
    ends=np.asarray(ray_end_xz,dtype=float)
    x_edges=np.asarray(x_edges_m,dtype=float).ravel()
    z_edges=np.asarray(z_edges_m,dtype=float).ravel()
    if starts.ndim!=2 or starts.shape[1]!=2 or ends.shape!=starts.shape or starts.shape[0]<1:
        raise ValueError("Ray start and end arrays must both have shape (n_rays, 2).")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("Ray endpoints must be finite.")
    for name,edges in (("x",x_edges),("z",z_edges)):
        if len(edges)<2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges)<=0):
            raise ValueError(f"{name}_edges_m must be a finite, strictly increasing edge vector.")
    nx=len(x_edges)-1; nz=len(z_edges)-1
    matrix=np.zeros((len(starts),nx*nz),dtype=float)
    for row,(p0,p1) in enumerate(zip(starts,ends)):
        delta=p1-p0
        length=float(np.linalg.norm(delta))
        if length<=np.finfo(float).eps:
            raise ValueError(f"Ray {row} has coincident start and end points.")
        ts=[0.0,1.0]
        if abs(delta[0])>np.finfo(float).eps:
            ts.extend(((x-p0[0])/delta[0] for x in x_edges if 0.0<(x-p0[0])/delta[0]<1.0))
        if abs(delta[1])>np.finfo(float).eps:
            ts.extend(((z-p0[1])/delta[1] for z in z_edges if 0.0<(z-p0[1])/delta[1]<1.0))
        ts.sort()
        unique=[ts[0]]
        for value in ts[1:]:
            if value-unique[-1]>1e-12: unique.append(value)
        for left,right in zip(unique[:-1],unique[1:]):
            if right-left<=1e-14: continue
            mid=p0+((left+right)/2.0)*delta
            ix=int(np.searchsorted(x_edges,mid[0],side="right")-1)
            iz=int(np.searchsorted(z_edges,mid[1],side="right")-1)
            if 0<=ix<nx and 0<=iz<nz:
                matrix[row,iz*nx+ix]+=length*(right-left)
    if not np.any(matrix>0):
        raise ValueError("No ray segment intersects the supplied reconstruction grid.")
    return matrix


def regularized_tomography(A: Any, y: Any, lam: float = 1e-2, nonnegative: bool = True,
                          measurement_sigma: Any | None = None,
                          regularization_order: int = 1) -> dict[str, Any]:
    """Solve a normalized Tikhonov inverse problem with order-0/1/2 regularization."""
    A=np.asarray(A,float); y=np.asarray(y,float).ravel()
    if A.ndim!=2 or A.shape[0]!=len(y) or A.size==0: raise ValueError('A must be a nonempty 2D matrix with one row per measurement.')
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(y)): raise ValueError('Response matrix and measurements must contain only finite values.')
    if lam<0 or not np.isfinite(lam): raise ValueError('Regularization must be finite and nonnegative.')
    if isinstance(regularization_order,bool) or regularization_order not in (0,1,2):
        raise ValueError("regularization_order must be 0 (magnitude), 1 (gradient) or 2 (curvature).")
    if measurement_sigma is not None:
        sigma=np.asarray(measurement_sigma,float).ravel()
        if len(sigma)!=len(y) or not np.all(np.isfinite(sigma)) or np.any(sigma<=0):
            raise ValueError("measurement_sigma must be finite, positive and have one value per measurement.")
        Aw=A/sigma[:,None]; yw=y/sigma
    else:
        sigma=None; Aw=A; yw=y
    # Normalize A by its weighted spectral norm so lambda is dimensionless with
    # respect to response scale. First-order regularization penalizes profile
    # gradients rather than shrinking every emissivity value toward zero.
    response_scale=float(np.linalg.norm(Aw,ord=2))
    if not np.isfinite(response_scale) or response_scale<=np.finfo(float).eps:
        raise ValueError("Response matrix has zero or invalid weighted norm.")
    An=Aw/response_scale
    yn=yw/response_scale
    identity=np.eye(A.shape[1])
    operator=identity if regularization_order==0 else np.diff(identity,n=regularization_order,axis=0)
    operator_norm=float(np.linalg.norm(operator,ord=2)) if operator.size else 1.0
    if operator_norm>0: operator=operator/operator_norm
    if nonnegative:
        augA=np.vstack([An,np.sqrt(lam)*operator]) if lam else An
        augy=np.concatenate([yn,np.zeros(operator.shape[0])]) if lam else yn
        res=optimize.lsq_linear(augA,augy,bounds=(0,np.inf),method='trf')
        x=res.x; success=bool(res.success); message=str(res.message)
    else:
        if lam>0:
            augA=np.vstack([An,np.sqrt(lam)*operator])
            augy=np.concatenate([yn,np.zeros(operator.shape[0])])
        else:
            augA,augy=An,yn
        x=np.linalg.lstsq(augA,augy,rcond=None)[0]
        success=True; message='least-squares / Tikhonov solution'
    residual=A@x-y
    weighted_residual=residual if sigma is None else residual/sigma
    regularization_norm=float(np.linalg.norm(operator@x))
    return {'solution':x,'residual':residual,'residual_norm':float(np.linalg.norm(residual)),
            'weighted_residual_norm':float(np.linalg.norm(weighted_residual)),
            'regularization':float(lam),'regularization_order':int(regularization_order),
            'regularization_operator_norm':regularization_norm,'response_spectral_norm':response_scale,
            'success':success,'message':message,'nonnegative':bool(nonnegative),
            'measurement_weighting':'unweighted' if sigma is None else 'inverse-variance (1/sigma²)'}


def select_tikhonov_lambda(A: Any, y: Any, lambdas: Any | None = None,
                           nonnegative: bool = True, measurement_sigma: Any | None = None) -> dict[str, Any]:
    """Choose a candidate Tikhonov parameter from discrete L-curve curvature.

    This is a heuristic selection aid, not a substitute for a noise model or validation
    against known profiles. The full curve is returned for review.
    """
    candidates=np.logspace(-8,2,31) if lambdas is None else np.asarray(lambdas,float).ravel()
    if len(candidates)<3 or not np.all(np.isfinite(candidates)) or np.any(candidates<=0) or np.any(np.diff(candidates)<=0):
        raise ValueError("Provide at least three finite, positive, strictly increasing candidate lambdas.")
    residuals=[]; norms=[]
    for value in candidates:
        result=regularized_tomography(A,y,float(value),nonnegative,measurement_sigma)
        residuals.append(max(result['weighted_residual_norm'],np.finfo(float).tiny))
        norms.append(max(float(result['regularization_operator_norm']),np.finfo(float).tiny))
    lx=np.log(residuals); ly=np.log(norms); lt=np.log(candidates)
    dx=np.gradient(lx,lt); dy=np.gradient(ly,lt)
    ddx=np.gradient(dx,lt); ddy=np.gradient(dy,lt)
    curvature=np.abs(dx*ddy-dy*ddx)/np.maximum((dx*dx+dy*dy)**1.5,np.finfo(float).tiny)
    interior=np.arange(1,len(candidates)-1)
    selected=int(interior[np.argmax(curvature[interior])])
    return {'selected_lambda':float(candidates[selected]),'candidate_lambdas':candidates,
            'weighted_residual_norms':np.asarray(residuals),
            'regularization_norms':np.asarray(norms),'solution_norms':np.asarray(norms),
            'discrete_log_lcurve_curvature':curvature,
            'selection_method':'maximum discrete curvature of log-weighted-residual vs log-regularization-norm L-curve',
            'warning':'Heuristic selection; inspect the returned L-curve and validate with measurement noise and synthetic/known-profile tests.'}


def spectral_calibration(raw_pixel: Any, reference_wavelength: Any, degree: int = 1) -> dict[str, Any]:
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError('Polynomial degree must be a positive integer.')
    x=np.asarray(raw_pixel,float).ravel(); y=np.asarray(reference_wavelength,float).ravel()
    if len(x) != len(y): raise ValueError('Pixel and reference-wavelength arrays must have equal length.')
    good=np.isfinite(x)&np.isfinite(y); x,y=x[good],y[good]
    if len(x)<degree+2: raise ValueError('Need at least degree+2 finite calibration points.')
    if len(np.unique(x))<degree+1: raise ValueError('Pixel coordinates do not span enough unique points for this polynomial degree.')
    coeff=np.polyfit(x,y,degree); pred=np.polyval(coeff,x); resid=y-pred
    return {'coefficients_high_to_low':coeff.tolist(),'rms_residual':float(np.sqrt(np.mean(resid**2))),
            'max_abs_residual':float(np.max(np.abs(resid))),'n_lines':int(len(x)),'degree':int(degree),
            'calibrated_values':np.polyval(coeff,x).tolist(),'residuals':resid.tolist()}


def probe_iv_summary(voltage: Any, current: Any) -> dict[str, Any]:
    """Descriptive I-V preprocessing only; does not silently infer Te/ne without a validated regime model."""
    v=np.asarray(voltage,float).ravel(); i=np.asarray(current,float).ravel()
    good=np.isfinite(v)&np.isfinite(i); v,i=v[good],i[good]
    if len(v)<5: raise ValueError('At least 5 finite I-V points are required.')
    order=np.argsort(v); v,i=v[order],i[order]
    unique_v, inverse = np.unique(v, return_inverse=True)
    if len(unique_v)<5: raise ValueError('Voltage must contain at least 5 unique values.')
    if len(unique_v)<len(v):
        sums=np.bincount(inverse,weights=i)
        counts=np.bincount(inverse)
        v,i=unique_v,sums/counts
    dIdV=np.gradient(i,v)
    return {'n_points':int(len(v)),'voltage_min_V':float(v.min()),'voltage_max_V':float(v.max()),
            'current_min_A':float(i.min()),'current_max_A':float(i.max()),
            'floating_potential_interpolated_V':float(np.interp(0, i[np.argsort(i)], v[np.argsort(i)])) if (np.all(np.diff(i)>=0) or np.all(np.diff(i)<=0)) and (i.min()<=0<=i.max()) else None,
            'max_dIdV_A_per_V':float(np.max(dIdV)),'min_dIdV_A_per_V':float(np.min(dIdV)),
            'warning':'Descriptive summary only. Plasma potential, Te and density require a justified probe regime/sheath model, collection geometry and uncertainty.'}

# Expanded diagnostic workflows (v0.3): calculations operate on user-supplied,
# calibrated data. They do not invent device calibration or geometry.
def langmuir_iv_analysis(voltage: Any, current: Any, electron_fit_range: tuple[float,float] | None = None,
                         ion_fit_range: tuple[float,float] | None = None, area_m2: float | None = None,
                         ion_mass_amu: float | None = None, ti_eV: float = 0.0,
                         sheath_factor: float = 0.61, gamma_i: float = 1.0,
                         current_convention: str = "electron_positive") -> dict[str, Any]:
    """Analyze a single Langmuir I-V sweep with explicit model-input validation.

    current_convention: "electron_positive" (default; ion saturation current is negative)
    or "ion_positive" (collected-ion current is positive, as in many probe circuits); the
    latter is flipped internally so the analysis always works in the electron-positive form.
    """
    if current_convention not in {"electron_positive", "ion_positive"}:
        raise ValueError("current_convention must be 'electron_positive' or 'ion_positive'.")
    for name, fit_range in (("electron_fit_range", electron_fit_range), ("ion_fit_range", ion_fit_range)):
        if fit_range is not None:
            if len(fit_range) != 2 or not np.all(np.isfinite(np.asarray(fit_range, dtype=float))):
                raise ValueError(f"{name} must contain exactly two finite voltage limits.")
    if area_m2 is None:
        ion_mass_amu = None
    elif ion_mass_amu is None:
        raise ValueError("area_m2 and ion_mass_amu must be supplied together for density estimation.")
    if area_m2 is not None and (not np.isfinite(area_m2) or area_m2 <= 0):
        raise ValueError("Collection area must be positive and finite.")
    if ion_mass_amu is not None and (not np.isfinite(ion_mass_amu) or ion_mass_amu <= 0):
        raise ValueError("Ion mass must be positive and finite.")
    if not np.all(np.isfinite([ti_eV, sheath_factor, gamma_i])) or ti_eV < 0 or sheath_factor <= 0 or gamma_i < 0:
        raise ValueError("ti_eV and gamma_i must be finite and nonnegative; sheath_factor must be positive and finite.")
    v=np.asarray(voltage,float).ravel(); i=np.asarray(current,float).ravel()
    if current_convention=="ion_positive": i=-i
    good=np.isfinite(v)&np.isfinite(i); v,i=v[good],i[good]
    if len(v)<8: raise ValueError('At least 8 valid voltage points are required.')
    order=np.argsort(v); v,i=v[order],i[order]
    unique_v, inverse = np.unique(v, return_inverse=True)
    if len(unique_v)<8: raise ValueError('At least 8 distinct voltage points are required after duplicate aggregation.')
    if len(unique_v)<len(v):
        sums=np.bincount(inverse,weights=i)
        counts=np.bincount(inverse)
        v,i=unique_v,sums/counts
    out=probe_iv_summary(v,i)
    # Handle exact zero-current samples explicitly; avoid 0/0 for repeated zero values.
    exact_zero=np.flatnonzero(i == 0)
    cross=np.where(i[:-1]*i[1:]<0)[0]
    if len(exact_zero):
        out['floating_potential_V']=float(v[exact_zero[0]])
    elif len(cross):
        k=int(cross[0])
        out['floating_potential_V']=float(v[k]-i[k]*(v[k+1]-v[k])/(i[k+1]-i[k]))
    else:
        out['floating_potential_V']=None
    didv=np.gradient(i,v)
    if not np.all(np.isfinite(didv)):
        raise ValueError("dI/dV is non-finite; voltage values must be distinct and well-conditioned.")
    out['max_dIdV_A_per_V']=float(np.max(didv)); out['plasma_potential_proxy_V']=float(v[np.argmax(didv)])
    if ion_fit_range is not None:
        m=(v>=min(ion_fit_range))&(v<=max(ion_fit_range))
        if m.sum()>=3:
            p=np.polyfit(v[m],i[m],1); out['ion_saturation_linear_slope_A_per_V']=float(p[0]); out['ion_saturation_intercept_A']=float(p[1]); out['ion_saturation_fit_R2']=_r2(i[m],np.polyval(p,v[m]))
            out['ion_saturation_current_A']=float(np.median(i[m]))
        else:
            raise ValueError("ion_fit_range must contain at least 3 valid samples.")
    if electron_fit_range is not None:
        ion_baseline=float(out.get('ion_saturation_current_A',0.0))
        electron_current=i-ion_baseline
        m=(v>=min(electron_fit_range))&(v<=max(electron_fit_range))&(electron_current>0)
        if m.sum()<4: raise ValueError('Electron-retarding fit range needs at least 4 points with positive current after ion-current baseline subtraction.')
        if m.sum()>=5:
            p,pcov=np.polyfit(v[m],np.log(electron_current[m]),1,cov=True); slope_se=float(np.sqrt(max(pcov[0,0],0.0)))
        else:
            p=np.polyfit(v[m],np.log(electron_current[m]),1); slope_se=None
        pred=np.polyval(p,v[m]); slope=float(p[0])
        out['electron_retarding_fit_R2']=_r2(np.log(electron_current[m]),pred)
        out['ion_current_baseline_subtracted_A']=ion_baseline
        if slope>0:
            out['electron_temperature_eV']=1.0/slope
            out['electron_temperature_standard_error_eV']=(slope_se/slope**2) if slope_se is not None else None
            out['electron_temperature_uncertainty_note']='Statistical (unweighted log-linear fit) only; excludes fit-window choice, sheath expansion and RF/noise systematics.'
            out['electron_retarding_log_slope_per_V']=slope
            out['electron_retarding_fit_intercept']=float(p[1])
        else: out['electron_temperature_eV']=None
    if area_m2 is not None and 'ion_saturation_current_A' not in out:
        out['ion_density_status'] = "not_estimated: provide an ion_fit_range containing at least three points"
    if area_m2 is not None and 'ion_saturation_current_A' in out:
        te_value = out.get('electron_temperature_eV')
        if te_value is None or not np.isfinite(te_value) or te_value <= 0:
            raise ValueError("Ion-density estimation requires a valid positive electron temperature from the electron-retarding fit; refusing to substitute Te=0.")
        te=float(te_value)
        cs=math.sqrt(E_CHARGE*(ti_eV*gamma_i + te)/(ion_mass_amu*AMU))
        denom=sheath_factor*E_CHARGE*area_m2*cs
        out['ion_sound_speed_m_s']=cs
        out['ion_density_estimate_m-3']=abs(out['ion_saturation_current_A'])/denom if denom>0 else None
        out['ion_density_model']='n_i≈|I_isat|/(sheath_factor·e·A·c_s); depends on sheath model, geometry and ion temperature.'
    out['current_convention']=current_convention
    if 'ion_saturation_current_A' in out and out['ion_saturation_current_A']>0:
        out['sign_warning']='Ion-saturation current is positive in the electron-positive convention; check current_convention.'
    out['interpretation_warning']='Single-probe estimates depend on sweep direction, magnetic field, collisions, secondary emission, sheath expansion, surface condition and fit-window choice. Check fit R² and device-specific probe theory.'
    return out

def floating_potential(voltage: Any, current: Any) -> float | None:
    v=np.asarray(voltage,float).ravel(); i=np.asarray(current,float).ravel(); m=np.isfinite(v)&np.isfinite(i); v,i=v[m],i[m]
    if len(v)<2: return None
    o=np.argsort(v); v,i=v[o],i[o]; ix=np.where(i[:-1]*i[1:]<=0)[0]
    if not len(ix): return None
    k=ix[0]
    if i[k+1]==i[k]: return float((v[k]+v[k+1])/2)
    return float(v[k]-i[k]*(v[k+1]-v[k])/(i[k+1]-i[k]))


_MODE_CONVENTIONS = {
    # relative phase (sensor - reference, scipy CSD) = +n * theta
    "phase_equals_n_theta": 1,
    # wave cos(omega*t - n*theta): propagating toward +theta has n>0; CSD phase = -n*theta
    "wave_cos_wt_minus_n_theta": -1,
}


def mode_number_from_phase(theta_rad: Any, phase_rad: Any, weights: Any | None = None,
                           max_mode_number: int | None = None,
                           convention: str = "phase_equals_n_theta") -> dict[str, float | str | int]:
    """Fit spatial phase after sorting sensors by angle; report the sampling alias limit.

    Phase-only data cannot distinguish aliased modes (for example n=7 and n=1
    on six equally spaced sensors). Supply a physically justified max_mode_number
    to reject unsupported modes; a mode at the spatial Nyquist limit is ambiguous.
    Optional weights should be coherence or inverse-variance weights.

    Sign convention (explicit, because it is the most common source of sign errors):
    "phase_equals_n_theta" (default) fits phase = +n*theta. For phases obtained from
    scipy cross-spectra of a wave cos(omega*t - n*theta) this returns -n, so choose
    "wave_cos_wt_minus_n_theta" to obtain +n for a wave propagating toward +theta.
    """
    if convention not in _MODE_CONVENTIONS:
        raise ValueError(f"convention must be one of {sorted(_MODE_CONVENTIONS)}.")
    th=np.asarray(theta_rad,float).ravel(); ph=np.asarray(phase_rad,float).ravel()
    if len(th)!=len(ph) or len(th)<3 or not np.all(np.isfinite(th)) or not np.all(np.isfinite(ph)):
        raise ValueError('Provide at least 3 paired finite sensor angles and phases.')
    th=np.mod(th,2*np.pi)
    order=np.argsort(th, kind='stable'); th,ph=th[order],ph[order]
    if np.unique(np.round(th,12)).size < 3:
        raise ValueError('At least 3 distinct sensor angles modulo 2π are required.')
    gaps=np.diff(np.r_[th, th[0]+2*np.pi])
    if np.any(gaps <= 1e-10):
        raise ValueError('Sensor angles must be unique modulo 2π.')
    alias_limit=max(0, int(np.ceil(np.pi/np.max(gaps) - 1e-12)) - 1)
    if alias_limit < 1:
        raise ValueError('Sensor geometry is too sparse for an unambiguous nonzero mode estimate.')
    if max_mode_number is not None and (isinstance(max_mode_number,bool) or int(max_mode_number)!=max_mode_number or max_mode_number < 0):
        raise ValueError('max_mode_number must be a nonnegative integer when supplied.')
    limit=min(alias_limit,int(max_mode_number)) if max_mode_number is not None else alias_limit
    if weights is None:
        w=np.ones(len(th))
    else:
        w0=np.asarray(weights,float).ravel()
        if len(w0)!=len(th) or not np.all(np.isfinite(w0)) or np.any(w0<0) or not np.any(w0>0):
            raise ValueError('Weights must be finite, nonnegative, match sensor count, and include a positive value.')
        w=w0[order]
    # Zero-weight sensors must not participate in phase branch selection.
    keep=w>0
    th,ph,w=th[keep],ph[keep],w[keep]
    if len(th)<3:
        raise ValueError("At least three positive-weight sensors are required for mode fitting.")
    # Recompute the resolvable mode bound after excluding zero-weight sensors:
    # the resulting angular gaps, not the original sensor count, set the limit.
    active_gaps=np.diff(np.r_[th, th[0]+2*np.pi])
    active_limit=max(0, int(np.ceil(np.pi/np.max(active_gaps)-1e-12))-1)
    if active_limit<1:
        raise ValueError("Positive-weight sensor geometry is too sparse for a nonzero mode estimate.")
    limit=min(active_limit,int(max_mode_number)) if max_mode_number is not None else active_limit
    th0=th-th[0]
    candidates=np.arange(-limit-1,limit+2,dtype=int)
    scores=np.asarray([abs(np.sum(w*np.exp(1j*(ph-n*th0))))/np.sum(w) for n in candidates])
    n_int=int(candidates[int(np.argmax(scores))])
    if abs(n_int)>limit:
        raise ValueError(
            f"Spatial phase pattern is consistent with boundary/aliased mode n={_MODE_CONVENTIONS[convention]*n_int} ({convention}); "
            f"the largest unambiguous integer mode for active sensor geometry is {limit}."
        )
    phase_offset=float(np.angle(np.sum(w*np.exp(1j*(ph-n_int*th0)))))
    residual=np.angle(np.exp(1j*(ph-(n_int*th0+phase_offset))))
    rms=float(np.sqrt(np.average(residual**2,weights=w)))
    base=np.angle(np.exp(1j*(ph-phase_offset)))
    denom=float(np.sum(w*base**2))
    r2=float(1.0-float(np.sum(w*residual**2))/denom) if denom>np.finfo(float).eps else (1.0 if rms<1e-12 else 0.0)
    sgn=_MODE_CONVENTIONS[convention]
    return {'mode_number_fit':float(sgn*n_int),'nearest_integer_mode_number':float(sgn*n_int),'sign_convention':convention,'phase_offset_rad':phase_offset,
            'phase_fit_R2':r2,'phase_rms_residual_rad':rms,
            'spatial_alias_limit_mode_number':limit,
            'aliasing_note':'Integer circular-phase search; modes above the spatial sampling limit alias and require a justified prior bound.'}


def mode_number_from_channels(theta_rad: Any, signals: Any, fs: float,
                              frequency_range_hz: tuple[float, float] | None = None,
                              min_coherence: float = 0.2,
                              convention: str = "phase_equals_n_theta") -> dict[str, Any]:
    """Extract relative cross-spectral phases from synchronized sensor traces and fit a mode number.

    The first sensor is the phase reference. The dominant shared spectral peak is selected
    unless a frequency band is supplied; coherence with the reference weights the phase fit.
    Sensor geometry must be a single toroidal/poloidal ring with a consistent angle convention.
    """
    theta = np.asarray(theta_rad, dtype=float).ravel()
    data = np.asarray(signals, dtype=float)
    if data.ndim != 2 or data.shape[0] != len(theta) or data.shape[0] < 3 or data.shape[1] < 64:
        raise ValueError("signals must have shape (at least 3 sensors, at least 64 time samples) matching theta.")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(data)) or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sensor angles and synchronized signals must be finite; fs must be positive.")
    if not np.isfinite(min_coherence) or not 0 <= min_coherence <= 1:
        raise ValueError("min_coherence must be in [0, 1].")
    if frequency_range_hz is not None:
        if len(frequency_range_hz) != 2 or not np.all(np.isfinite(frequency_range_hz)) or frequency_range_hz[0] < 0 or frequency_range_hz[0] >= frequency_range_hz[1] or frequency_range_hz[1] > fs/2:
            raise ValueError("frequency_range_hz must be a finite increasing interval within [0, Nyquist].")
    data = data - np.mean(data, axis=1, keepdims=True)
    nperseg = min(data.shape[1], max(64, min(4096, data.shape[1]//2)))
    noverlap = nperseg // 2
    f, p0 = signal.welch(data[0], fs=fs, nperseg=nperseg, noverlap=noverlap)
    spectra = [p0]
    coherence_rows = [np.ones_like(f)]
    phase_rows = [np.zeros_like(f)]
    for row in data[1:]:
        _, p = signal.welch(row, fs=fs, nperseg=nperseg, noverlap=noverlap)
        fc, pxy = signal.csd(data[0], row, fs=fs, nperseg=nperseg, noverlap=noverlap)
        _, coh = signal.coherence(data[0], row, fs=fs, nperseg=nperseg, noverlap=noverlap)
        spectra.append(p)
        coherence_rows.append(np.clip(coh, 0.0, 1.0))
        phase_rows.append(np.angle(pxy))
    power = np.asarray(spectra)
    coherence_matrix = np.asarray(coherence_rows)
    # Score common coherent structure, not the largest raw summed PSD peak:
    # a large low-frequency drift in one channel must not mask a shared mode.
    normalized_power = power / np.maximum(np.sum(power, axis=1, keepdims=True), np.finfo(float).tiny)
    common_score = np.sum(normalized_power, axis=0) * np.mean(coherence_matrix, axis=0)
    allowed = f > 0
    if frequency_range_hz is not None:
        allowed &= (f >= frequency_range_hz[0]) & (f <= frequency_range_hz[1])
    if not np.any(allowed):
        raise ValueError("No positive-frequency bins fall inside the selected frequency band.")
    candidates = np.flatnonzero(allowed)
    peak_index = int(candidates[np.argmax(common_score[candidates])])
    peak_frequency = float(f[peak_index])
    phases = [float(phase_row[peak_index]) for phase_row in phase_rows]
    coherences = [float(coh_row[peak_index]) for coh_row in coherence_matrix]
    # Magnitude-squared coherence significance threshold under the null:
    # C_alpha = 1 - alpha**(1/(K-1)), with K Welch segments.
    step = max(1, nperseg - noverlap)
    n_segments = 1 + max(0, (data.shape[1] - nperseg) // step)
    significance_threshold = float(1.0 - 0.05 ** (1.0 / max(1, n_segments - 1)))
    effective_threshold = max(float(min_coherence), significance_threshold)
    weights = np.asarray(coherences)
    if np.count_nonzero(weights >= effective_threshold) < 3:
        raise ValueError(
            f"Fewer than three sensors exceed the coherence significance threshold "
            f"({effective_threshold:.3f}; {n_segments} Welch segments, alpha=0.05). "
            "Increase record length or choose a physically justified frequency band."
        )
    weights = np.where(weights >= effective_threshold, weights, 0.0)
    fit = mode_number_from_phase(theta, phases, weights=weights, convention=convention)
    fit.update({
        "selected_frequency_Hz": peak_frequency,
        "sensor_phase_rad_relative_to_first": [float(v) for v in phases],
        "coherence_with_reference": [float(v) for v in coherences],
        "minimum_coherence_threshold_user": float(min_coherence),
        "coherence_significance_threshold": significance_threshold,
        "effective_coherence_threshold": effective_threshold,
        "welch_segment_count": int(n_segments),
        "frequency_selection_score": "sum of per-channel normalized PSD weighted by mean reference coherence",
        "phase_extraction_method": "cross-spectral phase relative to first sensor; coherence-weighted integer circular fit",
    })
    return fit


def fluctuation_flux(density: Any, radial_velocity: Any, temperature_eV: Any | None = None) -> dict[str, float]:
    """Compute covariance fluxes from aligned physical-unit time series."""
    n=np.asarray(density,float).ravel(); vr=np.asarray(radial_velocity,float).ravel()
    if len(n)!=len(vr) or len(n)<3: raise ValueError('Density and radial velocity must be aligned and have at least 3 samples.')
    te=None
    if temperature_eV is not None:
        te=np.asarray(temperature_eV,float).ravel()
        if len(te)!=len(n): raise ValueError('Temperature series must have same original length as density.')
    good=np.isfinite(n)&np.isfinite(vr)
    if te is not None: good &= np.isfinite(te)
    n,vr=n[good],vr[good]
    if te is not None: te=te[good]
    if len(n)<3: raise ValueError('Not enough finite aligned samples after applying the common validity mask.')
    nf=n-np.mean(n); vf=vr-np.mean(vr)
    out={'particle_flux_covariance_m-2_s-1':float(np.mean(nf*vf)),'mean_density_m-3':float(np.mean(n)),'mean_radial_velocity_m_s':float(np.mean(vr))}
    if te is not None:
        tf=te-np.mean(te)
        mean_n=float(np.mean(n)); mean_te=float(np.mean(te))
        term_n=mean_te*float(np.mean(nf*vf))
        term_te=mean_n*float(np.mean(tf*vf))
        term_triple=float(np.mean(nf*tf*vf))
        out['density_temperature_velocity_covariance_eV_m-2_s-1']=term_triple
        out['electron_heat_flux_density_fluctuation_component_W_m-2']=float(1.5*E_CHARGE*term_n)
        out['electron_heat_flux_temperature_fluctuation_component_W_m-2']=float(1.5*E_CHARGE*term_te)
        out['electron_heat_flux_triple_fluctuation_component_W_m-2']=float(1.5*E_CHARGE*term_triple)
        out['electron_heat_flux_proxy_W_m-2']=float(1.5*E_CHARGE*(term_n+term_te+term_triple))
        out['heat_flux_note']='Decomposition of <n T_e delta(v_r)> into density, temperature and triple-fluctuation terms. Requires synchronized calibrated series; it is not a complete transport power balance.'
    out['note']='Covariance estimates require synchronized measurements, calibrated units and justified radial-velocity reconstruction.'
    return out


def exponential_heat_flux_fit(distance_m: Any, heat_flux_W_m2: Any) -> dict[str, Any]:
    """Fit q(s)=q0 exp(-(s-s0)/lambda_q)+background with covariance diagnostics."""
    x=np.asarray(distance_m,float).ravel(); y=np.asarray(heat_flux_W_m2,float).ravel(); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<8 or len(np.unique(x))<8: raise ValueError('At least 8 distinct valid target-coordinate points are required for a stable three-parameter fit.')
    o=np.argsort(x); x,y=x[o],y[o]
    span=float(np.ptp(x))
    if span<=0: raise ValueError("Target coordinate must span a nonzero interval.")
    base0=float(np.percentile(y,10)); q0=max(float(np.max(y)-base0),1e-12)
    def fn(z,q,l,b): return q*np.exp(-(z-x[0])/l)+b
    lower=[0,span/10000,-np.inf]; upper=[np.inf,span*100,np.inf]
    try:
        p,cov=optimize.curve_fit(fn,x,y,p0=[q0,max(span/5,1e-9),base0],bounds=(lower,upper),maxfev=50000)
    except (RuntimeError,ValueError,FloatingPointError) as exc:
        raise ValueError(f"Exponential heat-flux fit failed to converge: {exc}") from exc
    if cov.shape!=(3,3) or not np.all(np.isfinite(cov)):
        raise ValueError("Exponential fit covariance is singular or non-finite; inspect target coverage and signal-to-noise.")
    pred=fn(x,*p); se=np.sqrt(np.maximum(np.diag(cov),0.0))
    boundary=bool(np.isclose(p[1],lower[1],rtol=1e-3,atol=0) or np.isclose(p[1],upper[1],rtol=1e-3,atol=0))
    return {'q0_W_m-2':float(p[0]),'lambda_q_m':float(p[1]),'background_W_m-2':float(p[2]),
            'standard_error_q0_W_m-2':float(se[0]),'standard_error_lambda_q_m':float(se[1]),
            'standard_error_background_W_m-2':float(se[2]),'r_squared':_r2(y,pred),'n_points':int(len(x)),
            'parameter_at_bound':boundary,
            'note':'Covariance uncertainties assume the fitted model and independent homoscedastic residuals. Inspect residuals, target mapping, parameter identifiability and boundary flags; this is not a heat-transport model.'}


def eich_heat_flux_fit(distance_m: Any, heat_flux_W_m2: Any) -> dict[str, Any]:
    """Fit the standard Eich-like divertor heat-flux profile including Gaussian spreading.

    q(s) = q0/2 * exp(-((s-s0)/S)^2) * erfcx(S/(2*lambda_q) - (s-s0)/S) + q_bg.
    The fit is descriptive and assumes the supplied coordinate is already mapped to target
    distance; it does not infer equilibrium mapping or account for diagnostic response.
    """
    x=np.asarray(distance_m,float).ravel(); y=np.asarray(heat_flux_W_m2,float).ravel()
    valid=np.isfinite(x)&np.isfinite(y); x,y=x[valid],y[valid]
    if len(x)<15 or len(np.unique(x))<15:
        raise ValueError("Eich fit requires at least 15 distinct finite target-coordinate points for five free parameters.")
    order=np.argsort(x); x,y=x[order],y[order]
    span=float(np.ptp(x))
    if span<=0: raise ValueError("Target coordinate must span a nonzero interval.")
    baseline=float(np.percentile(y,10)); peak=float(np.max(y))
    if peak<=baseline: raise ValueError("Heat-flux profile must contain a positive signal above its baseline.")
    def model(s,q0,lam,S,s0,bg):
        u=(s-s0)/S
        a=S/(2*lam)
        arg=a-u
        core=np.empty_like(u,dtype=float)
        positive=arg>=0
        # For arg>=0 use erfcx; for arg<0 use the equivalent erfc form whose
        # exponential factor is bounded because u>a in this branch.
        core[positive]=np.exp(-u[positive]*u[positive])*erfcx(arg[positive])
        core[~positive]=np.exp(a*a-2*a*u[~positive])*erfc(arg[~positive])
        return 0.5*q0*core+bg
    p0=[max(peak-baseline,1e-12),max(span/30,span/10000),max(span/50,span/10000),float(x[np.argmax(y)]),baseline]
    lower=[0.0,span/10000,span/10000,float(x[0]-span),-np.inf]
    upper=[np.inf,span*100,span*10,float(x[-1]+span),np.inf]
    try:
        p,cov=optimize.curve_fit(model,x,y,p0=p0,bounds=(lower,upper),maxfev=100000)
    except (RuntimeError,ValueError,FloatingPointError) as exc:
        raise ValueError(f"Eich-like heat-flux fit failed to converge: {exc}") from exc
    if cov.shape!=(5,5) or not np.all(np.isfinite(cov)):
        raise ValueError("Eich fit covariance is singular or non-finite; lambda_q and spreading S may be unidentifiable.")
    pred=model(x,*p); se=np.sqrt(np.maximum(np.diag(cov),0.0))
    near_bounds=[bool(np.isclose(p[i],lower[i],rtol=1e-3,atol=0) or (np.isfinite(upper[i]) and np.isclose(p[i],upper[i],rtol=1e-3,atol=0))) for i in range(5)]
    return {
        "q0_W_m-2":float(p[0]),"lambda_q_m":float(p[1]),"spreading_S_m":float(p[2]),
        "strike_point_s0_m":float(p[3]),"background_W_m-2":float(p[4]),
        "standard_error_q0_W_m-2":float(se[0]),"standard_error_lambda_q_m":float(se[1]),
        "standard_error_spreading_S_m":float(se[2]),"standard_error_strike_point_s0_m":float(se[3]),
        "standard_error_background_W_m-2":float(se[4]),"r_squared":_r2(y,pred),"n_points":int(len(x)),
        "parameters_at_bounds":{"q0":near_bounds[0],"lambda_q":near_bounds[1],"S":near_bounds[2],"s0":near_bounds[3],"background":near_bounds[4]},
        "lambda_q_S_identifiability_warning":bool(near_bounds[1] or near_bounds[2] or abs(float(cov[1,2]))/max(float(np.sqrt(abs(cov[1,1]*cov[2,2]))),np.finfo(float).tiny)>0.98),
        "model":"Eich-like exponential-convolved-with-Gaussian profile",
        "note":"Parameter identifiability depends on target coverage, noise, spatial mapping and instrument response; inspect covariance and fit residuals before physical interpretation."
    }


def integrate_profile(coordinate: Any, profile: Any) -> dict[str, float]:
    x=np.asarray(coordinate,float).ravel(); y=np.asarray(profile,float).ravel(); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<2: raise ValueError('At least two finite coordinate-profile points are required.')
    o=np.argsort(x); x,y=x[o],y[o]
    if np.any(np.diff(x)<=0): raise ValueError('Coordinate values must be unique after sorting.')
    return {'integral_trapezoid':float(integrate.trapezoid(y,x)),'coordinate_min':float(x[0]),'coordinate_max':float(x[-1]),'profile_max':float(np.max(y)),'profile_min':float(np.min(y))}


def spectrum_moments(energy_eV: Any, counts: Any, background: Any | None = None,
                     count_uncertainty: Any | None = None,
                     assume_poisson_counts: bool = False,
                     bin_representation: str = "density_per_eV") -> dict[str, float | str | int | bool]:
    """Calculate moments with an explicit interpretation of spectral bins.

    "density_per_eV" integrates signal density over the energy coordinate with trapezoidal
    quadrature. "counts_per_bin" treats each input as a bin-integrated count and sums bins
    directly. Poisson uncertainty is inferred only when explicitly requested for raw counts.
    """
    if bin_representation not in {"density_per_eV", "counts_per_bin"}:
        raise ValueError("bin_representation must be 'density_per_eV' or 'counts_per_bin'.")
    e=np.asarray(energy_eV,float).ravel(); raw=np.asarray(counts,float).ravel(); c=raw.copy()
    if len(e)!=len(c) or len(e)<3: raise ValueError('Energy and counts must have same length and at least 3 samples.')
    if not np.all(np.isfinite(e)): raise ValueError('Energy coordinates must be finite.')
    if assume_poisson_counts and background is not None and count_uncertainty is None:
        raise ValueError("Poisson uncertainty cannot be inferred from background-subtracted net counts; supply count_uncertainty including background uncertainty.")
    if count_uncertainty is not None:
        sigma=np.asarray(count_uncertainty,float).ravel()
        if len(sigma)!=len(c) or not np.all(np.isfinite(sigma)) or np.any(sigma<0):
            raise ValueError('count_uncertainty must be finite, nonnegative and match counts length.')
    elif assume_poisson_counts:
        if np.any(raw<0): raise ValueError("Raw Poisson counts must be nonnegative.")
        sigma=np.sqrt(raw)
    else:
        sigma=None
    if background is not None:
        b=np.asarray(background,float).ravel()
        if len(b)!=len(c) or not np.all(np.isfinite(b)): raise ValueError('Background must be finite and match counts length.')
        c=c-b
    m=np.isfinite(e)&np.isfinite(c)
    if sigma is not None: m &= np.isfinite(sigma)
    e,c=e[m],c[m]
    if sigma is not None: sigma=sigma[m]
    o=np.argsort(e); e,c=e[o],c[o]
    if sigma is not None: sigma=sigma[o]
    if np.any(np.diff(e) <= 0): raise ValueError('Energy coordinates must be unique after sorting.')
    # Signed net bins are retained after background subtraction; do not clip
    # measured noise. Total area and variance are checked below.
    if bin_representation == "counts_per_bin":
        quad=np.ones_like(e)
    else:
        quad=np.empty_like(e)
        quad[0]=(e[1]-e[0])/2; quad[-1]=(e[-1]-e[-2])/2
        quad[1:-1]=(e[2:]-e[:-2])/2
    weighted=quad*c
    area=float(np.sum(weighted))
    base={'integrated_signal':area,
          'integrated_signal_unit':'counts' if bin_representation=="counts_per_bin" else 'signal·energy_unit',
          'bin_representation':bin_representation,'n_points':int(len(e)),
          'uncertainty_propagated':sigma is not None}
    if area<=0:
        return {**base,'moments_valid':False,'moment_status':'not_defined_nonpositive_net_area',
                'note':'Signed background-subtracted bins are retained without clipping. The net integrated signal is nonpositive, so centroid and width are not physically defined; inspect background subtraction and uncertainty.'}
    mean=float(np.sum(e*weighted)/area)
    var=float(np.sum((e-mean)**2*weighted)/area)
    if var < -1e-10 * max(float(np.max(e*e)), 1.0):
        return {**base,'moments_valid':False,'mean_energy_eV':mean,'rms_energy_spread_eV':None,
                'moment_status':'not_defined_negative_second_moment',
                'note':'Signed background-subtracted bins produce a negative second central moment. No bins were clipped; inspect background subtraction, bin uncertainties and fit a physically constrained spectral model if appropriate.'}
    spread=math.sqrt(max(var,0))
    result={'integrated_signal':area,
            'integrated_signal_unit':'counts' if bin_representation=="counts_per_bin" else 'signal·energy_unit',
            'bin_representation':bin_representation,'moments_valid':True,'moment_status':'valid_signed_net_moments',
            'mean_energy_eV':mean,'rms_energy_spread_eV':spread,
            'peak_energy_eV':float(e[np.argmax(c)]),'n_points':int(len(e)),
            'uncertainty_propagated':sigma is not None,
            'note':'Moments retain signed background-subtracted bins. Detector response, efficiency, pile-up and background covariance are not corrected automatically.'}
    if bin_representation == "density_per_eV":
        result['integrated_counts_eV']=area
    else:
        result['integrated_counts']=area
    if sigma is not None:
        dmean=quad*(e-mean)/area
        var_derivative=quad*((e-mean)**2-var)/area
        u_mean=float(np.sqrt(np.sum((dmean*sigma)**2)))
        u_spread=float(np.sqrt(np.sum((var_derivative*sigma)**2))/(2*spread)) if spread>0 else None
        result['standard_uncertainty_mean_energy_eV']=u_mean
        result['standard_uncertainty_rms_energy_spread_eV']=u_spread
        result['uncertainty_note']='First-order independent-bin uncertainty propagation; background covariance, detector response and model systematics are excluded unless included in supplied bin uncertainties.'
    return result


def neutron_rate_from_counts(counts_per_s: float, efficiency: float, geometric_factor: float = 1.0, calibration_factor: float = 1.0) -> dict[str, float]:
    vals=[counts_per_s, efficiency, geometric_factor, calibration_factor]
    if not np.all(np.isfinite(vals)) or counts_per_s<0 or not 0<efficiency<=1 or geometric_factor<=0 or calibration_factor<=0:
        raise ValueError('Counts must be finite and nonnegative; efficiency in (0,1], geometric and calibration factors finite and positive.')
    rate=float(counts_per_s/(efficiency*geometric_factor)*calibration_factor)
    return {'measured_count_rate_s-1':float(counts_per_s),'estimated_source_rate_s-1':rate,'efficiency':float(efficiency),'geometric_factor':float(geometric_factor),'calibration_factor':float(calibration_factor),'note':'Only valid when these factors represent the detector response and geometry for the specified neutron energy and configuration.'}


def equilibrium_grid_derivatives(R_m: Any, Z_m: Any, psi: Any, Bphi_T: float) -> dict[str, np.ndarray]:
    """Compute derivatives and Bp from a supplied poloidal-flux grid; does not solve equilibrium."""
    R=np.asarray(R_m,float).ravel(); Z=np.asarray(Z_m,float).ravel(); p=np.asarray(psi,float)
    if p.shape!=(len(Z),len(R)) or len(R)<3 or len(Z)<3: raise ValueError('psi must have shape (len(Z), len(R)) with at least 3 points per axis.')
    if np.any(np.diff(R)<=0) or np.any(np.diff(Z)<=0) or np.any(R<=0): raise ValueError('R and Z must be strictly increasing and R positive.')
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)) or not np.all(np.isfinite(p)):
        raise ValueError('R, Z and psi grid must contain only finite values.')
    if not np.isfinite(Bphi_T):
        raise ValueError('Toroidal magnetic field must be finite.')
    dpsi_dZ,dpsi_dR=np.gradient(p,Z,R,edge_order=2)
    RR=np.broadcast_to(R[None,:],p.shape)
    BR=-dpsi_dZ/RR; BZ=dpsi_dR/RR; BP=np.sqrt(BR**2+BZ**2); BT=np.full_like(BP,float(Bphi_T))
    return {'dpsi_dR':dpsi_dR,'dpsi_dZ':dpsi_dZ,'BR_T':BR,'BZ_T':BZ,'Bp_T':BP,'Bphi_T':BT,'Btotal_T':np.sqrt(BP**2+BT**2),'note':'Uses the supplied psi convention; verify whether psi is poloidal flux or flux divided by 2π and match units/sign conventions.'}


def xray_spectrum_summary(energy_keV: Any, counts: Any, background: Any | None = None,
                          count_uncertainty: Any | None = None,
                          assume_poisson_counts: bool = False,
                          bin_representation: str = "density_per_eV") -> dict[str, Any]:
    r=spectrum_moments(energy_keV, counts, background, count_uncertainty,
                       assume_poisson_counts, bin_representation)
    if 'peak_energy_eV' in r: r['peak_energy_keV']=r.pop('peak_energy_eV')
    if 'mean_energy_eV' in r: r['mean_energy_keV']=r.pop('mean_energy_eV')
    if 'rms_energy_spread_eV' in r: r['rms_energy_spread_keV']=r.pop('rms_energy_spread_eV')
    if 'standard_uncertainty_mean_energy_eV' in r: r['standard_uncertainty_mean_energy_keV']=r.pop('standard_uncertainty_mean_energy_eV')
    if 'standard_uncertainty_rms_energy_spread_eV' in r: r['standard_uncertainty_rms_energy_spread_keV']=r.pop('standard_uncertainty_rms_energy_spread_eV')
    if 'integrated_counts_eV' in r:
        r['integrated_signal_keV_weighted']=r.pop('integrated_counts_eV')
        # Retained for compatibility with existing callers; actual units follow
        # bin_representation and the supplied energy coordinate.
        r['integrated_counts_keV']=r['integrated_signal_keV_weighted']
    elif 'integrated_signal' in r:
        r['integrated_signal_keV_weighted']=r['integrated_signal']
    r['energy_axis_unit']='keV'
    r['note']='Energy axis is interpreted as keV. Select counts-per-bin or signal density explicitly; detector response and absolute calibration are not applied.'
    return r


def abel_reconstruction(chord_impact_m: Any, line_integrals: Any, radial_edges_m: Any,
                        lam: float = 1e-2, nonnegative: bool = True,
                        regularization_order: int = 1,
                        measurement_sigma: Any | None = None) -> dict[str, Any]:
    """Axisymmetric Abel inversion with chord-coverage and response-rank diagnostics."""
    b=np.asarray(chord_impact_m,float).ravel(); y=np.asarray(line_integrals,float).ravel(); edges=np.asarray(radial_edges_m,float).ravel()
    if len(b)!=len(y) or len(b)<3: raise ValueError('At least 3 paired chord impact parameters and line integrals are required.')
    if len(edges)<3 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges)<=0) or edges[0]<0:
        raise ValueError('Radial edges must be finite, increasing, nonnegative and contain at least 3 values.')
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(y)) or np.any(b<0): raise ValueError('Chord coordinates and line integrals must be finite; impact parameters nonnegative.')
    if np.any(b>=edges[-1]): raise ValueError('Every chord impact parameter must be smaller than the outer radial edge.')
    if len(np.unique(np.round(b,12)))<3: raise ValueError('At least 3 distinct chord impact parameters are required.')
    rin=edges[:-1][None,:]; rout=edges[1:][None,:]; bb=b[:,None]
    A=2*(np.sqrt(np.maximum(rout*rout-bb*bb,0))-np.sqrt(np.maximum(rin*rin-bb*bb,0)))
    A=np.where(bb>=rout,0,A)
    rank=int(np.linalg.matrix_rank(A))
    n_shells=len(edges)-1
    coverage_warning=None
    if rank<n_shells:
        coverage_warning=f"Response matrix rank {rank} is below the {n_shells} reconstructed shells; the profile is not uniquely constrained by these chords."
    elif len(np.unique(np.round(b,12)))<n_shells:
        coverage_warning=f"Only {len(np.unique(np.round(b,12)))} distinct chords constrain {n_shells} shells; inspect resolution and regularization."
    if float(np.max(b))<float(edges[-2]):
        coverage_warning=(coverage_warning+" " if coverage_warning else "")+"No chord is tangent within the outermost radial shell; outer-profile sensitivity may be weak."
    result=regularized_tomography(A,y,lam,nonnegative,measurement_sigma=measurement_sigma,
                                  regularization_order=regularization_order)
    centers=.5*(edges[:-1]+edges[1:])
    return {'radius_centers_m':centers,'emissivity_or_density_profile':result['solution'],'response_matrix':A,
            'residual_norm':result['residual_norm'],'success':result['success'],'message':result['message'],
            'lambda':float(lam),'regularization_order':int(regularization_order),
            'regularization_operator_norm':result['regularization_operator_norm'],'nonnegative':bool(nonnegative),
            'response_matrix_rank':rank,'number_of_radial_shells':int(n_shells),
            'number_of_distinct_chords':int(len(np.unique(np.round(b,12)))),
            'chord_coverage_warning':coverage_warning,
            'measurement_weighting':result['measurement_weighting'],
            'note':'Axisymmetry, straight chord geometry and the units of the supplied line integrals are assumed. Lambda is dimensionless relative to the normalized response matrix; default first-order regularization penalizes profile gradients, not absolute amplitude.'}


def axisymmetric_radiated_power(R_m: Any, Z_m: Any, emissivity_W_m3: Any) -> dict[str, float]:
    """Approximate total radiated power from a poloidal emissivity grid assuming toroidal symmetry."""
    R=np.asarray(R_m,float).ravel(); Z=np.asarray(Z_m,float).ravel(); eps=np.asarray(emissivity_W_m3,float)
    if eps.shape!=(len(Z),len(R)) or len(R)<2 or len(Z)<2: raise ValueError('Emissivity grid shape must be (len(Z),len(R)).')
    if np.any(np.diff(R)<=0) or np.any(np.diff(Z)<=0) or np.any(R<=0): raise ValueError('R and Z must be increasing and R positive.')
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)):
        raise ValueError('R and Z coordinates must contain only finite values.')
    if not np.all(np.isfinite(eps)): raise ValueError('Emissivity grid must be finite.')
    if np.any(eps < 0):
        raise ValueError('Emissivity must be nonnegative for a physical radiated-power integral; inspect background subtraction or inversion artifacts first.')
    # Integrate toroidal volume element 2*pi*R dR dZ over the provided rectangular grid.
    radial=integrate.trapezoid(2*np.pi*R[None,:]*eps,R,axis=1)
    power=float(integrate.trapezoid(radial,Z))
    return {'total_radiated_power_W':power,'emissivity_max_W_m-3':float(np.max(eps)),'emissivity_min_W_m-3':float(np.min(eps)),'assumption':'Axisymmetric emissivity and supplied grid covers the intended integration domain.'}


def profile_gradient_analysis(coordinate: Any, profile: Any) -> dict[str, Any]:
    """Compute calibrated profile gradients, interpolated half-height width and peak descriptors."""
    x=np.asarray(coordinate,float).ravel(); y=np.asarray(profile,float).ravel()
    m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<5: raise ValueError('At least 5 finite profile points are required.')
    o=np.argsort(x); x,y=x[o],y[o]
    if np.any(np.diff(x)<=0): raise ValueError('Profile coordinate must be unique and increasing.')
    grad=np.gradient(y,x,edge_order=2); peak_idx=int(np.argmax(y)); k=int(np.argmax(np.abs(grad)))
    result={'peak_value':float(np.max(y)),'minimum_value':float(np.min(y)),
            'peak_coordinate':float(x[peak_idx]),'profile_integral':float(integrate.trapezoid(y,x)),
            'max_abs_gradient':float(np.max(np.abs(grad))),'max_gradient_coordinate':float(x[k]),
            'gradient_at_max_abs':float(grad[k]),'n_points':int(len(x)),
            'gradient_method':'second-order finite differences on the supplied coordinate grid'}
    # FWHM is defined around the global peak and uses linear interpolation at crossings.
    if np.max(y)>0 and np.min(y)>=0:
        half=0.5*float(y[peak_idx])
        left_candidates=np.flatnonzero(y[:peak_idx] < half)
        right_candidates=np.flatnonzero(y[peak_idx+1:] < half)
        if len(left_candidates) and len(right_candidates):
            il=int(left_candidates[-1]); ir=int(peak_idx+1+right_candidates[0])
            left=float(x[il]+(half-y[il])*(x[il+1]-x[il])/(y[il+1]-y[il]))
            right=float(x[ir-1]+(half-y[ir-1])*(x[ir]-x[ir-1])/(y[ir]-y[ir-1]))
            result['FWHM_coordinate_units']=right-left
            result['FWHM_method']='linearly interpolated half-height crossings around global peak'
        else:
            result['FWHM_coordinate_units']=None
            result['FWHM_status']='profile does not cross half maximum on both sides; coordinate coverage is insufficient'
        peak_count=int(np.count_nonzero((y[1:-1]>y[:-2])&(y[1:-1]>=y[2:])))
        if peak_count>1:
            result['multiple_local_peaks_detected']=True
            result['FWHM_warning']='Multiple local peaks are present; the global-peak FWHM may not characterize the full profile.'
    result['pedestal_note']='No pedestal location/width is inferred automatically; pedestal characterization requires a chosen physical model or user-defined core/edge windows.'
    return result


def triple_probe_temperature_eV(v_difference_V: float, model_factor: float = 1.0) -> dict[str, float | str]:
    """First-order triple-probe Te estimate under a user-selected idealized exponential-current model."""
    if not np.isfinite(v_difference_V) or v_difference_V<=0 or not np.isfinite(model_factor) or model_factor<=0:
        raise ValueError('Voltage difference and model factor must be positive and finite.')
    te=float(v_difference_V/(np.log(2)*model_factor))
    return {'electron_temperature_estimate_eV':te,'voltage_difference_V':float(v_difference_V),'model_factor':float(model_factor),'assumption':'Idealized triple-probe relation; confirm the actual electrode configuration, current balance, sheath model and circuit definition before use.'}


def reflectometry_cutoff_density(frequency_hz: float, mode: str = 'O') -> dict[str, float | str]:
    """Cold-plasma cutoff density estimate for O/X mode at user-supplied probing frequency."""
    if not np.isfinite(frequency_hz) or frequency_hz<=0: raise ValueError('Frequency must be positive and finite.')
    mode=mode.upper()
    if mode not in {'O','X'}: raise ValueError("Mode must be 'O' or 'X'.")
    # O-mode: omega_pe = omega. X-mode requires B and has multiple cutoffs, so do not fake it.
    if mode=='X': raise ValueError('X-mode cutoff depends on magnetic field and branch; use the O-mode formula only or supply a validated X-mode model.')
    n=float(EPS0*M_E*(2*np.pi*frequency_hz)**2/E_CHARGE**2)
    return {'frequency_Hz':float(frequency_hz),'mode':'O','cutoff_density_m-3':n,'formula':'n_cut = ε0 m_e (2π f)^2 / e²','assumption':'Cold, collisionless O-mode cutoff; actual reflectometry inversion also requires propagation and cutoff-layer interpretation.'}


def electron_cyclotron_resonance_field(frequency_hz: float, harmonic: int = 1) -> dict[str, float | int | str]:
    if not np.isfinite(frequency_hz) or frequency_hz<=0 or not np.isfinite(harmonic) or float(harmonic) != int(harmonic) or int(harmonic)<1:
        raise ValueError('Frequency must be positive and harmonic a positive integer.')
    B=float(2*np.pi*frequency_hz*M_E/(E_CHARGE*int(harmonic)))
    return {'frequency_Hz':float(frequency_hz),'harmonic':int(harmonic),'resonance_field_T':B,'formula':'ω = h e B / m_e','assumption':'Relativistic and Doppler shifts are neglected; resonance geometry must be considered for actual ECE interpretation.'}


def corrected_line_ratio(intensity_1: float, intensity_2: float, response_1: float = 1.0, response_2: float = 1.0) -> dict[str, float | str]:
    vals=[intensity_1,intensity_2,response_1,response_2]
    if not all(np.isfinite(v) for v in vals) or intensity_1<0 or intensity_2<=0 or response_1<=0 or response_2<=0: raise ValueError('Intensities must be nonnegative (line 2 positive) and instrument responses positive and finite.')
    ratio=float((intensity_1/response_1)/(intensity_2/response_2))
    return {'raw_intensity_ratio':float(intensity_1/intensity_2),'instrument_response_corrected_ratio':ratio,'response_1':float(response_1),'response_2':float(response_2),'note':'A physical Te/ne/impurity inference requires validated atomic emissivity coefficients or a collisional-radiative model; this routine only corrects the measured ratio for supplied responses.'}


def grid_from_table(frame: pd.DataFrame, z_column: str = "z_m") -> dict[str, np.ndarray]:
    """Parse a wide grid table (one z column + one column per R value) into R, Z and a (nZ, nR) array.

    Accepts the format TIS itself exports for FAIR-MAST EFIT grids: columns named 'R=0.123_m' (or plain
    numbers). The function only reshapes; it makes no claim about what the grid physically represents.
    """
    if z_column not in frame.columns:
        raise ValueError(f"Column {z_column!r} (vertical coordinate) not found.")
    r_cols = [c for c in frame.columns if c != z_column]
    if len(r_cols) < 3 or len(frame) < 3:
        raise ValueError("A grid needs at least 3 R columns and 3 Z rows.")
    r_vals = []
    for c in r_cols:
        text = str(c).strip()
        if text.upper().startswith("R="):
            text = text[2:]
        if text.lower().endswith("_m"):
            text = text[:-2]
        try:
            r_vals.append(float(text))
        except ValueError as exc:
            raise ValueError(f"Cannot parse an R coordinate from column name {c!r}.") from exc
    z = pd.to_numeric(frame[z_column], errors="coerce").to_numpy(dtype=float)
    grid = frame[r_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    r = np.asarray(r_vals, dtype=float)
    order_r = np.argsort(r); order_z = np.argsort(z)
    return {"R_m": r[order_r], "Z_m": z[order_z], "grid": grid[np.ix_(order_z, order_r)]}

