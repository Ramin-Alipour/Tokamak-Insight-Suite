"""Electrostatic probe I-V analysis and temperature estimates."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from .constants import E_CHARGE, AMU
from .spectral_analysis import _r2

def probe_iv_summary(voltage: Any, current: Any) -> dict[str, Any]:
    """Descriptive I-V preprocessing only; does not silently infer Te/ne without a validated regime model."""
    v=np.asarray(voltage,float).ravel(); i=np.asarray(current,float).ravel()
    if len(v) != len(i): raise ValueError('Voltage and current arrays must have equal lengths.')
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
            try:
                limits = np.asarray(fit_range, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain exactly two finite voltage limits.") from exc
            if limits.shape != (2,) or not np.all(np.isfinite(limits)):
                raise ValueError(f"{name} must contain exactly two finite voltage limits.")
    if (area_m2 is None) != (ion_mass_amu is None):
        raise ValueError("area_m2 and ion_mass_amu must be supplied together for density estimation.")
    if area_m2 is not None and (not np.isfinite(area_m2) or area_m2 <= 0):
        raise ValueError("Collection area must be positive and finite.")
    if ion_mass_amu is not None and (not np.isfinite(ion_mass_amu) or ion_mass_amu <= 0):
        raise ValueError("Ion mass must be positive and finite.")
    if not np.all(np.isfinite([ti_eV, sheath_factor, gamma_i])) or ti_eV < 0 or sheath_factor <= 0 or gamma_i < 0:
        raise ValueError("ti_eV and gamma_i must be finite and nonnegative; sheath_factor must be positive and finite.")
    raw_v=np.asarray(voltage,float); raw_i=np.asarray(current,float)
    if raw_v.ndim != 1 or raw_i.ndim != 1:
        raise ValueError("Voltage and current inputs must be one-dimensional arrays.")
    v=raw_v.copy(); i=raw_i.copy()
    if len(v) != len(i): raise ValueError('Voltage and current arrays must have equal lengths.')
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
    # Always provide explicitly labelled descriptive saturation-tail estimates.
    # These are fallbacks only; a user-supplied ion_fit_range overrides the ion tail.
    # Tail size is bounded to avoid letting a few endpoint samples dominate.
    tail_n = max(3, int(math.ceil(0.1 * len(v))))
    out["ion_saturation_current_A"] = float(np.median(i[:tail_n]))
    out["ion_saturation_current_method"] = "median_lowest_voltage_tail_fallback"
    out["electron_saturation_current_A"] = float(np.median(i[-tail_n:]))
    out["electron_saturation_current_method"] = "median_highest_voltage_tail_descriptive"
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
        if max(ion_fit_range) < min(v) or min(ion_fit_range) > max(v):
            raise ValueError("ion_fit_range must overlap the measured voltage sweep.")
        m=(v>=min(ion_fit_range))&(v<=max(ion_fit_range))
        if m.sum()>=3:
            p=np.polyfit(v[m],i[m],1); out['ion_saturation_linear_slope_A_per_V']=float(p[0]); out['ion_saturation_intercept_A']=float(p[1]); out['ion_saturation_fit_R2']=_r2(i[m],np.polyval(p,v[m]))
            out['ion_saturation_current_A']=float(np.median(i[m])); out['ion_saturation_current_method']='median_user_selected_ion_fit_range'
        else:
            raise ValueError("ion_fit_range must contain at least 3 valid samples.")
    if electron_fit_range is not None:
        if max(electron_fit_range) <= min(v) or min(electron_fit_range) >= max(v):
            raise ValueError("electron_fit_range must overlap the measured voltage sweep.")
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
    if area_m2 is not None and ion_fit_range is None:
        out['ion_density_status'] = "not_estimated: provide a physically selected ion_fit_range containing at least three points; tail fallback is descriptive only"
    if area_m2 is not None and ion_fit_range is not None and 'ion_saturation_current_A' in out:
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
    v=np.asarray(voltage,float).ravel(); i=np.asarray(current,float).ravel()
    if len(v) != len(i): raise ValueError('Voltage and current arrays must have equal lengths.')
    m=np.isfinite(v)&np.isfinite(i); v,i=v[m],i[m]
    if len(v)<2: return None
    o=np.argsort(v); v,i=v[o],i[o]; ix=np.where(i[:-1]*i[1:]<=0)[0]
    if not len(ix): return None
    k=ix[0]
    if i[k+1]==i[k]: return float((v[k]+v[k+1])/2)
    return float(v[k]-i[k]*(v[k+1]-v[k])/(i[k+1]-i[k]))

def triple_probe_temperature_eV(v_difference_V: float, model_factor: float = 1.0) -> dict[str, float | str]:
    """First-order triple-probe Te estimate under a user-selected idealized exponential-current model."""
    if not np.isfinite(v_difference_V) or v_difference_V<=0 or not np.isfinite(model_factor) or model_factor<=0:
        raise ValueError('Voltage difference and model factor must be positive and finite.')
    te=float(v_difference_V/(np.log(2)*model_factor))
    return {'electron_temperature_estimate_eV':te,'voltage_difference_V':float(v_difference_V),'model_factor':float(model_factor),'assumption':'Idealized triple-probe relation; confirm the actual electrode configuration, current balance, sheath model and circuit definition before use.'}
