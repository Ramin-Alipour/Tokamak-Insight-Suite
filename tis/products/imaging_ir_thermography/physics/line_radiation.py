"""Line-radiation emissivity and radiated-power integration."""
import numpy as np
def analyze_line_radiation(wavelength_nm, calibrated_radiance, *, integration_range_nm=None, solid_angle_sr=None, area_m2=None):
    x=np.asarray(wavelength_nm,dtype=float); y=np.asarray(calibrated_radiance,dtype=float)
    if x.ndim!=1 or y.shape!=x.shape or len(x)<2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(np.diff(x)<=0): raise ValueError("Wavelength must increase and radiance match")
    m=np.ones_like(x,dtype=bool)
    if integration_range_nm is not None:
        lo,hi=map(float,integration_range_nm)
        if not np.all(np.isfinite([lo,hi])) or hi<=lo: raise ValueError("Wavelength range must have finite increasing bounds")
        m=(x>=lo)&(x<=hi)
        if np.count_nonzero(m)<2: raise ValueError("Fewer than two samples in selected range")
    integral=float((getattr(np, "trapezoid", None) or getattr(np, "trapz"))(y[m],x[m]))
    out={"integrated_spectral_radiance":integral,"wavelength_range_nm":[float(x[m][0]),float(x[m][-1])],"peak_radiance":float(np.max(y[m]))}
    if solid_angle_sr is not None and area_m2 is not None:
        om=float(solid_angle_sr); area=float(area_m2)
        if not np.isfinite(om) or not np.isfinite(area) or om<=0 or area<=0: raise ValueError("Solid angle and area must be finite and positive")
        out["estimated_radiant_power_W"]=integral*om*area
    out["caveat"]="Radiant power estimate requires consistent spectral units and calibrated collection geometry; line-of-sight integration is not local emissivity."
    return out

def photon_energy_from_wavelength_eV(wavelength_nm):
    """Photon energy-wavelength relation E_gamma=hc/lambda."""
    lam=np.asarray(wavelength_nm,dtype=float)
    if not np.all(np.isfinite(lam)) or np.any(lam<=0): raise ValueError("Wavelength must be finite and positive")
    H=6.62607015e-34; C=299792458.0; E=1.602176634e-19
    return {"photon_energy_eV":H*C/(lam*1e-9)/E,"equation":"E_gamma[eV]=h*c/(lambda[m]*e)"}
