"""Reflectometry cutoff density and frequency-spectrum summaries."""
import numpy as np
EPS0=8.8541878128e-12; ME=9.1093837139e-31; E=1.602176634e-19
def cutoff_density(frequency_Hz, mode="O"):
    f=np.asarray(frequency_Hz,dtype=float)
    if not np.all(np.isfinite(f)) or np.any(f<=0): raise ValueError("Frequency must be finite and positive")
    if mode.upper()=="O": n=EPS0*ME*(2*np.pi*f)**2/E**2
    elif mode.upper()=="X":
        raise ValueError("X-mode cutoff requires the magnetic field and the selected cutoff branch; this function currently supports O-mode only.")
    else: raise ValueError("mode must be O")
    return {"cutoff_density_m-3":n,"mode":"O","equation":"n_cut = epsilon_0*m_e*(2*pi*f)^2/e^2","caveat":"Cold-plasma O-mode cutoff; propagation, relativistic and magnetic effects may require a more complete model."}
def analyze_reflectometry_spectrum(frequency_Hz, amplitude):
    f=np.asarray(frequency_Hz,dtype=float); a=np.asarray(amplitude,dtype=float)
    if f.ndim!=1 or a.shape!=f.shape or len(f)<3 or not np.all(np.isfinite(f)) or not np.all(np.isfinite(a)) or np.any(f<0) or np.any(np.diff(f)<=0): raise ValueError("Finite, non-negative increasing frequency and matching amplitude required")
    idx=int(np.argmax(np.abs(a))); peak=float(np.max(np.abs(a)))
    return {"peak_frequency_Hz":float(f[idx]) if peak>0 else None,"peak_amplitude":float(a[idx]) if peak>0 else 0.0,"frequency_Hz":f,"amplitude":a,"caveat":"Cutoff interpretation and density-profile inversion require a propagation model and instrument geometry; zero-amplitude spectra have no defined peak."}

def density_from_o_mode_refractive_index(frequency_Hz, refractive_index):
    """Invert cold-plasma O-mode dispersion N^2=1-(f_pe/f)^2 for electron density."""
    f,n=np.broadcast_arrays(np.asarray(frequency_Hz,dtype=float),np.asarray(refractive_index,dtype=float))
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(n)) or np.any(f<=0) or np.any(n<0) or np.any(n>1):
        raise ValueError("Frequency must be positive and O-mode refractive index must lie in [0,1]")
    density=EPS0*ME*(2*np.pi*f)**2/E**2*(1-n**2)
    return {"electron_density_m3":density,
            "equation":"N_O^2=1-(f_pe/f)^2; ne=epsilon0*me*(2*pi*f)^2/e^2*(1-N_O^2)",
            "assumption":"Cold, unmagnetized O-mode propagation; ray path and density-gradient effects are not inverted here."}
