"""Spectral linewidth decomposition for Gaussian broadening contributions."""
import numpy as np
def decompose_gaussian_linewidth(observed_fwhm_nm, *, instrumental_fwhm_nm=0.0, natural_fwhm_nm=0.0, zeeman_fwhm_equivalent_nm=0.0):
    obs,inst,nat,zee=map(float,(observed_fwhm_nm,instrumental_fwhm_nm,natural_fwhm_nm,zeeman_fwhm_equivalent_nm))
    if not all(np.isfinite(x) for x in (obs,inst,nat,zee)) or obs<=0 or min(inst,nat,zee)<0: raise ValueError("Observed width positive; contributions finite and non-negative")
    residual=obs**2-inst**2-nat**2-zee**2
    if residual<0: raise ValueError("Known broadening contributions exceed observed width")
    return {"observed_fwhm_nm":obs,"instrumental_fwhm_nm":inst,"natural_fwhm_nm":nat,"zeeman_equivalent_fwhm_nm":zee,"residual_gaussian_fwhm_nm":float(np.sqrt(residual)),"caveat":"Quadrature subtraction is valid for independent Gaussian widths; Lorentzian/Voigt contributions need profile-specific fitting."}