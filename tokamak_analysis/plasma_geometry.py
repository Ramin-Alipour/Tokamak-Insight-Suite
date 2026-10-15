"""Plasma volume and cross-sectional geometry from a boundary contour."""
import numpy as np
def geometry_from_boundary(R_boundary_m,Z_boundary_m):
    R=np.asarray(R_boundary_m,dtype=float); Z=np.asarray(Z_boundary_m,dtype=float)
    if R.ndim!=1 or Z.ndim!=1 or R.size!=Z.size or R.size<4 or not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)): raise ValueError("Boundary coordinates must be finite equal-length 1-D arrays")
    if not np.allclose([R[0],Z[0]],[R[-1],Z[-1]]): R=np.r_[R,R[0]]; Z=np.r_[Z,Z[0]]
    signed=.5*np.sum(R[:-1]*Z[1:]-R[1:]*Z[:-1]); area=abs(float(signed))
    rcent=float(np.sum((R[:-1]+R[1:])*(R[:-1]*Z[1:]-R[1:]*Z[:-1]))/(6*signed)) if signed else float("nan")
    major=float(np.sum((R[:-1]+R[1:])*(R[:-1]*Z[1:]-R[1:]*Z[:-1]))/(6*signed)) if signed else float("nan")
    volume=2*np.pi*major*area if np.isfinite(major) else float("nan")
    return {"cross_section_area_m2":area,"geometric_major_radius_m":major,"toroidal_volume_m3":volume,"R_boundary_m":R,"Z_boundary_m":Z,"caveat":"Volume uses axisymmetric Pappus theorem for a closed contour not crossing the symmetry axis."}