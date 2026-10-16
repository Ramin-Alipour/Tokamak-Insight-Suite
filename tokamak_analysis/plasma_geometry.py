"""Plasma volume and cross-sectional geometry from a boundary contour."""
import numpy as np
def geometry_from_boundary(R_boundary_m,Z_boundary_m):
    R=np.asarray(R_boundary_m,dtype=float); Z=np.asarray(Z_boundary_m,dtype=float)
    if R.ndim!=1 or Z.ndim!=1 or R.size!=Z.size or R.size<4 or not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)): raise ValueError("Boundary coordinates must be finite equal-length 1-D arrays")
    if np.any(R<=0): raise ValueError("Tokamak boundary must remain at positive major radius and cannot cross the symmetry axis")
    if not np.allclose([R[0],Z[0]],[R[-1],Z[-1]]): R=np.r_[R,R[0]]; Z=np.r_[Z,Z[0]]
    signed=.5*np.sum(R[:-1]*Z[1:]-R[1:]*Z[:-1]); area=abs(float(signed))
    rcent=float(np.sum((R[:-1]+R[1:])*(R[:-1]*Z[1:]-R[1:]*Z[:-1]))/(6*signed)) if signed else float("nan")
    major=float(np.sum((R[:-1]+R[1:])*(R[:-1]*Z[1:]-R[1:]*Z[:-1]))/(6*signed)) if signed else float("nan")
    volume=2*np.pi*major*area if np.isfinite(major) else float("nan")
    return {"cross_section_area_m2":area,"geometric_major_radius_m":major,"toroidal_volume_m3":volume,"R_boundary_m":R,"Z_boundary_m":Z,"caveat":"Volume uses axisymmetric Pappus theorem for a closed contour not crossing the symmetry axis."}

def cross_section_area_from_boundary(R_boundary_m,Z_boundary_m):
    """Shoelace line-integral relation A=1/2*|integral(R dZ-Z dR)| for a closed contour."""
    R=np.asarray(R_boundary_m,dtype=float); Z=np.asarray(Z_boundary_m,dtype=float)
    if R.ndim!=1 or Z.shape!=R.shape or len(R)<3 or not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)):
        raise ValueError("Boundary coordinates must be finite equal-length 1-D arrays with at least 3 points")
    if not np.allclose([R[0],Z[0]],[R[-1],Z[-1]]): R=np.r_[R,R[0]]; Z=np.r_[Z,Z[0]]
    area=0.5*abs(float(np.sum(R[:-1]*Z[1:]-R[1:]*Z[:-1])))
    return {"cross_section_area_m2":area,"equation":"A=0.5*abs(sum(R_i*Z_(i+1)-R_(i+1)*Z_i))"}


def axisymmetric_volume_from_boundary(R_boundary_m,Z_boundary_m):
    """Pappus theorem: V=2*pi*R_centroid*A for revolution about the symmetry axis."""
    out=geometry_from_boundary(R_boundary_m,Z_boundary_m)
    if not np.isfinite(out["geometric_major_radius_m"]) or out["geometric_major_radius_m"]<=0:
        raise ValueError("Boundary must enclose a valid cross-section with positive centroid radius")
    return {"cross_section_area_m2":out["cross_section_area_m2"],
            "centroid_major_radius_m":out["geometric_major_radius_m"],
            "toroidal_volume_m3":2*np.pi*out["geometric_major_radius_m"]*out["cross_section_area_m2"],
            "equation":"V=2*pi*R_centroid*A (Pappus centroid theorem)"}
