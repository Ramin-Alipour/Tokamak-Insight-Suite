"""Plasma boundary and shape metrics from a measured or reconstructed R-Z flux map."""
import numpy as np

def analyze_plasma_shape(R_m, Z_m, psi_normalized, *, boundary_level=0.95):
    """Estimate LCFS-like contour geometry from normalized poloidal flux.

    The contour is extracted from gridded psi_N using matplotlib. Metrics are
    geometric only; users must choose a physically appropriate boundary level.
    """
    R=np.asarray(R_m,dtype=float); Z=np.asarray(Z_m,dtype=float); psi=np.asarray(psi_normalized,dtype=float)
    if R.ndim!=1 or Z.ndim!=1 or len(R)<3 or len(Z)<3 or np.any(np.diff(R)<=0) or np.any(np.diff(Z)<=0): raise ValueError("R and Z must be increasing 1-D grids")
    if psi.shape!=(len(Z),len(R)) or not np.all(np.isfinite(psi)): raise ValueError("psi_normalized must be finite with shape (len(Z), len(R))")
    if not 0<boundary_level<1: raise ValueError("boundary_level must lie between 0 and 1")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(); cs=ax.contour(R,Z,psi,levels=[boundary_level]);
    paths=[np.asarray(seg,dtype=float) for seg in cs.allsegs[0] if len(seg)>=5]
    plt.close(fig)
    if not paths: raise ValueError("No closed/usable boundary contour found at requested level")
    contour=max(paths,key=lambda v: len(v))
    rr=contour[:,0]; zz=contour[:,1]
    if not np.allclose(contour[0],contour[-1]): contour=np.vstack([contour,contour[0]])
    rr=contour[:,0]; zz=contour[:,1]
    area=0.5*abs(np.sum(rr[:-1]*zz[1:]-rr[1:]*zz[:-1]))
    Rgeo=0.5*(float(np.max(rr))+float(np.min(rr))); a=0.5*(float(np.max(rr))-float(np.min(rr)))
    ztop=float(np.max(zz)); zbottom=float(np.min(zz)); zaxis=0.5*(ztop+zbottom); amin=0.5*(ztop-zbottom)
    delta_top=(Rgeo-float(rr[np.argmax(zz)]))/a if a>0 else np.nan
    delta_bottom=(Rgeo-float(rr[np.argmin(zz)]))/a if a>0 else np.nan
    return {"boundary_level_psi_N":float(boundary_level),"contour_R_m":rr,"contour_Z_m":zz,"geometric_major_radius_m":Rgeo,"minor_radius_horizontal_m":a,"vertical_half_height_m":amin,"elongation_estimate":amin/a if a>0 else np.nan,"triangularity_upper_estimate":delta_top,"triangularity_lower_estimate":delta_bottom,"cross_section_area_m2":area,"contour_point_count":int(len(rr)),"caveat":"Contour-derived geometry depends on the selected flux level and grid resolution; this does not determine a physical LCFS without equilibrium interpretation."}

def miller_boundary(R0_m, minor_radius_m, elongation, triangularity, theta_rad):
    """Miller parameterization of an axisymmetric tokamak plasma boundary.

    R = R0 + a cos(theta + asin(delta) sin(theta)); Z = kappa a sin(theta).
    This supplies a physical parameterized boundary, not an equilibrium solver.
    """
    R0=float(R0_m); a=float(minor_radius_m); kappa=float(elongation); delta=float(triangularity)
    theta=np.asarray(theta_rad,dtype=float)
    if not np.isfinite(R0) or not np.isfinite(a) or not np.isfinite(kappa) or not np.isfinite(delta):
        raise ValueError("Boundary parameters must be finite")
    if R0<=0 or a<=0 or kappa<=0 or not -1<delta<1:
        raise ValueError("Require R0>0, a>0, elongation>0, and -1<triangularity<1")
    if theta.ndim!=1 or len(theta)<3 or not np.all(np.isfinite(theta)):
        raise ValueError("theta_rad must be a finite 1-D array with at least 3 points")
    phase=theta+np.arcsin(delta)*np.sin(theta)
    R=R0+a*np.cos(phase)
    Z=kappa*a*np.sin(theta)
    area=0.5*abs(float(np.sum(R[:-1]*Z[1:]-R[1:]*Z[:-1])))
    return {"R_m":R,"Z_m":Z,"cross_section_area_m2":area,
            "minor_radius_m":a,"elongation":kappa,"triangularity":delta,
            "equation":"R=R0+a*cos(theta+asin(delta)*sin(theta)); Z=kappa*a*sin(theta)",
            "area_analytic_for_delta_zero_m2":float(np.pi*kappa*a*a) if delta==0 else None}
