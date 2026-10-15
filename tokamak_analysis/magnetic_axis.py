"""Magnetic-axis estimate from a gridded poloidal-flux map."""
import numpy as np
def locate_magnetic_axis(R_m,Z_m,psi_Wb_rad,*,axis_type="minimum"):
    R=np.asarray(R_m,dtype=float); Z=np.asarray(Z_m,dtype=float); p=np.asarray(psi_Wb_rad,dtype=float)
    if R.ndim!=1 or Z.ndim!=1 or p.shape!=(len(Z),len(R)) or not np.all(np.isfinite(p)): raise ValueError("Flux map must match finite 1-D R,Z grids")
    if np.any(np.diff(R)<=0) or np.any(np.diff(Z)<=0): raise ValueError("R and Z must be strictly increasing")
    if axis_type not in ("minimum","maximum"): raise ValueError("axis_type must be minimum or maximum")
    j,i=np.unravel_index(np.argmin(p) if axis_type=="minimum" else np.argmax(p),p.shape)
    return {"R_axis_m":float(R[i]),"Z_axis_m":float(Z[j]),"psi_axis_Wb_rad":float(p[j,i]),"grid_index":(int(j),int(i)),"method":"grid extremum","caveat":"Grid extremum is a first estimate; refine with local fitting and equilibrium constraints."}