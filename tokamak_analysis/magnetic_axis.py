"""Magnetic-axis estimate from a gridded poloidal-flux map."""
import numpy as np
def locate_magnetic_axis(R_m,Z_m,psi_Wb_rad,*,axis_type="minimum"):
    R=np.asarray(R_m,dtype=float); Z=np.asarray(Z_m,dtype=float); p=np.asarray(psi_Wb_rad,dtype=float)
    if R.ndim!=1 or Z.ndim!=1 or len(R)<2 or len(Z)<2 or p.shape!=(len(Z),len(R)) or not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)) or not np.all(np.isfinite(p)): raise ValueError("Flux map and 1-D R,Z grids must be finite and shape-compatible")
    if np.any(np.diff(R)<=0) or np.any(np.diff(Z)<=0): raise ValueError("R and Z must be strictly increasing")
    if axis_type not in ("minimum","maximum"): raise ValueError("axis_type must be minimum or maximum")
    j,i=np.unravel_index(np.argmin(p) if axis_type=="minimum" else np.argmax(p),p.shape)
    return {"R_axis_m":float(R[i]),"Z_axis_m":float(Z[j]),"psi_axis_Wb_rad":float(p[j,i]),"grid_index":(int(j),int(i)),"method":"grid extremum","caveat":"Grid extremum is a first estimate; refine with local fitting and equilibrium constraints."}

def quadratic_magnetic_axis_refinement(R_m, Z_m, psi_Wb_rad, *, axis_type="minimum"):
    """Refine a gridded flux extremum by fitting a local 2-D quadratic surface."""
    R=np.asarray(R_m,dtype=float); Z=np.asarray(Z_m,dtype=float); p=np.asarray(psi_Wb_rad,dtype=float)
    if R.ndim!=1 or Z.ndim!=1 or len(R)<3 or len(Z)<3 or p.shape!=(len(Z),len(R)):
        raise ValueError("Need 1-D R,Z grids of length >=3 and matching flux map")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)) or not np.all(np.isfinite(p)) or np.any(np.diff(R)<=0) or np.any(np.diff(Z)<=0):
        raise ValueError("Grids and flux must be finite and strictly increasing")
    if axis_type not in ("minimum","maximum"): raise ValueError("axis_type must be minimum or maximum")
    j,i=np.unravel_index(np.argmin(p) if axis_type=="minimum" else np.argmax(p),p.shape)
    if i==0 or i==len(R)-1 or j==0 or j==len(Z)-1: raise ValueError("Flux extremum lies on boundary; no centered 3x3 fit possible")
    rr,zz=np.meshgrid(R[i-1:i+2],Z[j-1:j+2]); vals=p[j-1:j+2,i-1:i+2].ravel()
    design=np.column_stack([rr.ravel()**2,rr.ravel()*zz.ravel(),zz.ravel()**2,rr.ravel(),zz.ravel(),np.ones(9)])
    c=np.linalg.lstsq(design,vals,rcond=None)[0]
    h=np.array([[2*c[0],c[1]],[c[1],2*c[2]]]); g=np.array([c[3],c[4]])
    eig=np.linalg.eigvalsh(h)
    if axis_type=="minimum" and np.any(eig<=0) or axis_type=="maximum" and np.any(eig>=0):
        raise ValueError("Local quadratic fit does not have the requested extremum curvature")
    axis=np.linalg.solve(h,-g); ra,za=map(float,axis)
    if not (R[i-1]<=ra<=R[i+1] and Z[j-1]<=za<=Z[j+1]): raise ValueError("Refined axis lies outside the local fitting cell")
    psi=float(c[0]*ra**2+c[1]*ra*za+c[2]*za**2+c[3]*ra+c[4]*za+c[5])
    return {"R_axis_m":ra,"Z_axis_m":za,"psi_axis_Wb_rad":psi,"method":"local 3x3 quadratic least-squares fit","hessian_eigenvalues":eig}
