"""X-point candidate detection from saddle points in poloidal flux."""
import numpy as np
def find_x_point_candidates(R_m,Z_m,psi_Wb_rad,*,gradient_tolerance=None):
    R=np.asarray(R_m,dtype=float); Z=np.asarray(Z_m,dtype=float); p=np.asarray(psi_Wb_rad,dtype=float)
    if R.ndim!=1 or Z.ndim!=1 or p.shape!=(len(Z),len(R)) or len(R)<3 or len(Z)<3: raise ValueError("Invalid R-Z flux grid")
    if not np.all(np.isfinite(p)) or np.any(np.diff(R)<=0) or np.any(np.diff(Z)<=0): raise ValueError("Flux and grids must be finite and increasing")
    dz,dr=np.gradient(p,Z,R,edge_order=2); candidates=[]
    for j in range(1,len(Z)-1):
        for i in range(1,len(R)-1):
            if dr[j-1:j+2,i-1:i+2].min()<=0<=dr[j-1:j+2,i-1:i+2].max() and dz[j-1:j+2,i-1:i+2].min()<=0<=dz[j-1:j+2,i-1:i+2].max():
                h=np.gradient(np.gradient(p,Z,axis=0),Z,axis=0)[j,i]*np.gradient(np.gradient(p,R,axis=1),R,axis=1)[j,i]-np.gradient(np.gradient(p,R,axis=1),Z,axis=0)[j,i]**2
                if h<0: candidates.append({"R_m":float(R[i]),"Z_m":float(Z[j]),"psi_Wb_rad":float(p[j,i]),"hessian_determinant":float(h)})
    return {"candidates":candidates,"count":len(candidates),"caveat":"Grid-based saddle candidates only; separatrix topology and equilibrium boundary conditions must be checked."}