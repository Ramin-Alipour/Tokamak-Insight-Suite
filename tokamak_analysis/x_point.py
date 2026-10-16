"""X-point candidate detection from saddle points in poloidal flux."""
import numpy as np
def find_x_point_candidates(R_m,Z_m,psi_Wb_rad,*,gradient_tolerance=None):
    R=np.asarray(R_m,dtype=float); Z=np.asarray(Z_m,dtype=float); p=np.asarray(psi_Wb_rad,dtype=float)
    if R.ndim!=1 or Z.ndim!=1 or p.shape!=(len(Z),len(R)) or len(R)<3 or len(Z)<3: raise ValueError("Invalid R-Z flux grid")
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)) or not np.all(np.isfinite(p)) or np.any(R<=0) or np.any(np.diff(R)<=0) or np.any(np.diff(Z)<=0): raise ValueError("Flux and grids must be finite and strictly increasing; major radius R must be positive")
    if gradient_tolerance is not None:
        gradient_tolerance=float(gradient_tolerance)
        if not np.isfinite(gradient_tolerance) or gradient_tolerance<0: raise ValueError("gradient_tolerance must be finite and non-negative")
    dz,dr=np.gradient(p,Z,R,edge_order=2); candidates=[]
    for j in range(1,len(Z)-1):
        for i in range(1,len(R)-1):
            if dr[j-1:j+2,i-1:i+2].min()<=0<=dr[j-1:j+2,i-1:i+2].max() and dz[j-1:j+2,i-1:i+2].min()<=0<=dz[j-1:j+2,i-1:i+2].max():
                d2psi_dz2=np.gradient(np.gradient(p,Z,axis=0,edge_order=2),Z,axis=0,edge_order=2)[j,i]
                d2psi_dr2=np.gradient(np.gradient(p,R,axis=1,edge_order=2),R,axis=1,edge_order=2)[j,i]
                d2psi_drdz=np.gradient(np.gradient(p,R,axis=1,edge_order=2),Z,axis=0,edge_order=2)[j,i]
                h=d2psi_dz2*d2psi_dr2-d2psi_drdz**2
                grad_norm=float(np.hypot(dr[j,i],dz[j,i]))
                if h<0 and (gradient_tolerance is None or grad_norm<=gradient_tolerance): candidates.append({"R_m":float(R[i]),"Z_m":float(Z[j]),"psi_Wb_rad":float(p[j,i]),"hessian_determinant":float(h),"gradient_norm":grad_norm})
    return {"candidates":candidates,"count":len(candidates),"caveat":"Grid-based saddle candidates only; separatrix topology and equilibrium boundary conditions must be checked."}

def classify_flux_critical_point(d2psi_dR2, d2psi_dZ2, d2psi_dRdZ):
    """Classify a 2-D flux critical point by Hessian determinant D=psi_RR*psi_ZZ-psi_RZ^2."""
    rr,zz,rz=map(float,(d2psi_dR2,d2psi_dZ2,d2psi_dRdZ))
    if not np.all(np.isfinite([rr,zz,rz])): raise ValueError("Hessian components must be finite")
    determinant=rr*zz-rz**2
    trace=rr+zz
    if determinant<0: kind="saddle (X-point candidate)"
    elif determinant>0 and trace>0: kind="local minimum"
    elif determinant>0 and trace<0: kind="local maximum"
    else: kind="degenerate or inconclusive"
    return {"hessian_determinant":determinant,"hessian_trace":trace,"classification":kind,
            "equation":"D=psi_RR*psi_ZZ-(psi_RZ)^2; D<0 implies saddle"}
