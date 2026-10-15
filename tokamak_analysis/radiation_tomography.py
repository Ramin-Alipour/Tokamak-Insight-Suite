"""Regularized linear reconstruction of emissivity from line-integrated signals."""
import numpy as np
def reconstruct_emissivity(response_matrix, line_integrals, *, regularization=1e-6, nonnegative=True):
    A=np.asarray(response_matrix,dtype=float); y=np.asarray(line_integrals,dtype=float); lam=float(regularization)
    if A.ndim!=2 or y.ndim!=1 or A.shape[0]!=len(y) or min(A.shape)<1 or not np.all(np.isfinite(A)) or not np.all(np.isfinite(y)): raise ValueError("Finite response matrix and matching measurements required")
    if not np.isfinite(lam) or lam<0: raise ValueError("Regularization must be nonnegative")
    lhs=A.T@A+lam*np.eye(A.shape[1]); rhs=A.T@y
    if nonnegative:
        from scipy.optimize import lsq_linear
        augA=np.vstack([A,np.sqrt(lam)*np.eye(A.shape[1])]) if lam>0 else A
        augy=np.r_[y,np.zeros(A.shape[1])] if lam>0 else y
        sol=lsq_linear(augA,augy,bounds=(0,np.inf)).x
    else: sol=np.linalg.lstsq(lhs,rhs,rcond=None)[0]
    residual=A@sol-y
    return {"emissivity":sol,"synthetic_measurements":A@sol,"residual":residual,"residual_norm":float(np.linalg.norm(residual)),"regularization":lam,"caveat":"Reconstruction quality depends on calibrated geometry matrix, coverage, regularization, and uncertainty model."}