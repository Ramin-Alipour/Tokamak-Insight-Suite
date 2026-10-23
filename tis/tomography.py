"""Tomographic inverse problems and reconstruction diagnostics."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy import optimize, linalg

def build_tomography_matrix_2d(ray_start_xz: Any, ray_end_xz: Any,
                               x_edges_m: Any, z_edges_m: Any) -> np.ndarray:
    """Build a 2-D line-integral response matrix from ray segments and rectangular pixel edges.

    Each row is one ray; columns are flattened as iz * nx + ix. Entries are path
    lengths through each pixel in metres. Ray start/end coordinates are (R or x, z)
    in metres. Segments are clipped to the grid; pixels outside the grid contribute 0.
    """
    starts=np.asarray(ray_start_xz,dtype=float)
    ends=np.asarray(ray_end_xz,dtype=float)
    x_edges=np.asarray(x_edges_m,dtype=float).ravel()
    z_edges=np.asarray(z_edges_m,dtype=float).ravel()
    if starts.ndim!=2 or starts.shape[1]!=2 or ends.shape!=starts.shape or starts.shape[0]<1:
        raise ValueError("Ray start and end arrays must both have shape (n_rays, 2).")
    if not np.all(np.isfinite(starts)) or not np.all(np.isfinite(ends)):
        raise ValueError("Ray endpoints must be finite.")
    for name,edges in (("x",x_edges),("z",z_edges)):
        if len(edges)<2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges)<=0):
            raise ValueError(f"{name}_edges_m must be a finite, strictly increasing edge vector.")
    nx=len(x_edges)-1; nz=len(z_edges)-1
    matrix=np.zeros((len(starts),nx*nz),dtype=float)
    for row,(p0,p1) in enumerate(zip(starts,ends)):
        delta=p1-p0
        length=float(np.linalg.norm(delta))
        if length<=np.finfo(float).eps:
            raise ValueError(f"Ray {row} has coincident start and end points.")
        ts=[0.0,1.0]
        if abs(delta[0])>np.finfo(float).eps:
            ts.extend(((x-p0[0])/delta[0] for x in x_edges if 0.0<(x-p0[0])/delta[0]<1.0))
        if abs(delta[1])>np.finfo(float).eps:
            ts.extend(((z-p0[1])/delta[1] for z in z_edges if 0.0<(z-p0[1])/delta[1]<1.0))
        ts.sort()
        unique=[ts[0]]
        for value in ts[1:]:
            if value-unique[-1]>1e-12: unique.append(value)
        for left,right in zip(unique[:-1],unique[1:]):
            if right-left<=1e-14: continue
            mid=p0+((left+right)/2.0)*delta
            ix=int(np.searchsorted(x_edges,mid[0],side="right")-1)
            iz=int(np.searchsorted(z_edges,mid[1],side="right")-1)
            if 0<=ix<nx and 0<=iz<nz:
                matrix[row,iz*nx+ix]+=length*(right-left)
    if not np.any(matrix>0):
        raise ValueError("No ray segment intersects the supplied reconstruction grid.")
    return matrix


def regularized_tomography(A: Any, y: Any, lam: float = 1e-2, nonnegative: bool = True,
                          measurement_sigma: Any | None = None,
                          regularization_order: int = 1) -> dict[str, Any]:
    """Solve a normalized Tikhonov inverse problem with order-0/1/2 regularization."""
    A=np.asarray(A,float); y=np.asarray(y,float).ravel()
    if A.ndim!=2 or A.shape[0]!=len(y) or A.size==0: raise ValueError('A must be a nonempty 2D matrix with one row per measurement.')
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(y)): raise ValueError('Response matrix and measurements must contain only finite values.')
    if lam<0 or not np.isfinite(lam): raise ValueError('Regularization must be finite and nonnegative.')
    if isinstance(regularization_order, bool) or not isinstance(regularization_order, (int, np.integer)) or regularization_order not in (0, 1, 2):
        raise ValueError("regularization_order must be an integer: 0 (magnitude), 1 (gradient) or 2 (curvature).")
    if measurement_sigma is not None:
        sigma=np.asarray(measurement_sigma,float).ravel()
        if len(sigma)!=len(y) or not np.all(np.isfinite(sigma)) or np.any(sigma<=0):
            raise ValueError("measurement_sigma must be finite, positive and have one value per measurement.")
        Aw=A/sigma[:,None]; yw=y/sigma
    else:
        sigma=None; Aw=A; yw=y
    # Normalize A by its weighted spectral norm so lambda is dimensionless with
    # respect to response scale. First-order regularization penalizes profile
    # gradients rather than shrinking every emissivity value toward zero.
    response_scale=float(np.linalg.norm(Aw,ord=2))
    if not np.isfinite(response_scale) or response_scale<=np.finfo(float).eps:
        raise ValueError("Response matrix has zero or invalid weighted norm.")
    An=Aw/response_scale
    yn=yw/response_scale
    identity=np.eye(A.shape[1])
    operator=identity if regularization_order==0 else np.diff(identity,n=regularization_order,axis=0)
    operator_norm=float(np.linalg.norm(operator,ord=2)) if operator.size else 1.0
    if operator_norm>0: operator=operator/operator_norm
    if nonnegative:
        augA=np.vstack([An,np.sqrt(lam)*operator]) if lam else An
        augy=np.concatenate([yn,np.zeros(operator.shape[0])]) if lam else yn
        res=optimize.lsq_linear(augA,augy,bounds=(0,np.inf),method='trf')
        x=res.x; success=bool(res.success); message=str(res.message)
    else:
        if lam>0:
            augA=np.vstack([An,np.sqrt(lam)*operator])
            augy=np.concatenate([yn,np.zeros(operator.shape[0])])
        else:
            augA,augy=An,yn
        x=np.linalg.lstsq(augA,augy,rcond=None)[0]
        success=True; message='least-squares / Tikhonov solution'
    residual=A@x-y
    weighted_residual=residual if sigma is None else residual/sigma
    regularization_norm=float(np.linalg.norm(operator@x))
    return {'solution':x,'residual':residual,'residual_norm':float(np.linalg.norm(residual)),
            'weighted_residual_norm':float(np.linalg.norm(weighted_residual)),
            'regularization':float(lam),'regularization_order':int(regularization_order),
            'regularization_operator_norm':regularization_norm,'response_spectral_norm':response_scale,
            'success':success,'message':message,'nonnegative':bool(nonnegative),
            'measurement_weighting':'unweighted' if sigma is None else 'inverse-variance (1/sigma²)'}


def select_tikhonov_lambda(A: Any, y: Any, lambdas: Any | None = None,
                           nonnegative: bool = True, measurement_sigma: Any | None = None) -> dict[str, Any]:
    """Choose a candidate Tikhonov parameter from discrete L-curve curvature.

    This is a heuristic selection aid, not a substitute for a noise model or validation
    against known profiles. The full curve is returned for review.
    """
    candidates=np.logspace(-8,2,31) if lambdas is None else np.asarray(lambdas,float).ravel()
    if len(candidates)<3 or not np.all(np.isfinite(candidates)) or np.any(candidates<=0) or np.any(np.diff(candidates)<=0):
        raise ValueError("Provide at least three finite, positive, strictly increasing candidate lambdas.")
    residuals=[]; norms=[]
    for value in candidates:
        result=regularized_tomography(A,y,float(value),nonnegative,measurement_sigma)
        residuals.append(max(result['weighted_residual_norm'],np.finfo(float).tiny))
        norms.append(max(float(result['regularization_operator_norm']),np.finfo(float).tiny))
    lx=np.log(residuals); ly=np.log(norms); lt=np.log(candidates)
    dx=np.gradient(lx,lt); dy=np.gradient(ly,lt)
    ddx=np.gradient(dx,lt); ddy=np.gradient(dy,lt)
    curvature=np.abs(dx*ddy-dy*ddx)/np.maximum((dx*dx+dy*dy)**1.5,np.finfo(float).tiny)
    interior=np.arange(1,len(candidates)-1)
    selected=int(interior[np.argmax(curvature[interior])])
    return {'selected_lambda':float(candidates[selected]),'candidate_lambdas':candidates,
            'weighted_residual_norms':np.asarray(residuals),
            'regularization_norms':np.asarray(norms),'solution_norms':np.asarray(norms),
            'discrete_log_lcurve_curvature':curvature,
            'selection_method':'maximum discrete curvature of log-weighted-residual vs log-regularization-norm L-curve',
            'warning':'Heuristic selection; inspect the returned L-curve and validate with measurement noise and synthetic/known-profile tests.'}


def abel_reconstruction(chord_impact_m: Any, line_integrals: Any, radial_edges_m: Any,
                        lam: float = 1e-2, nonnegative: bool = True,
                        regularization_order: int = 1,
                        measurement_sigma: Any | None = None) -> dict[str, Any]:
    """Axisymmetric Abel inversion with chord-coverage and response-rank diagnostics."""
    b=np.asarray(chord_impact_m,float).ravel(); y=np.asarray(line_integrals,float).ravel(); edges=np.asarray(radial_edges_m,float).ravel()
    if len(b)!=len(y) or len(b)<3: raise ValueError('At least 3 paired chord impact parameters and line integrals are required.')
    if len(edges)<3 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges)<=0) or edges[0]<0:
        raise ValueError('Radial edges must be finite, increasing, nonnegative and contain at least 3 values.')
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(y)) or np.any(b<0): raise ValueError('Chord coordinates and line integrals must be finite; impact parameters nonnegative.')
    if np.any(b>=edges[-1]): raise ValueError('Every chord impact parameter must be smaller than the outer radial edge.')
    if len(np.unique(np.round(b,12)))<3: raise ValueError('At least 3 distinct chord impact parameters are required.')
    rin=edges[:-1][None,:]; rout=edges[1:][None,:]; bb=b[:,None]
    A=2*(np.sqrt(np.maximum(rout*rout-bb*bb,0))-np.sqrt(np.maximum(rin*rin-bb*bb,0)))
    A=np.where(bb>=rout,0,A)
    rank=int(np.linalg.matrix_rank(A))
    n_shells=len(edges)-1
    coverage_warning=None
    if rank<n_shells:
        coverage_warning=f"Response matrix rank {rank} is below the {n_shells} reconstructed shells; the profile is not uniquely constrained by these chords."
    elif len(np.unique(np.round(b,12)))<n_shells:
        coverage_warning=f"Only {len(np.unique(np.round(b,12)))} distinct chords constrain {n_shells} shells; inspect resolution and regularization."
    if float(np.max(b))<float(edges[-2]):
        coverage_warning=(coverage_warning+" " if coverage_warning else "")+"No chord is tangent within the outermost radial shell; outer-profile sensitivity may be weak."
    result=regularized_tomography(A,y,lam,nonnegative,measurement_sigma=measurement_sigma,
                                  regularization_order=regularization_order)
    centers=.5*(edges[:-1]+edges[1:])
    return {'radius_centers_m':centers,'emissivity_or_density_profile':result['solution'],'response_matrix':A,
            'residual_norm':result['residual_norm'],'success':result['success'],'message':result['message'],
            'lambda':float(lam),'regularization_order':int(regularization_order),
            'regularization_operator_norm':result['regularization_operator_norm'],'nonnegative':bool(nonnegative),
            'response_matrix_rank':rank,'number_of_radial_shells':int(n_shells),
            'number_of_distinct_chords':int(len(np.unique(np.round(b,12)))),
            'chord_coverage_warning':coverage_warning,
            'measurement_weighting':result['measurement_weighting'],
            'note':'Axisymmetry, straight chord geometry and the units of the supplied line integrals are assumed. Lambda is dimensionless relative to the normalized response matrix; default first-order regularization penalizes profile gradients, not absolute amplitude.'}
