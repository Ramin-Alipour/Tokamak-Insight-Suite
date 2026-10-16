"""Radial/poloidal transport proxies and Reynolds stress from fluctuations."""
import numpy as np
def fluctuation_transport(density_fluctuation, velocity_fluctuation, *, coordinate_m=None, direction="radial"):
    n=np.asarray(density_fluctuation,dtype=float); v=np.asarray(velocity_fluctuation,dtype=float)
    if n.ndim!=1 or v.shape!=n.shape or len(n)<2 or not np.all(np.isfinite(n)) or not np.all(np.isfinite(v)): raise ValueError("Matching finite 1-D fluctuation signals required")
    dn=n-np.mean(n); dv=v-np.mean(v); flux=float(np.mean(dn*dv))
    out={"direction":direction,"particle_flux_covariance_m-2_s-1":flux,"density_fluctuation_rms":float(np.std(dn)),"velocity_fluctuation_rms_m_s":float(np.std(dv))}
    if coordinate_m is not None:
        x=np.asarray(coordinate_m,dtype=float)
        if x.shape!=n.shape or not np.all(np.isfinite(x)) or np.any(np.diff(x)<=0): raise ValueError("Coordinate must match and increase")
        out["coordinate_m"]=x; out["mean_density_gradient_m-4"]=float(np.mean(np.gradient(n,x)))
    out["caveat"]="Covariance is a transport proxy; absolute flux requires synchronized, calibrated density and velocity fluctuations."
    return out
def reynolds_stress(v_radial_m_s,v_poloidal_m_s,density_kg_m3=1.0):
    vr=np.asarray(v_radial_m_s,dtype=float); vp=np.asarray(v_poloidal_m_s,dtype=float); rho=float(density_kg_m3)
    if vr.shape!=vp.shape or vr.ndim!=1 or len(vr)<2 or not np.all(np.isfinite(vr)) or not np.all(np.isfinite(vp)) or not np.isfinite(rho) or rho<=0: raise ValueError("Invalid velocity signals or mass density")
    ar=vr-np.mean(vr); ap=vp-np.mean(vp)
    return {"reynolds_stress_Pa":float(rho*np.mean(ar*ap)),"velocity_covariance_m2_s-2":float(np.mean(ar*ap)),"density_kg_m3":rho}

def radial_diffusive_particle_flux(diffusion_coefficient_m2_s, radial_density_gradient_m4):
    """Radial Fick-law flux Gamma_r=-D_r*d<n>/dr."""
    D,g=np.broadcast_arrays(np.asarray(diffusion_coefficient_m2_s,dtype=float),np.asarray(radial_density_gradient_m4,dtype=float))
    if not np.all(np.isfinite(D)) or not np.all(np.isfinite(g)) or np.any(D<0): raise ValueError("Diffusivity non-negative and inputs finite")
    return {"radial_particle_flux_m2_s":-D*g,"equation":"Gamma_r=-D_r*d<n>/dr"}


def poloidal_diffusive_particle_flux(diffusion_coefficient_m2_s, poloidal_density_gradient_m4):
    """Poloidal Fick-law flux Gamma_theta=-D_theta*d<n>/dtheta_phys."""
    D,g=np.broadcast_arrays(np.asarray(diffusion_coefficient_m2_s,dtype=float),np.asarray(poloidal_density_gradient_m4,dtype=float))
    if not np.all(np.isfinite(D)) or not np.all(np.isfinite(g)) or np.any(D<0): raise ValueError("Diffusivity non-negative and inputs finite")
    return {"poloidal_particle_flux_m2_s":-D*g,"equation":"Gamma_theta=-D_theta*d<n>/dtheta_phys"}


def turbulent_momentum_flux(v_radial_m_s, v_poloidal_m_s, mass_density_kg_m3):
    """Outward turbulent momentum flux convention Pi_rtheta=-rho*<v_r' v_theta'>."""
    vr=np.asarray(v_radial_m_s,dtype=float); vt=np.asarray(v_poloidal_m_s,dtype=float); rho=float(mass_density_kg_m3)
    if vr.ndim!=1 or vt.shape!=vr.shape or len(vr)<2 or not np.all(np.isfinite(vr)) or not np.all(np.isfinite(vt)) or not np.isfinite(rho) or rho<=0:
        raise ValueError("Matching finite velocity signals and positive mass density required")
    cov=float(np.mean((vr-np.mean(vr))*(vt-np.mean(vt))))
    return {"turbulent_momentum_flux_Pa":-rho*cov,"velocity_covariance_m2_s2":cov,
            "equation":"Pi_rtheta=-rho*<v_r_prime*v_theta_prime>",
            "convention":"Negative covariance sign chosen for outward flux convention; verify sign conventions for the target publication."}
