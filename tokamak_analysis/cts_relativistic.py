"""Relativistic gyrotropic fast-ion response and Klimontovich shot noise.

This module solves the linearized relativistic Vlasov equation in a uniform
magnetic field by Fourier collocation in gyrophase and momentum-space
quadrature. It supports an anisotropic bi-Gaussian momentum distribution and
user-supplied tabulated gyrotropic distributions f(p_perp,p_parallel).

The distribution is defined in momentum space; relativistic kinematics
v=p/(gamma*m), gyrofrequency qB/(gamma*m), and the exact Lorentz force are
used. The fluctuation source is computed from the Klimontovich particle
shot-noise spectrum (not an equilibrium fluctuation-dissipation closure),
with a documented Lorentzian representation of the causal resonance delta.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
from scipy.special import roots_laguerre, roots_hermite

EPS0 = 8.8541878128e-12
C = 299792458.0
E_CHARGE = 1.602176634e-19
ME = 9.1093837015e-31
AMU = 1.66053906660e-27


def _positive(x, name):
    x = float(x)
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return x


def _momentum_grid(distribution, mass_kg, quadrature_order):
    if not isinstance(distribution, Mapping):
        raise ValueError("fast-ion distribution must be a mapping")
    kind = distribution.get("kind")
    if kind == "relativistic_bigaussian":
        tperp = _positive(distribution.get("temperature_perpendicular_eV"), "temperature_perpendicular_eV")
        tpar = _positive(distribution.get("temperature_parallel_eV"), "temperature_parallel_eV")
        xp, wp = roots_laguerre(int(quadrature_order))
        xa, wa = roots_hermite(int(quadrature_order))
        pperp_th = np.sqrt(2.0 * mass_kg * tperp * E_CHARGE)
        ppar_th = np.sqrt(2.0 * mass_kg * tpar * E_CHARGE)
        pperp, ppar = np.meshgrid(pperp_th*np.sqrt(xp), ppar_th*xa, indexing="ij")
        f0 = np.exp(-xp[:, None] - xa[None, :]**2) / (
            np.pi**1.5 * pperp_th**2 * ppar_th
        )
        weights = np.broadcast_to(wp[:, None]*wa[None, :]/np.sqrt(np.pi), f0.shape).copy()
        dpperp = -pperp/(mass_kg*tperp*E_CHARGE)*f0
        dppar = -ppar/(mass_kg*tpar*E_CHARGE)*f0
        return pperp, ppar, f0, dpperp, dppar, weights
    if kind == "tabulated_gyrotropic":
        pp = np.asarray(distribution.get("p_perpendicular_kg_m_s"), dtype=float)
        pa = np.asarray(distribution.get("p_parallel_kg_m_s"), dtype=float)
        f0 = np.asarray(distribution.get("distribution_kg_m_s_inverse3"), dtype=float)
        if pp.ndim != 1 or pa.ndim != 1 or pp.size < 4 or pa.size < 4:
            raise ValueError("tabulated gyrotropic momentum grids must be 1-D with >=4 nodes")
        if np.any(~np.isfinite(pp)) or np.any(~np.isfinite(pa)) or np.any(np.diff(pp)<=0) or np.any(np.diff(pa)<=0) or pp[0]<0:
            raise ValueError("momentum grids must be finite, increasing and p_perpendicular non-negative")
        if f0.shape != (pp.size, pa.size) or np.any(~np.isfinite(f0)) or np.any(f0<0):
            raise ValueError("tabulated distribution must be finite, non-negative and shaped (n_pperp,n_ppar)")
        pperp, ppar = np.meshgrid(pp, pa, indexing="ij")
        dpp = np.gradient(pp)
        dpa = np.gradient(pa)
        weights = 2.0*np.pi*pperp*dpp[:,None]*dpa[None,:]
        norm = float(np.sum(weights*f0))
        if norm <= 0:
            raise ValueError("tabulated gyrotropic distribution has zero normalization")
        f0 = f0/norm
        dpperp, dppar = np.gradient(f0, pp, pa, edge_order=2)
        return pperp, ppar, f0, dpperp, dppar, weights
    raise ValueError("fast-ion distribution kind must be relativistic_bigaussian or tabulated_gyrotropic")


def relativistic_gyrotropic_three_wave_vertex(
    omega_incident_rad_s, wavevector_incident_rad_m, incident_electric_field,
    omega_fluctuation_rad_s, wavevector_fluctuation_rad_m,
    magnetic_field_T, *, density_m3, mass_kg, charge_C, distribution,
    quadrature_order=8, gyrophase_points=32, resonance_width_rad_s=1e6,
):
    """Return the relativistic kinetic second-order current vertex for CTS.

    The calculation solves the relativistic Vlasov equation in momentum and
    gyrophase space. First-order responses to the incident and fluctuation
    waves are computed from the same supplied gyrotropic VDF; their quadratic
    Lorentz-force terms drive the scattered-frequency response. The returned
    3x3 matrix maps the fluctuation electric-field components (laboratory
    coordinates) to the second-order scattered current density.

    This is a kinetic three-wave vertex, not a relativistic correction factor
    applied to the low-temperature Bindslev tensor. The fluctuation magnetic
    field is constrained by Faraday's law B_f = k_f x E_f / omega_f.
    The implementation assumes a homogeneous local plasma and a prescribed
    gyrotropic equilibrium distribution. It does not include collisions,
    finite-orbit-width effects, or nonlinear modification of the equilibrium.
    """
    wi = float(omega_incident_rad_s)
    wf = float(omega_fluctuation_rad_s)
    ws = wi + wf
    ki_lab = np.asarray(wavevector_incident_rad_m, dtype=float)
    kf_lab = np.asarray(wavevector_fluctuation_rad_m, dtype=float)
    b_lab = np.asarray(magnetic_field_T, dtype=float)
    ei_lab = np.asarray(incident_electric_field, dtype=complex)
    density = _positive(density_m3, "density_m3")
    mass = _positive(mass_kg, "mass_kg")
    charge = float(charge_C)
    width = _positive(resonance_width_rad_s, "resonance_width_rad_s")
    if (not np.isfinite(wi) or wi <= 0 or not np.isfinite(wf) or wf == 0
            or not np.isfinite(ws) or ws <= 0 or charge == 0
            or ki_lab.shape != (3,) or kf_lab.shape != (3,) or b_lab.shape != (3,)
            or ei_lab.shape != (3,) or np.any(~np.isfinite(ki_lab))
            or np.any(~np.isfinite(kf_lab)) or np.any(~np.isfinite(b_lab))
            or np.any(~np.isfinite(ei_lab)) or np.linalg.norm(ki_lab) == 0
            or np.linalg.norm(kf_lab) == 0 or np.linalg.norm(b_lab) == 0
            or np.linalg.norm(ei_lab) == 0):
        raise ValueError("finite non-zero wavevectors, magnetic field, incident field and valid three-wave frequencies are required")
    if int(quadrature_order) < 4 or int(gyrophase_points) < 16 or int(gyrophase_points) % 2:
        raise ValueError("quadrature_order >=4 and even gyrophase_points >=16 are required")

    bhat = b_lab/np.linalg.norm(b_lab)
    trial = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(trial, bhat)) > 0.9:
        trial = np.array([0.0, 1.0, 0.0])
    e1 = trial - np.dot(trial, bhat)*bhat
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(bhat, e1)
    rotation = np.column_stack((e1, e2, bhat))
    ki = rotation.T@ki_lab
    kf = rotation.T@kf_lab
    ei = rotation.T@ei_lab
    ei = ei/np.linalg.norm(ei)
    bi = np.cross(ki, ei)/wi
    pperp, ppar, f0, dfperp, dfpar, weights = _momentum_grid(
        distribution, mass, int(quadrature_order)
    )
    pperp_axis = pperp[:, 0]
    ppar_axis = ppar[0, :]
    npp, npa = pperp.shape
    nphi = int(gyrophase_points)
    phi = 2.0*np.pi*np.arange(nphi)/nphi
    cosphi, sinphi = np.cos(phi), np.sin(phi)
    harmonics = np.fft.fftfreq(nphi, d=1.0/nphi)
    omega_g = np.empty((npp, npa), dtype=float)
    velocity_grid = np.empty((npp, npa, nphi, 3), dtype=float)
    operator_i = np.empty((npp, npa, nphi, nphi), dtype=complex)
    operator_f = np.empty_like(operator_i)
    operator_s = np.empty_like(operator_i)
    force_i = np.empty((npp, npa, nphi, 3), dtype=complex)
    force_f = np.empty((npp, npa, nphi, 3, 3), dtype=complex)
    rhs_i = np.empty((npp, npa, nphi), dtype=complex)
    rhs_f = np.empty((npp, npa, nphi, 3), dtype=complex)
    fluctuation_fields = np.eye(3, dtype=complex)
    bf_columns = np.column_stack([
        np.cross(kf, fluctuation_fields[:, j])/wf for j in range(3)
    ])
    for ip in range(npp):
        for ia in range(npa):
            pp = float(pperp[ip, ia])
            pa = float(ppar[ip, ia])
            ptotal = np.hypot(pp, pa)
            gamma = np.sqrt(1.0+(ptotal/(mass*C))**2)
            vp = pp/(gamma*mass)
            va = pa/(gamma*mass)
            velocity = np.column_stack((vp*cosphi, vp*sinphi, np.full(nphi, va)))
            velocity_grid[ip, ia] = velocity
            omega_c = charge*np.linalg.norm(b_lab)/(gamma*mass)
            omega_g[ip, ia] = omega_c
            grad0 = np.column_stack((
                dfperp[ip, ia]*cosphi,
                dfperp[ip, ia]*sinphi,
                np.full(nphi, dfpar[ip, ia]),
            ))
            for label, omega, kval, efield, bfield, store_force, store_rhs, store_op in (
                ("incident", wi, ki, ei, bi, force_i, rhs_i, operator_i),
            ):
                force = charge*(efield[None, :]+np.cross(velocity, bfield))
                store_force[ip, ia] = force
                store_rhs[ip, ia] = -np.sum(force*grad0, axis=1)
                detuning = omega-np.einsum("j,ij->i", kval, velocity)
                operator = np.diag(width-1j*detuning)-omega_c*np.diag(harmonics*0+1) @ np.zeros((nphi,nphi))
                # Fourier differentiation in gyroangle is assembled explicitly.
                dphi = np.fft.ifft(np.diag(1j*harmonics) @ np.fft.fft(np.eye(nphi), axis=0), axis=0)
                operator = np.diag(width-1j*detuning)-omega_c*dphi
                store_op[ip, ia] = operator
            for j in range(3):
                force = charge*(fluctuation_fields[:, j][None, :]
                                 + np.cross(velocity, bf_columns[:, j]))
                force_f[ip, ia, :, :, j] = force
                rhs_f[ip, ia, :, j] = -np.sum(force*grad0, axis=1)
            detuning_f = wf-np.einsum("j,ij->i", kf, velocity)
            detuning_s = ws-np.einsum("j,ij->i", ki+kf, velocity)
            dphi = np.fft.ifft(np.diag(1j*harmonics) @ np.fft.fft(np.eye(nphi), axis=0), axis=0)
            operator_f[ip, ia] = np.diag(width-1j*detuning_f)-omega_c*dphi
            operator_s[ip, ia] = np.diag(width-1j*detuning_s)-omega_c*dphi

    # First-order distribution perturbations for probe and each unit fluctuation E.
    delta_i = np.empty((npp, npa, nphi), dtype=complex)
    delta_f = np.empty((npp, npa, nphi, 3), dtype=complex)
    for ip in range(npp):
        for ia in range(npa):
            try:
                delta_i[ip, ia] = np.linalg.solve(operator_i[ip, ia], rhs_i[ip, ia])
                delta_f[ip, ia] = np.linalg.solve(operator_f[ip, ia], rhs_f[ip, ia])
            except np.linalg.LinAlgError as exc:
                raise ValueError("singular first-order relativistic Vlasov operator; increase resonance width or quadrature resolution") from exc

    def momentum_gradient(values):
        dpperp, dppar = np.gradient(
            values, pperp_axis, ppar_axis, axis=(0, 1), edge_order=1
        )
        dphi = np.fft.ifft(
            np.fft.fft(values, axis=2)*(1j*harmonics)[None, None, :, None],
            axis=2,
        ) if values.ndim == 4 else np.fft.ifft(
            np.fft.fft(values, axis=2)*(1j*harmonics)[None, None, :],
            axis=2,
        )
        pp_safe = np.maximum(pperp[:, :, None], np.finfo(float).tiny)
        gx = cosphi[None, None, :]*dpperp - sinphi[None, None, :]*dphi/pp_safe
        gy = sinphi[None, None, :]*dpperp + cosphi[None, None, :]*dphi/pp_safe
        gz = dppar
        return np.stack((gx, gy, gz), axis=-1)

    grad_i = momentum_gradient(delta_i)
    grad_f = momentum_gradient(delta_f)
    current_local = np.zeros((3, 3), dtype=complex)
    for ip in range(npp):
        for ia in range(npa):
            rhs2 = -np.einsum("pi,pji->pj", force_i[ip, ia], grad_f[ip, ia])
            rhs2 -= np.einsum("pji,pi->pj", force_f[ip, ia], grad_i[ip, ia])
            try:
                delta2 = np.linalg.solve(operator_s[ip, ia], rhs2)
            except np.linalg.LinAlgError as exc:
                raise ValueError("singular second-order relativistic Vlasov operator; increase resonance width or quadrature resolution") from exc
            velocity = velocity_grid[ip, ia]
            current_local += charge*density*float(weights[ip, ia])*(velocity.T@delta2)/nphi
    current_lab = rotation@current_local
    return {
        "current_vertex_A_m2_per_V_m": current_lab,
        "incident_frequency_rad_s": wi,
        "fluctuation_frequency_rad_s": wf,
        "scattered_frequency_rad_s": ws,
        "distribution_kind": distribution.get("kind"),
        "relativistic_kinematics": True,
        "second_order_vlasov": True,
        "vertex_definition": "J^(2)(omega_i+omega_f,k_i+k_f) from the relativistic Vlasov equation, quadratic Lorentz-force coupling",
        "momentum_quadrature_order": int(quadrature_order),
        "gyrophase_points": nphi,
        "resonance_width_rad_s": width,
        "assumptions": [
            "Homogeneous local magnetic field and gyrotropic equilibrium VDF.",
            "Prescribed first-order electromagnetic waves with Faraday-consistent magnetic fields.",
            "Collisionless relativistic Vlasov equation with explicit Lorentzian resonance broadening.",
            "No finite-orbit-width, collisions, nonlinear equilibrium evolution or mode conversion."
        ],
    }



def relativistic_gyrotropic_species_response(
    omega_rad_s, wavevector_rad_m, magnetic_field_T, *,
    density_m3, mass_kg, charge_C, distribution,
    quadrature_order=8, gyrophase_points=32, resonance_width_rad_s=1e6,
):
    """Return relativistic gyrotropic conductivity and bare Klimontovich Q_jj.

    Fourier collocation solves the linearized Vlasov equation in gyrophase.
    Q_jj is the uncorrelated-particle spectrum using the same relativistic
    gyro-orbits and a normalized Lorentzian approximation to the resonance
    delta function. Both outputs are 3x3 complex Hermitian-compatible tensors.
    """
    omega = float(omega_rad_s)
    k = np.asarray(wavevector_rad_m, dtype=float)
    b = np.asarray(magnetic_field_T, dtype=float)
    density = _positive(density_m3, "density_m3")
    mass = _positive(mass_kg, "mass_kg")
    charge = float(charge_C)
    width = _positive(resonance_width_rad_s, "resonance_width_rad_s")
    if not np.isfinite(omega) or omega == 0 or k.shape!=(3,) or b.shape!=(3,) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(b)) or np.linalg.norm(b)==0 or charge==0:
        raise ValueError("finite non-zero frequency, charge and magnetic field plus finite 3-vectors are required")
    if int(quadrature_order)<4 or int(gyrophase_points)<16 or int(gyrophase_points)%2:
        raise ValueError("quadrature_order >=4 and even gyrophase_points >=16 are required")
    bhat=b/np.linalg.norm(b)
    trial=np.array([1.0,0.0,0.0])
    if abs(np.dot(trial,bhat))>0.9: trial=np.array([0.0,1.0,0.0])
    e1=trial-np.dot(trial,bhat)*bhat; e1/=np.linalg.norm(e1)
    e2=np.cross(bhat,e1)
    rotation=np.column_stack((e1,e2,bhat))
    klab=k.copy(); k=rotation.T@k
    kperp,kpar=float(np.hypot(k[0],k[1])),float(k[2])
    if kperp>0:
        # Align local x with k_perp by a rotation about B.
        angle=np.arctan2(k[1],k[0])
        rz=np.array([[np.cos(angle),np.sin(angle),0.0],[-np.sin(angle),np.cos(angle),0.0],[0.0,0.0,1.0]])
        rotation=rotation@rz
        k=rz.T@k
    pperp,ppar,f0,dfperp,dfpar,weights=_momentum_grid(distribution,mass,int(quadrature_order))
    nphi=int(gyrophase_points)
    phi=2*np.pi*np.arange(nphi)/nphi
    harmonics=np.fft.fftfreq(nphi,d=1.0/nphi)
    # Spectral derivative matrix: D f = F^{-1} diag(i m) F f.
    dft=np.fft.fft(np.eye(nphi),axis=0)/nphi
    invdft=np.fft.ifft(np.eye(nphi),axis=0)*nphi
    derivative=invdft@np.diag(1j*harmonics)@dft
    conductivity=np.zeros((3,3),dtype=complex)
    bare=np.zeros((3,3),dtype=complex)
    eye=np.eye(3)
    for ip in range(pperp.shape[0]):
        for ia in range(pperp.shape[1]):
            pp=float(pperp[ip,ia]); pa=float(ppar[ip,ia]); f=float(f0[ip,ia])
            if f<=0 or pp<=0: continue
            pvec=np.sqrt(pp*pp+pa*pa)
            gamma=np.sqrt(1.0+(pvec/(mass*C))**2)
            vperp=pp/(gamma*mass); vpar=pa/(gamma*mass)
            omega_g=charge*np.linalg.norm(b)/(gamma*mass)
            velocity=np.column_stack((vperp*np.cos(phi),vperp*np.sin(phi),np.full(nphi,vpar)))
            gradf=np.column_stack((dfperp[ip,ia]*np.cos(phi),dfperp[ip,ia]*np.sin(phi),np.full(nphi,dfpar[ip,ia])))
            phase=np.exp(1j*kperp*pp/(charge*np.linalg.norm(b))*np.sin(phi))
            # Exact relativistic Lorentz force for each unit electric-field component.
            rhs=np.empty((nphi,3),dtype=complex)
            for col in range(3):
                efield=eye[col]
                bfield=np.cross(k,efield)/omega
                force=charge*(efield[None,:]+np.cross(velocity,bfield))
                rhs[:,col]=-np.sum(force*gradf,axis=1)
            streaming=width-1j*(omega-kpar*vpar-kperp*vperp*np.cos(phi))
            operator=np.diag(streaming)-omega_g*derivative
            try:
                delta_f=np.linalg.solve(operator,rhs)
            except np.linalg.LinAlgError as exc:
                raise ValueError("relativistic gyrophase Vlasov operator is singular; increase resonance_width_rad_s") from exc
            measure=float(weights[ip,ia])*f
            # delta_f already contains the derivative of the normalized f0.
            # Integrating it with weights*f0 would incorrectly square the VDF.
            conductivity += charge*density*float(weights[ip,ia])*(velocity.T@delta_f)/nphi
            # Klimontovich shot noise: Fourier harmonics of the exact
            # relativistic orbit current, broadened only by the explicit width.
            orbit=velocity*phase[:,None]
            coeff=np.fft.fft(orbit,axis=0)/nphi
            detuning=omega-kpar*vpar-harmonics*omega_g
            lorentz=width/(np.pi*(detuning*detuning+width*width))
            for m in range(nphi):
                a=coeff[m]
                bare += (2.0*np.pi*density*charge*charge*measure*lorentz[m]
                         * np.outer(a,a.conj()))
    p_total=np.sqrt(pperp*pperp+ppar*ppar)
    gamma_grid=np.sqrt(1.0+(p_total/(mass*C))**2)
    conductivity_lab=rotation@conductivity@rotation.T
    bare_lab=rotation@bare@rotation.T
    bare_lab=0.5*(bare_lab+bare_lab.conj().T)
    # The gyrotropic Vlasov response is generally non-Hermitian; do not
    # symmetrize conductivity because its anti-Hermitian part is dispersive.
    return {
        "conductivity_S_m": conductivity_lab,
        "bare_current_spectral_covariance": bare_lab,
        "distribution_kind": distribution["kind"],
        "relativistic_kinematics": True,
        "anisotropy": "p_perpendicular and p_parallel distributions are independently specified",
        "momentum_space_normalization": "integral 2*pi*p_perp*f dp_perp dp_parallel = 1",
        "resonance_broadening_rad_s": width,
        "gyrophase_points": nphi,
        "quadrature_order": int(quadrature_order),
        "velocity_max_fraction_c": float(np.max(np.sqrt((pperp/(gamma_grid*mass))**2+(ppar/(gamma_grid*mass))**2))/C),
        "noise_model": "Klimontovich uncorrelated-particle orbit spectrum; not equilibrium FDT",
        "wavevector_lab_rad_m": klab,
    }
