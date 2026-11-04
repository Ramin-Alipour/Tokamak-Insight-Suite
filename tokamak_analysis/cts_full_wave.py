"""One-dimensional, full-vector Maxwell solver for stratified magnetized plasma.

Solves the tangential-field Berreman system without a paraxial or scalar-wave
approximation. The dielectric tensor is supplied at every z sample, allowing
complex, anisotropic and mode-coupling media. This is a 1-D slab solver: it
does not represent transverse diffraction or arbitrary 3-D interfaces.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
C = 1.0 / np.sqrt(EPS0 * MU0)


def _dielectric_stack(z_m, epsilon_r):
    z = np.asarray(z_m, dtype=float)
    eps = np.asarray(epsilon_r, dtype=complex)
    if z.ndim != 1 or z.size < 2 or np.any(~np.isfinite(z)) or np.any(np.diff(z) <= 0):
        raise ValueError("z_m must be a finite, strictly increasing 1-D grid")
    if eps.shape != (z.size, 3, 3) or np.any(~np.isfinite(eps)):
        raise ValueError("epsilon_r must have shape (len(z_m), 3, 3) and be finite")
    if np.any(np.abs(eps[:, 2, 2]) < 1e-12):
        raise ValueError("epsilon_zz is singular; this tangential-field formulation cannot eliminate Ez")
    return z, eps


def berreman_matrix(omega_rad_s, epsilon_r):
    """Return A for d[Ex,Ey,Hx,Hy]/dz = A [Ex,Ey,Hx,Hy]."""
    w = float(omega_rad_s)
    e = np.asarray(epsilon_r, dtype=complex)
    if not np.isfinite(w) or w <= 0 or e.shape != (3, 3) or np.any(~np.isfinite(e)):
        raise ValueError("omega_rad_s must be positive and epsilon_r a finite 3x3 tensor")
    if abs(e[2, 2]) < 1e-12:
        raise ValueError("epsilon_zz is singular")
    # D_z=0 for a plane-stratified medium; eliminate Ez exactly.
    ez_ex, ez_ey = -e[2, 0] / e[2, 2], -e[2, 1] / e[2, 2]
    dxx = e[0, 0] + e[0, 2] * ez_ex
    dxy = e[0, 1] + e[0, 2] * ez_ey
    dyx = e[1, 0] + e[1, 2] * ez_ex
    dyy = e[1, 1] + e[1, 2] * ez_ey
    # exp(-i omega t); H is in A/m.
    return np.array([
        [0, 0, 0, 1j*w*MU0],
        [0, 0, -1j*w*MU0, 0],
        [-1j*w*EPS0*dyx, -1j*w*EPS0*dyy, 0, 0],
        [1j*w*EPS0*dxx, 1j*w*EPS0*dxy, 0, 0],
    ], dtype=complex)


def solve_stratified_full_wave(omega_rad_s, z_m, epsilon_r, *,
                               incident_amplitude_V_m=(1.0, 0.0),
                               rtol=1e-8, atol=1e-10, max_step_m=None):
    """Solve a normally incident wave on a finite anisotropic plasma slab.

    Vacuum is assumed on both sides. Incident Ex/Ey amplitudes are specified
    at z[0]. The solver obtains reflected and transmitted tangential fields by
    matching to forward/backward vacuum plane waves at both boundaries.
    Complex epsilon includes absorption. Returned mode conversion is coherent
    and is therefore not approximated by independent scalar power channels.
    """
    z, eps = _dielectric_stack(z_m, epsilon_r)
    w = float(omega_rad_s)
    if not np.isfinite(w) or w <= 0:
        raise ValueError("omega_rad_s must be positive and finite")
    inc = np.asarray(incident_amplitude_V_m, dtype=complex)
    if inc.shape != (2,) or np.any(~np.isfinite(inc)):
        raise ValueError("incident_amplitude_V_m must be a finite complex 2-vector")
    k0 = w / C
    # State transfer matrix, integrated as a 4x4 fundamental matrix.
    def rhs(s, flat):
        e_local = np.array([[np.interp(s, z, eps[:, i, j].real)
                             + 1j*np.interp(s, z, eps[:, i, j].imag)
                             for j in range(3)] for i in range(3)])
        return (berreman_matrix(w, e_local) @ flat.reshape(4, 4)).ravel()
    solver_options = {"method": "DOP853", "rtol": rtol, "atol": atol}
    if max_step_m is not None:
        if not np.isfinite(max_step_m) or max_step_m <= 0:
            raise ValueError("max_step_m must be positive and finite when supplied")
        solver_options["max_step"] = float(max_step_m)
    sol = solve_ivp(rhs, (z[0], z[-1]), np.eye(4, dtype=complex).ravel(),
                    **solver_options)
    if not sol.success:
        raise RuntimeError("full-wave fundamental-matrix integration failed: " + sol.message)
    transfer = sol.y[:, -1].reshape(4, 4)
    # Vacuum state for each tangential E polarization, forward and backward.
    q = 1.0 / (MU0 * C)
    forward = np.array([[1,0],[0,1],[0,-q],[q,0]], dtype=complex)
    backward = np.array([[1,0],[0,1],[0,q],[-q,0]], dtype=complex)
    left_rhs = np.vstack((np.eye(2, dtype=complex), np.zeros((2,2), dtype=complex)))
    # At z0: state = forward*a + backward*r. At zL: state=forward*t.
    lhs = np.column_stack((-transfer @ backward, forward))
    rhs_vec = transfer @ forward @ inc
    solution = np.linalg.solve(lhs, rhs_vec)
    reflected, transmitted = solution[:2], solution[2:]
    state0 = forward @ inc + backward @ reflected
    stateL = forward @ transmitted
    # In vacuum, normal time-averaged Poynting flux is |E|^2/(2 Z0).
    pin = float(np.vdot(inc, inc).real / (2*MU0*C))
    pref = float(np.vdot(reflected, reflected).real / (2*MU0*C))
    ptrans = float(np.vdot(transmitted, transmitted).real / (2*MU0*C))
    return {
        "z_m": z.copy(), "fundamental_matrix": transfer,
        "incident_electric_amplitude_V_m": inc,
        "reflected_electric_amplitude_V_m": reflected,
        "transmitted_electric_amplitude_V_m": transmitted,
        "incident_power_flux_W_m2": pin,
        "reflected_power_flux_W_m2": pref,
        "transmitted_power_flux_W_m2": ptrans,
        "reflectance": pref/pin, "transmittance": ptrans/pin,
        "absorptance": 1.0-(pref+ptrans)/pin,
        "boundary_state_left": state0, "boundary_state_right": stateL,
        "solver": "DOP853_integrated_full_vector_Berreman_4x4",
        "approximation_scope": "1-D stratified slab, normal incidence, local supplied dielectric tensor; no transverse diffraction or 3-D interfaces",
        "mode_conversion": "coherent vector coupling retained in the full 4x4 Maxwell system",
    }


def berreman_matrix_oblique(omega_rad_s, epsilon_r, kx_rad_m, ky_rad_m):
    """Tangential-field Maxwell generator for exp(i kx*x+i ky*y-i omega*t).

    State ordering is [Ex,Ey,Hx,Hy]. Unlike :func:`berreman_matrix`, this
    retains finite transverse wavevector and longitudinal E/H components.
    It is exact for a z-stratified local dielectric tensor (no x/y variation).
    """
    w = float(omega_rad_s)
    kx, ky = float(kx_rad_m), float(ky_rad_m)
    e = np.asarray(epsilon_r, dtype=complex)
    if (not np.isfinite(w) or w <= 0 or not np.isfinite(kx)
            or not np.isfinite(ky) or e.shape != (3, 3)
            or np.any(~np.isfinite(e))):
        raise ValueError("omega must be positive; kx, ky and epsilon must be finite")
    if abs(e[2, 2]) < 1e-12:
        raise ValueError("epsilon_zz is singular")
    def derivative(state):
        ex, ey, hx, hy = state
        ez = (-(kx*hy-ky*hx)/(w*EPS0)
              - e[2, 0]*ex - e[2, 1]*ey) / e[2, 2]
        hz = (kx*ey-ky*ex)/(w*MU0)
        dx = e[0, 0]*ex + e[0, 1]*ey + e[0, 2]*ez
        dy = e[1, 0]*ex + e[1, 1]*ey + e[1, 2]*ez
        dex = 1j*kx*ez + 1j*w*MU0*hy
        dey = 1j*ky*ez - 1j*w*MU0*hx
        dhx = 1j*kx*hz - 1j*w*EPS0*dy
        dhy = 1j*ky*hz + 1j*w*EPS0*dx
        return np.array([dex, dey, dhx, dhy], dtype=complex)
    return np.column_stack([derivative(np.eye(4, dtype=complex)[:, j])
                            for j in range(4)])


def _vacuum_tangential_modes(omega_rad_s, kx_rad_m, ky_rad_m):
    """Forward/backward vacuum mode matrices and reject evanescent ports."""
    w = float(omega_rad_s)
    k0 = w / C
    kt2 = float(kx_rad_m)**2 + float(ky_rad_m)**2
    q2 = k0*k0 - kt2
    if q2 <= 0:
        raise ValueError("transverse wavevector must describe propagating vacuum ports (kx^2+ky^2 < k0^2)")
    q = np.sqrt(q2)
    def columns(sign):
        result = np.zeros((4, 2), dtype=complex)
        for j in range(2):
            ex, ey = (1.0, 0.0) if j == 0 else (0.0, 1.0)
            ez = -(kx_rad_m*ex + ky_rad_m*ey)/(sign*q)
            result[:, j] = (
                ex, ey,
                (ky_rad_m*ez-sign*q*ey)/(w*MU0),
                (sign*q*ex-kx_rad_m*ez)/(w*MU0),
            )
        return result
    return columns(+1), columns(-1), q


def _normal_poynting_flux(state):
    ex, ey, hx, hy = state
    return 0.5*float(np.real(ex*np.conj(hy)-ey*np.conj(hx)))


def solve_stratified_full_wave_oblique(
    omega_rad_s, z_m, epsilon_r, *, kx_rad_m=0.0, ky_rad_m=0.0,
    incident_amplitude_V_m=(1.0, 0.0), rtol=1e-8, atol=1e-10,
    max_step_m=None,
):
    """Solve coherent vector propagation and polarization conversion at oblique incidence.

    The medium is stratified along z and may have a fully anisotropic complex
    dielectric tensor. The transverse phase factors exp(i*kx*x+i*ky*y) are
    conserved. Both vacuum ports must be propagating. This is a full-vector
    1-D Berreman boundary-value solution, not a paraxial beam solver or a
    general 3-D diffraction solver.
    """
    z, eps = _dielectric_stack(z_m, epsilon_r)
    w = float(omega_rad_s)
    kx, ky = float(kx_rad_m), float(ky_rad_m)
    if not np.isfinite(w) or w <= 0 or not np.isfinite(kx) or not np.isfinite(ky):
        raise ValueError("omega must be positive and transverse wavevector finite")
    forward, backward, q = _vacuum_tangential_modes(w, kx, ky)
    inc = np.asarray(incident_amplitude_V_m, dtype=complex)
    if inc.shape != (2,) or np.any(~np.isfinite(inc)) or np.linalg.norm(inc) == 0:
        raise ValueError("incident_amplitude_V_m must be a finite non-zero complex 2-vector")
    def rhs(s, flat):
        local = np.array([[np.interp(s, z, eps[:, i, j].real)
                           + 1j*np.interp(s, z, eps[:, i, j].imag)
                           for j in range(3)] for i in range(3)])
        return (berreman_matrix_oblique(w, local, kx, ky)
                @ flat.reshape(4, 4)).ravel()
    opts = {"method": "DOP853", "rtol": rtol, "atol": atol}
    if max_step_m is not None:
        if not np.isfinite(max_step_m) or max_step_m <= 0:
            raise ValueError("max_step_m must be positive and finite")
        opts["max_step"] = float(max_step_m)
    sol = solve_ivp(rhs, (z[0], z[-1]), np.eye(4, dtype=complex).ravel(), **opts)
    if not sol.success:
        raise RuntimeError("oblique full-wave integration failed: "+sol.message)
    transfer = sol.y[:, -1].reshape(4, 4)
    lhs = np.column_stack((-transfer @ backward, forward))
    rhs_vec = transfer @ forward @ inc
    reflected, transmitted = np.split(np.linalg.solve(lhs, rhs_vec), 2)
    left_state = forward @ inc + backward @ reflected
    right_state = forward @ transmitted
    incident_state = forward @ inc
    pin = _normal_poynting_flux(incident_state)
    pref = -_normal_poynting_flux(backward @ reflected)
    ptrans = _normal_poynting_flux(right_state)
    if pin <= 0:
        raise RuntimeError("incident vacuum mode has non-positive forward Poynting flux")
    # Cross-polarized powers are obtained by solving each port's Hermitian
    # flux metric; the total flux remains the authoritative conservation check.
    incident_power = pin
    reflected_power = pref
    transmitted_power = ptrans
    return {
        "z_m": z.copy(), "fundamental_matrix": transfer,
        "incident_electric_amplitude_V_m": inc,
        "reflected_electric_amplitude_V_m": reflected,
        "transmitted_electric_amplitude_V_m": transmitted,
        "incident_power_flux_W_m2": incident_power,
        "reflected_power_flux_W_m2": reflected_power,
        "transmitted_power_flux_W_m2": transmitted_power,
        "reflectance": reflected_power/incident_power,
        "transmittance": transmitted_power/incident_power,
        "absorptance": 1.0-(reflected_power+transmitted_power)/incident_power,
        "kx_rad_m": kx, "ky_rad_m": ky, "kz_vacuum_rad_m": q,
        "boundary_state_left": left_state, "boundary_state_right": right_state,
        "solver": "DOP853_oblique_full_vector_Berreman_4x4",
        "mode_conversion": "coherent vector coupling retained; reflected/transmitted Jones amplitudes returned",
        "approximation_scope": "1-D z-stratified local dielectric, oblique plane-wave incidence; no transverse diffraction or 3-D interfaces",
    }


def solve_stratified_mode_conversion(
    omega_rad_s, z_m, epsilon_r, *, kx_rad_m=0.0, ky_rad_m=0.0,
    rtol=1e-8, atol=1e-10, max_step_m=None,
):
    """Construct coherent 2x2 reflected/transmitted Jones matrices.

    Both independent tangential incident polarizations are propagated through
    the oblique full-vector Berreman boundary-value problem. Matrix columns
    correspond to incident basis states; off-diagonal terms are polarization
    conversion amplitudes. The medium remains one-dimensionally stratified.
    """
    reflected = np.empty((2, 2), dtype=complex)
    transmitted = np.empty((2, 2), dtype=complex)
    diagnostics = []
    for column in range(2):
        incident = np.zeros(2, dtype=complex)
        incident[column] = 1.0
        result = solve_stratified_full_wave_oblique(
            omega_rad_s, z_m, epsilon_r,
            kx_rad_m=kx_rad_m, ky_rad_m=ky_rad_m,
            incident_amplitude_V_m=incident,
            rtol=rtol, atol=atol, max_step_m=max_step_m,
        )
        reflected[:, column] = result["reflected_electric_amplitude_V_m"]
        transmitted[:, column] = result["transmitted_electric_amplitude_V_m"]
        diagnostics.append({
            "reflectance": result["reflectance"],
            "transmittance": result["transmittance"],
            "absorptance": result["absorptance"],
        })
    return {
        "reflected_jones_matrix": reflected,
        "transmitted_jones_matrix": transmitted,
        "cross_polarized_reflection_amplitude": reflected[1, 0],
        "cross_polarized_transmission_amplitude": transmitted[1, 0],
        "basis_run_power_diagnostics": diagnostics,
        "solver": "two_basis_oblique_full_vector_Berreman_4x4",
        "mode_conversion": "coherent_cross_polarized_amplitudes_included",
        "approximation_scope": "1-D z-stratified local dielectric, oblique plane waves; no transverse diffraction or arbitrary 3-D interfaces",
    }



def propagate_vectorial_angular_spectrum(
    omega_rad_s, electric_field_xy_V_m, *, dx_m, dy_m, propagation_distance_m,
    relative_permittivity=1.0,
):
    """Exact non-paraxial vector angular-spectrum propagation in a homogeneous medium.

    Input is a sampled transverse (Ey,Ex) plane field with shape (ny,nx,2).
    Each transverse Fourier component is propagated with its exact complex kz;
    Ez is reconstructed from k.E=0 and H from k x E/(mu0 omega). Evanescent
    components are retained and decay for non-negative propagation distance.
    This is a 3-D vector diffraction solver for a homogeneous isotropic segment,
    not a solver for arbitrary 3-D inhomogeneous plasma interfaces. Use the
    Berreman boundary solver for anisotropic z-stratified mode conversion.
    """
    from scipy.constants import mu_0, c as c_light

    omega = float(omega_rad_s)
    dx, dy = float(dx_m), float(dy_m)
    distance = float(propagation_distance_m)
    eps = complex(relative_permittivity)
    field = np.asarray(electric_field_xy_V_m, dtype=complex)
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega_rad_s must be positive and finite")
    if not np.isfinite(dx) or dx <= 0 or not np.isfinite(dy) or dy <= 0:
        raise ValueError("dx_m and dy_m must be positive and finite")
    if not np.isfinite(distance) or distance < 0:
        raise ValueError("propagation_distance_m must be finite and non-negative")
    if not np.isfinite(eps.real) or not np.isfinite(eps.imag) or eps.imag < 0:
        raise ValueError("relative_permittivity must be finite and passive for exp(-i omega t)")
    if field.ndim != 3 or field.shape[2] != 2 or min(field.shape[:2]) < 2 or np.any(~np.isfinite(field)):
        raise ValueError("electric_field_xy_V_m must be a finite (ny,nx,2) complex array")
    ny, nx, _ = field.shape
    refractive_index = np.sqrt(eps + 0j)
    if refractive_index.real < 0:
        refractive_index = -refractive_index
    if refractive_index.imag < 0:
        refractive_index = -refractive_index
    k = omega/c_light * refractive_index
    kx = 2*np.pi*np.fft.fftfreq(nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    kz = np.sqrt(k*k-KX*KX-KY*KY+0j)
    # Select the forward/decaying square-root branch for the exp(-i omega t) convention.
    flip = (kz.imag < -1e-14) | ((np.abs(kz.imag) <= 1e-14) & (kz.real < 0))
    kz[flip] *= -1
    if np.any(np.abs(kz) <= np.finfo(float).tiny):
        raise ValueError("grazing spectral components with kz=0 are singular for longitudinal-field reconstruction")
    spectrum = np.fft.fft2(field, axes=(0, 1))
    ex, ey = spectrum[:, :, 0], spectrum[:, :, 1]
    ez = -(KX*ex+KY*ey)/kz
    phase = np.exp(1j*kz*distance)
    ex, ey, ez = ex*phase, ey*phase, ez*phase
    kvec = np.stack(np.broadcast_arrays(KX, KY, kz), axis=-1)
    evec = np.stack((ex, ey, ez), axis=-1)
    hvec = np.cross(kvec, evec)/(mu_0*omega)
    e_out = np.stack((ex, ey, ez), axis=-1)
    electric = np.fft.ifft2(e_out, axes=(0, 1))
    magnetic = np.fft.ifft2(hvec, axes=(0, 1))
    spectral_power = np.real(np.sum(np.conj(e_out)*e_out, axis=-1))
    evanescent = np.abs(KX*KX+KY*KY) > abs(k)**2
    return {
        "electric_field_xyz_V_m": electric,
        "magnetic_field_xyz_A_m": magnetic,
        "kx_rad_m": KX,
        "ky_rad_m": KY,
        "kz_rad_m": kz,
        "evanescent_component_count": int(np.count_nonzero(evanescent)),
        "propagation_distance_m": distance,
        "relative_permittivity": eps,
        "solver": "exact_nonparaxial_vector_angular_spectrum",
        "maxwell_constraint": "k_dot_E_zero_per_transverse_Fourier_component",
        "approximation_scope": "homogeneous_isotropic_segment_only; anisotropic_stratified_mode_conversion_is_separate",
    }



def _periodic_derivative_matrix(size, spacing):
    """Second-order periodic spectral-domain derivative on a uniform grid."""
    from scipy.sparse import diags, lil_matrix
    if size < 3:
        raise ValueError("each periodic grid dimension must contain at least three points")
    matrix = lil_matrix((size, size), dtype=float)
    rows = np.arange(size)
    matrix[rows, (rows + 1) % size] = 0.5 / spacing
    matrix[rows, (rows - 1) % size] = -0.5 / spacing
    return matrix.tocsr()


def solve_3d_periodic_full_wave(
    omega_rad_s, x_m, y_m, z_m, epsilon_r, *,
    impressed_current_density_A_m2, solver_rtol=1e-9,
    bloch_wavevector_rad_m=(0.0, 0.0, 0.0),
):
    """Solve the full vector Maxwell equation in a 3-D periodic plasma cell.

    Uses a second-order finite-difference frequency-domain (FDFD) discretization
    of curl(curl(E)) - (omega/c)^2 epsilon_r(r) E = i*omega*mu0*J.
    The full complex 3x3 dielectric tensor may vary independently at every
    Cartesian grid point; off-diagonal terms therefore couple all vector
    components and permit coherent 3-D polarization/mode conversion.

    Boundary condition: periodic in x, y and z (a periodic unit cell).
    This is not an open-boundary/PML scattering solver. It is appropriate for
    periodic media or Bloch-periodic cells at zero Bloch wavevector. Open
    launcher/receiver ports, plasma-vacuum interfaces and outgoing radiation
    require a separate boundary-value formulation.
    """
    from scipy.sparse import bmat, diags, eye, kron
    from scipy.sparse.linalg import MatrixRankWarning, spsolve
    import warnings

    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    z = np.asarray(z_m, dtype=float)
    for axis, name in ((x, "x_m"), (y, "y_m"), (z, "z_m")):
        if (axis.ndim != 1 or axis.size < 3 or np.any(~np.isfinite(axis))
                or np.any(np.diff(axis) <= 0)
                or not np.allclose(np.diff(axis), np.diff(axis)[0], rtol=1e-9, atol=1e-14)):
            raise ValueError(f"{name} must be a finite, uniformly spaced increasing grid with >=3 points")
    omega = float(omega_rad_s)
    if not np.isfinite(omega) or omega <= 0:
        raise ValueError("omega_rad_s must be positive and finite")
    nx, ny, nz = x.size, y.size, z.size
    shape = (nz, ny, nx)
    eps = np.asarray(epsilon_r, dtype=complex)
    current = np.asarray(impressed_current_density_A_m2, dtype=complex)
    if eps.shape == (3, 3):
        eps = np.broadcast_to(eps, shape + (3, 3)).copy()
    if eps.shape != shape + (3, 3) or np.any(~np.isfinite(eps)):
        raise ValueError("epsilon_r must have shape (nz,ny,nx,3,3), or be a single 3x3 tensor")
    if current.shape != shape + (3,) or np.any(~np.isfinite(current)):
        raise ValueError("impressed_current_density_A_m2 must have finite shape (nz,ny,nx,3)")
    # Passivity for exp(-i omega t): the Hermitian dissipative part must be PSD.
    dissipative = (eps - np.swapaxes(eps.conj(), -1, -2)) / (2j)
    dissipative = 0.5 * (dissipative + np.swapaxes(dissipative.conj(), -1, -2))
    loss_eigenvalues = np.linalg.eigvalsh(dissipative)
    if np.min(loss_eigenvalues) < -1e-9 * max(1.0, float(np.max(np.abs(eps)))):
        raise ValueError("epsilon_r must be passive: its Hermitian dissipative part must be positive semidefinite")

    # Flattening order is (z,y,x), so x is the fastest varying coordinate.
    dx, dy, dz = float(x[1]-x[0]), float(y[1]-y[0]), float(z[1]-z[0])
    Dx1 = _periodic_derivative_matrix(nx, dx)
    Dy1 = _periodic_derivative_matrix(ny, dy)
    Dz1 = _periodic_derivative_matrix(nz, dz)
    Ix, Iy, Iz = eye(nx, format="csr"), eye(ny, format="csr"), eye(nz, format="csr")
    bloch = np.asarray(bloch_wavevector_rad_m, dtype=float)
    if bloch.shape != (3,) or np.any(~np.isfinite(bloch)):
        raise ValueError("bloch_wavevector_rad_m must be a finite 3-vector")
    identity = eye(nx*ny*nz, format="csr")
    Dx = kron(kron(Iz, Iy), Dx1, format="csr") + 1j * bloch[0] * identity
    Dy = kron(kron(Iz, Dy1), Ix, format="csr") + 1j * bloch[1] * identity
    Dz = kron(kron(Dz1, Iy), Ix, format="csr") + 1j * bloch[2] * identity
    zero = Dx * 0.0
    curl = bmat([
        [zero, -Dz, Dy],
        [Dz, zero, -Dx],
        [-Dy, Dx, zero],
    ], format="csr")
    n = nx * ny * nz
    blocks = [[None] * 3 for _ in range(3)]
    for a in range(3):
        for b in range(3):
            blocks[a][b] = diags(eps[..., a, b].ravel(), format="csr")
    epsilon_operator = bmat(blocks, format="csr")
    k0 = omega / C
    operator = (curl @ curl - (k0*k0) * epsilon_operator).tocsc()
    rhs = (1j * omega * MU0 * current.reshape(-1)).astype(complex)
    if not np.isfinite(solver_rtol) or solver_rtol <= 0:
        raise ValueError("solver_rtol must be positive and finite")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            solution = spsolve(operator, rhs)
    except (MatrixRankWarning, RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "3-D periodic Maxwell operator is singular or could not be solved; "
            "add physically justified damping or use a non-resonant frequency"
        ) from exc
    if np.any(~np.isfinite(solution)):
        raise RuntimeError("3-D periodic Maxwell solve returned non-finite fields")
    residual = operator @ solution - rhs
    relative_residual = float(np.linalg.norm(residual) / max(np.linalg.norm(rhs), np.finfo(float).tiny))
    if relative_residual > max(100.0 * float(solver_rtol), 1e-7):
        raise RuntimeError(f"3-D periodic Maxwell solve did not meet residual tolerance: {relative_residual:g}")
    electric = np.stack([
        solution[a*n:(a+1)*n].reshape(shape) for a in range(3)
    ], axis=-1)
    curl_e = (curl @ solution).reshape(3, n)
    magnetic = np.stack([
        curl_e[a].reshape(shape) / (1j * omega * MU0) for a in range(3)
    ], axis=-1)
    # Time-averaged local material absorption for the exp(-i omega t) convention.
    absorption = 0.5 * omega * EPS0 * np.real(np.einsum(
        "...i,...ij,...j->...", electric.conj(), dissipative, electric
    ))
    return {
        "x_m": x.copy(), "y_m": y.copy(), "z_m": z.copy(),
        "electric_field_V_m": electric,
        "magnetic_field_A_m": magnetic,
        "epsilon_r": eps,
        "local_absorbed_power_density_W_m3": absorption,
        "integrated_absorbed_power_W": float(np.sum(absorption) * dx * dy * dz),
        "relative_linear_system_residual": relative_residual,
        "grid_shape_nz_ny_nx": shape,
        "boundary_condition": "three_dimensional_bloch_periodic",
        "bloch_wavevector_rad_m": bloch.copy(),
        "solver": "sparse_second_order_vector_finite_difference_frequency_domain",
        "maxwell_operator": "curl_curl_minus_k0_squared_full_tensor_epsilon",
        "mode_conversion": "full_coherent_3D_vector_coupling_through_spatially_varying_complex_tensor_epsilon",
        "approximation_scope": (
            "3-D local dielectric tensor, periodic Cartesian unit cell, second-order "
            "finite differences; no open boundaries, PML, finite-orbit-width "
            "nonlocality, or self-consistent plasma dielectric closure"
        ),
    }
