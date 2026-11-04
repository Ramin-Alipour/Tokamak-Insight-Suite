"""Polarized two-mode transfer driven by the magnetized kinetic ECE conductivity."""
from __future__ import annotations
import numpy as np
from scipy.constants import Boltzmann, elementary_charge
from scipy.linalg import expm
from .ece_kinetic import _species_conductivity, ME, E_CHARGE


def kinetic_ece_absorption_matrix(
    frequency_Hz, wavevector_rad_m, magnetic_field_T, *,
    electron_density_m3, electron_temperature_eV,
    mode_electric_field_unit_flux, harmonic_order=8,
    quadrature_order=48, damping_rate_rad_s=0.0,
):
    """Project the kinetic conductivity onto two supplied unit-flux eigenmodes."""
    basis = np.asarray(mode_electric_field_unit_flux, dtype=complex)
    k = np.asarray(wavevector_rad_m, dtype=float)
    b = np.asarray(magnetic_field_T, dtype=float)
    f, ne, te = map(float, (frequency_Hz, electron_density_m3, electron_temperature_eV))
    if basis.shape != (3, 2) or np.any(~np.isfinite(basis)) or np.linalg.matrix_rank(basis) != 2:
        raise ValueError("mode_electric_field_unit_flux must be a finite rank-2 (3,2) matrix")
    if k.shape != (3,) or b.shape != (3,) or np.any(~np.isfinite(k)) or np.any(~np.isfinite(b)):
        raise ValueError("wavevector and magnetic field must be finite 3-vectors")
    if np.linalg.norm(k) == 0 or np.linalg.norm(b) == 0:
        raise ValueError("wavevector and magnetic field must be non-zero")
    if not all(np.isfinite(x) and x > 0 for x in (f, ne, te)):
        raise ValueError("frequency, density and temperature must be positive and finite")
    damping = float(damping_rate_rad_s)
    if not np.isfinite(damping) or damping < 0:
        raise ValueError("damping_rate_rad_s must be finite and non-negative")
    if isinstance(harmonic_order, (bool, np.bool_)) or int(harmonic_order) != harmonic_order or harmonic_order < 1:
        raise ValueError("harmonic_order must be a positive integer")
    if isinstance(quadrature_order, (bool, np.bool_)) or int(quadrature_order) != quadrature_order or quadrature_order < 16:
        raise ValueError("quadrature_order must be an integer >= 16")
    species = {"name": "e-", "density": ne, "temperature": te,
               "mass": ME, "charge": -E_CHARGE, "flow": np.zeros(3)}
    sigma = _species_conductivity(
        2*np.pi*f, k, b, species, harmonic_order=int(harmonic_order),
        quadrature_order=int(quadrature_order), damping_rate_rad_s=damping,
    )
    sigma_h = 0.5*(sigma+sigma.conj().T)
    opacity = 0.5*basis.conj().T@sigma_h@basis
    opacity = 0.5*(opacity+opacity.conj().T)
    eig = np.linalg.eigvalsh(opacity)
    scale = max(float(np.max(np.abs(eig))), np.finfo(float).tiny)
    if eig.min() < -1e-9*scale:
        raise ValueError("kinetic absorption matrix contains net gain")
    if eig.min() < 0:
        values, vectors = np.linalg.eigh(opacity)
        opacity = (vectors*np.maximum(values, 0.0))@vectors.conj().T
    return {
        "absorption_matrix_m_inverse": opacity,
        "conductivity_S_m": sigma,
        "mode_electric_field_unit_flux": basis,
        "model": "magnetized_nonrelativistic_Maxwellian_linear_Vlasov_conductivity",
        "cross_mode_absorption": True,
        "limitations": [
            "The supplied basis columns must be orthogonal local eigenmodes normalized to unit normal power flux.",
            "No relativistic or conserving collisional Landau correction.",
        ],
    }


def integrate_mode_coupled_kinetic_ece_transfer(
    frequency_Hz, segment_length_m, absorption_matrix_m_inverse,
    electron_temperature_eV, *, mode_coupling_matrix_rad_m=None,
    incident_coherency_K=None,
):
    """Exact piecewise-constant polarized LTE transfer for two coherently coupled modes.

    The matrix equation is dJ/ds=M J+J M^H+A*T_RJ, with M=-A/2-iH.
    A is the Hermitian PSD kinetic absorption matrix; H is a Hermitian
    mode-coupling generator. A 5x5 augmented exponential gives each segment's
    exact update, retaining cross-polarization coherence.
    """
    f = np.asarray(frequency_Hz, dtype=float)
    if f.ndim == 0:
        f = f.reshape(1)
    lengths = np.asarray(segment_length_m, dtype=float)
    te = np.asarray(electron_temperature_eV, dtype=float)
    A = np.asarray(absorption_matrix_m_inverse, dtype=complex)
    if f.ndim != 1 or f.size == 0 or np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequency_Hz must be finite and positive")
    if lengths.ndim != 1 or lengths.size == 0 or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("segment_length_m must be a positive vector")
    if te.shape != lengths.shape or np.any(~np.isfinite(te)) or np.any(te <= 0):
        raise ValueError("electron_temperature_eV must have one positive value per segment")
    if A.shape == (lengths.size, 2, 2):
        A = np.broadcast_to(A[None], (f.size,)+A.shape).copy()
    if A.shape != (f.size, lengths.size, 2, 2) or np.any(~np.isfinite(A)):
        raise ValueError("absorption matrix has an invalid shape or non-finite values")
    H = np.zeros_like(A) if mode_coupling_matrix_rad_m is None else np.asarray(mode_coupling_matrix_rad_m, complex)
    if H.shape == (lengths.size, 2, 2):
        H = np.broadcast_to(H[None], A.shape).copy()
    if H.shape != A.shape or np.any(~np.isfinite(H)):
        raise ValueError("mode coupling matrix must match the absorption matrix shape")
    for idx in np.ndindex(A.shape[:2]):
        if not np.allclose(A[idx], A[idx].conj().T, rtol=1e-9, atol=1e-12):
            raise ValueError("absorption matrix must be Hermitian")
        if not np.allclose(H[idx], H[idx].conj().T, rtol=1e-9, atol=1e-12):
            raise ValueError("mode coupling matrix must be Hermitian")
        eig = np.linalg.eigvalsh(A[idx])
        if eig.min() < -1e-9*max(float(np.max(np.abs(eig))), np.finfo(float).tiny):
            raise ValueError("absorption matrix must be positive semidefinite")
    if incident_coherency_K is None:
        J0 = np.zeros((f.size, 2, 2), complex)
    else:
        J0 = np.asarray(incident_coherency_K, complex)
        if J0.shape == (2, 2):
            J0 = np.broadcast_to(J0[None], (f.size, 2, 2)).copy()
        if J0.shape != (f.size, 2, 2) or np.any(~np.isfinite(J0)):
            raise ValueError("incident_coherency_K must be (2,2) or (n_frequency,2,2)")
        for j in J0:
            if not np.allclose(j, j.conj().T, rtol=1e-9, atol=1e-12) or np.linalg.eigvalsh(j).min() < -1e-10:
                raise ValueError("incident coherency must be Hermitian positive semidefinite")
    result = J0.copy()
    eye2 = np.eye(2, dtype=complex)
    for fi in range(f.size):
        for si, length in enumerate(lengths):
            a, h = A[fi, si], H[fi, si]
            m = -0.5*a-1j*h
            generator = np.kron(eye2, m)+np.kron(m.conj(), eye2)
            source = a*(te[si]*elementary_charge/Boltzmann)
            augmented = np.zeros((5, 5), complex)
            augmented[:4, :4] = generator
            augmented[:4, 4] = source.reshape(4, order="F")
            transfer = expm(augmented*length)
            state = np.r_[result[fi].reshape(4, order="F"), 1.0]
            result[fi] = (transfer@state)[:4].reshape((2, 2), order="F")
            result[fi] = 0.5*(result[fi]+result[fi].conj().T)
    return {
        "frequency_Hz": f.copy(),
        "modal_coherency_out_K": result,
        "brightness_temperature_out_K": np.real(np.trace(result, axis1=1, axis2=2))/2,
        "absorption_matrix_m_inverse": A,
        "mode_coupling_matrix_rad_m": H,
        "transfer_model": "exact_piecewise_constant_two_mode_polarized_LTE_transfer",
        "mode_conversion": "coherent_two_mode_coupling_retained",
        "limitations": [
            "Local mode basis and coupling generator are supplied by the caller.",
            "LTE Rayleigh-Jeans source function is assumed.",
            "Relativistic/collisional emissivity and arbitrary 3-D ray/interface coupling are not included.",
        ],
    }



def solve_two_mode_kinetic_ece_profile(
    frequency_Hz,
    segment_length_m,
    *,
    wavevector_rad_m,
    magnetic_field_T,
    electron_density_m3,
    electron_temperature_eV,
    mode_electric_field_unit_flux,
    mode_coupling_matrix_rad_m=None,
    incident_coherency_K=None,
    harmonic_order=8,
    quadrature_order=48,
    damping_rate_rad_s=0.0,
):
    """Connect local kinetic absorption to coherent two-mode ECE transfer.

    Every segment/frequency receives a supplied local wavevector, magnetic
    field and two-column unit-normal-flux eigenmode basis. The magnetized
    Maxwellian conductivity is projected to a Hermitian 2x2 opacity matrix;
    the ordered matrix transfer then retains cross-polarization coherence
    and an optional Hermitian mode-coupling generator.

    Geometry/eigenmodes are explicit inputs: this routine does not fabricate
    a ray, infer a mode through a cutoff, or replace the Berreman boundary
    solver. The local kinetic kernel remains non-relativistic Maxwellian.
    """
    f = np.asarray(frequency_Hz, dtype=float)
    if f.ndim == 0:
        f = f.reshape(1)
    if f.ndim != 1 or f.size == 0 or np.any(~np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequency_Hz must be a positive finite scalar or vector")
    lengths = np.asarray(segment_length_m, dtype=float)
    if lengths.ndim != 1 or lengths.size == 0 or np.any(~np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("segment_length_m must be a non-empty positive finite vector")
    nseg, nf = lengths.size, f.size

    def vector_stack(value, name):
        a = np.asarray(value, dtype=float)
        if a.shape == (nseg, 3):
            a = np.broadcast_to(a[None, :, :], (nf, nseg, 3))
        if a.shape != (nf, nseg, 3) or np.any(~np.isfinite(a)):
            raise ValueError(f"{name} must have shape (n_segment,3) or (n_frequency,n_segment,3)")
        return a

    def mode_stack(value):
        a = np.asarray(value, dtype=complex)
        if a.shape == (nseg, 3, 2):
            a = np.broadcast_to(a[None, :, :, :], (nf, nseg, 3, 2))
        if a.shape != (nf, nseg, 3, 2) or np.any(~np.isfinite(a.real)) or np.any(~np.isfinite(a.imag)):
            raise ValueError("mode_electric_field_unit_flux must have shape (n_segment,3,2) or (n_frequency,n_segment,3,2)")
        return a

    kstack = vector_stack(wavevector_rad_m, "wavevector_rad_m")
    bstack = vector_stack(magnetic_field_T, "magnetic_field_T")
    basis = mode_stack(mode_electric_field_unit_flux)
    ne = np.asarray(electron_density_m3, dtype=float)
    te = np.asarray(electron_temperature_eV, dtype=float)
    if ne.shape != (nseg,) or np.any(~np.isfinite(ne)) or np.any(ne <= 0):
        raise ValueError("electron_density_m3 must contain one positive finite value per segment")
    if te.shape != (nseg,) or np.any(~np.isfinite(te)) or np.any(te <= 0):
        raise ValueError("electron_temperature_eV must contain one positive finite value per segment")

    opacity = np.empty((nf, nseg, 2, 2), dtype=complex)
    local = []
    for i in range(nf):
        row = []
        for j in range(nseg):
            result = kinetic_ece_absorption_matrix(
                f[i], kstack[i, j], bstack[i, j],
                electron_density_m3=ne[j],
                electron_temperature_eV=te[j],
                mode_electric_field_unit_flux=basis[i, j],
                harmonic_order=harmonic_order,
                quadrature_order=quadrature_order,
                damping_rate_rad_s=damping_rate_rad_s,
            )
            opacity[i, j] = result["absorption_matrix_m_inverse"]
            row.append(result)
        local.append(row)
    transfer = integrate_mode_coupled_kinetic_ece_transfer(
        f, lengths, opacity, te,
        mode_coupling_matrix_rad_m=mode_coupling_matrix_rad_m,
        incident_coherency_K=incident_coherency_K,
    )
    return {
        **transfer,
        "local_kinetic_absorption": local,
        "mode_electric_field_unit_flux": basis,
        "profile_solver": "magnetized_kinetic_Maxwellian_opacity_plus_coherent_two_mode_LTE_transfer",
        "geometry_source": "explicit_user_supplied_local_wavevectors_and_unit_flux_eigenmodes",
        "limitations": [
            "Local mode basis must be computed consistently with the supplied dielectric and normalized to unit normal power flux.",
            "Ray trajectory and mode basis are not automatically derived; use the oblique Berreman solver for 1-D stratified interfaces.",
            "Non-relativistic Maxwellian response and classical LTE transfer only.",
            "No arbitrary 3-D anisotropic mode conversion or collisional Landau dielectric is included.",
        ],
    }
