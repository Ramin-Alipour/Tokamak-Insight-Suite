"""Bindslev three-wave coupling for magnetized electromagnetic CTS.

The low-temperature magnetized-plasma source-current tensor follows
H. Bindslev, JET-P(93)41, Eqs. (98)-(102), (111), and (117)-(136),
and the compact correlated-fluctuation transfer contraction of Bindslev
(1996).  The derivation is intentionally exposed as tensor operations so
index conventions and assumptions can be tested independently.
"""
from __future__ import annotations

import numpy as np

EPS0 = 8.8541878128e-12
C = 299792458.0
E_CHARGE = 1.602176634e-19
ME = 9.1093837015e-31


def _v3(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite 3-vector")
    return value


def _chi(value, name):
    value = np.asarray(value, dtype=complex)
    if value.shape != (3, 3) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite complex 3x3 tensor")
    return value


def bindslev_gamma_tensor(omega_scattered_rad_s, magnetic_field_T):
    """Return Gamma_abij and its inverse, Bindslev Eqs. (98)-(101).

    Components are evaluated in a right-handed frame with B0 along +z.
    The sign of the electron cyclotron frequency follows q_e < 0.
    """
    omega = float(omega_scattered_rad_s)
    b = _v3(magnetic_field_T, "magnetic_field_T")
    bmag = float(np.linalg.norm(b))
    if not np.isfinite(omega) or omega == 0 or bmag == 0:
        raise ValueError("non-zero finite scattered frequency and magnetic field are required")
    omega_c = -E_CHARGE * bmag / ME
    Omega = omega_c / omega
    eta = np.diag([1.0, 1.0, 0.0])
    xi = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    gamma = np.zeros((3, 3, 3, 3), dtype=complex)
    for a in range(3):
        for bidx in range(3):
            for i in range(3):
                for j in range(3):
                    gamma[a, bidx, i, j] = (
                        (1.0 if a == i and bidx == j else 0.0)
                        + 1j * Omega * (
                            xi[a, i] * (1.0 if bidx == j else 0.0)
                            + (1.0 if a == i else 0.0) * xi[bidx, j]
                        )
                    )
    inverse = np.linalg.inv(gamma.reshape(9, 9)).reshape(3, 3, 3, 3)
    identity = np.einsum("ai,bj->abij", np.eye(3), np.eye(3))
    residual = np.einsum("ijab,abkl->ijkl", inverse, gamma)
    target = np.einsum("ik,jl->ijkl", np.eye(3), np.eye(3))
    error = float(np.max(np.abs(residual - target)))
    if error > 1e-9:
        raise ArithmeticError("Bindslev Gamma inverse failed its defining tensor identity")
    return {"Gamma": gamma, "Gamma_inverse": inverse, "Omega": Omega,
            "inverse_identity_residual": error}


def bindslev_x_tensor(omega_scattered_rad_s, wavevector_scattered_rad_m,
                      magnetic_field_T):
    """Compute Bindslev rank-three X tensor from the inverse Gamma tensor.

    Implements the contraction in Bindslev JET-P(93)41 Eq. (102), equivalent
    to the X definition in Eq. (53). The wavevector is transformed to a frame
    with B0 along z and k_sigma in the x-z plane before contraction.
    """
    omega = float(omega_scattered_rad_s)
    k_lab = _v3(wavevector_scattered_rad_m, "wavevector_scattered_rad_m")
    b_lab = _v3(magnetic_field_T, "magnetic_field_T")
    bmag = np.linalg.norm(b_lab)
    if omega == 0 or bmag == 0:
        raise ValueError("non-zero scattered frequency and magnetic field are required")
    zhat = b_lab / bmag
    kperp_vec = k_lab - np.dot(k_lab, zhat) * zhat
    if np.linalg.norm(kperp_vec) > 1e-14 * np.linalg.norm(k_lab):
        xhat = kperp_vec / np.linalg.norm(kperp_vec)
    else:
        trial = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(trial, zhat)) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        xhat = trial - np.dot(trial, zhat) * zhat
        xhat /= np.linalg.norm(xhat)
    yhat = np.cross(zhat, xhat)
    rotation = np.column_stack((xhat, yhat, zhat))
    k = rotation.T @ k_lab
    tensors = bindslev_gamma_tensor(omega, b_lab)
    inverse = tensors["Gamma_inverse"]
    x_local = np.zeros((3, 3, 3), dtype=complex)
    # Eq. (102): X_ikl/c = (k_j/omega) Gamma^{-1}_{ijab}
    # (delta_ak delta_bl + delta_al delta_bk).
    for i in range(3):
        for kidx in range(3):
            for l in range(3):
                contraction = 0.0j
                for j in range(3):
                    for a in range(3):
                        for bidx in range(3):
                            sym = ((1.0 if a == kidx and bidx == l else 0.0)
                                   + (1.0 if a == l and bidx == kidx else 0.0))
                            contraction += (k[j] / omega) * inverse[i, j, a, bidx] * sym
                x_local[i, kidx, l] = C * contraction
    # Three covariant Cartesian indices transform with one R on each index.
    x_lab = np.einsum("ia,jb,kc,abc->ijk", rotation, rotation, rotation, x_local)
    return {"X": x_lab, "X_B_aligned": x_local, "rotation_lab_from_B": rotation,
            "Gamma": tensors["Gamma"], "Gamma_inverse": inverse,
            "Omega": tensors["Omega"]}


def bindslev_source_current_operator(
    *, omega_incident_rad_s, omega_scattered_rad_s,
    wavevector_incident_rad_m, wavevector_scattered_rad_m,
    incident_electric_field, incident_susceptibility,
    scattered_susceptibility, magnetic_field_T, electron_density_m3,
):
    """Return the exact low-temperature Bindslev linear map from [n,v,E,B]
    fluctuation amplitudes to the scattered source-current vector.

    Implements Eq. (111), with incident-wave perturbations eliminated using
    Eqs. (114)-(116). Fluctuation vector ordering is
    [delta_n, delta_v_x,delta_v_y,delta_v_z, delta_E_x..z, delta_B_x..z].
    """
    wi = float(omega_incident_rad_s)
    ws = float(omega_scattered_rad_s)
    ne = float(electron_density_m3)
    ki = _v3(wavevector_incident_rad_m, "wavevector_incident_rad_m")
    ks = _v3(wavevector_scattered_rad_m, "wavevector_scattered_rad_m")
    ei = np.asarray(incident_electric_field, dtype=complex)
    if ei.shape != (3,) or not np.all(np.isfinite(ei)):
        raise ValueError("incident_electric_field must be a finite complex 3-vector")
    chi_i = _chi(incident_susceptibility, "incident_susceptibility")
    chi_s = _chi(scattered_susceptibility, "scattered_susceptibility")
    b0 = _v3(magnetic_field_T, "magnetic_field_T")
    if wi <= 0 or ws == 0 or ne <= 0:
        raise ValueError("positive incident frequency/density and non-zero scattered frequency required")
    # B_i = (k_i x E_i)/omega_i, v_i = -i omega_i eps0 chi_i E_i/q,
    # n_i = -i eps0 k_i.chi_i.E_i/q.
    bi = np.cross(ki, ei) / wi
    vi = -1j * wi * EPS0 / (-E_CHARGE) * (chi_i @ ei)
    ni = -1j * EPS0 / (-E_CHARGE) * np.dot(ki, chi_i @ ei)
    xdata = bindslev_x_tensor(ws, ks, b0)
    x = xdata["X"]
    eps = np.zeros((3, 3, 3), dtype=float)
    eps[0, 1, 2] = eps[1, 2, 0] = eps[2, 0, 1] = 1.0
    eps[0, 2, 1] = eps[1, 0, 2] = eps[2, 1, 0] = -1.0
    operator = np.zeros((3, 10), dtype=complex)
    prefactor = -1j * ws * EPS0 / ne
    # delta_n term.
    operator[:, 0] = chi_s @ ei
    # delta_v terms: chi_s . (delta_v x B_i) + X:(delta_v E_i)/c
    for j in range(3):
        unit = np.eye(3)[j]
        operator[:, 1 + j] = chi_s @ np.cross(unit, bi)
        operator[:, 1 + j] += np.einsum("il,l->i", x[:, j, :], ei) / C
    # delta_E terms: chi_s . (n_i delta_E) + X:(v_i delta_E)/c
    for l in range(3):
        unit = np.eye(3)[l]
        operator[:, 4 + l] = chi_s @ (ni * unit)
        operator[:, 4 + l] += np.einsum("ij,j->i", x[:, :, l], vi) / C
    # delta_B terms: chi_s . (v_i x delta_B)
    for n in range(3):
        unit = np.eye(3)[n]
        operator[:, 7 + n] = chi_s @ np.cross(vi, unit)
    operator *= prefactor
    return {
        "source_current_operator": operator,
        "incident_density_perturbation": ni,
        "incident_particle_flux_m_s_m3": vi,
        "incident_magnetic_field": bi,
        "X_tensor": x,
        "X_tensor_B_aligned": xdata["X_B_aligned"],
        "Gamma_tensor": xdata["Gamma"],
        "Gamma_inverse_tensor": xdata["Gamma_inverse"],
        "fluctuation_order": ["density", "particle_flux_x", "particle_flux_y",
            "particle_flux_z", "electric_x", "electric_y", "electric_z",
            "magnetic_x", "magnetic_y", "magnetic_z"],
        "formalism": "Bindslev JET-P(93)41 Eqs. (98)-(102), (111), (114)-(116)",
        "scope": "low-temperature magnetized source-current closure; relativistic source-current kernel is separate",
    }


def bindslev_correlated_transfer_spectrum(
    fluctuation_covariance, source_current_operator, receiver_polarization,
):
    """Contract all n,v,E,B auto- and cross-spectra with the Bindslev source.

    covariance is a 10x10 Hermitian PSD matrix in [n,v,E,B] ordering.
    """
    covariance = np.asarray(fluctuation_covariance, dtype=complex)
    operator = np.asarray(source_current_operator, dtype=complex)
    receiver = np.asarray(receiver_polarization, dtype=complex)
    if covariance.shape != (10, 10) or operator.shape != (3, 10) or receiver.shape != (3,):
        raise ValueError("expected covariance (10,10), source operator (3,10), receiver (3,)")
    if not np.all(np.isfinite(covariance)) or not np.all(np.isfinite(operator)):
        raise ValueError("covariance and source-current operator must be finite")
    if not np.allclose(covariance, covariance.conj().T, rtol=1e-8, atol=1e-20):
        raise ValueError("fluctuation covariance must be Hermitian")
    effective = receiver.conj() @ operator
    power = effective @ covariance @ effective.conj().T
    return {
        "received_source_current_spectral_power": float(max(np.real(power), 0.0)),
        "complex_transfer_amplitude": effective,
        "transfer_contributions": {
            "density": effective[0:1] @ covariance[0:1, :] @ effective.conj(),
            "particle_flux": effective[1:4] @ covariance[1:4, :] @ effective.conj(),
            "electric_field": effective[4:7] @ covariance[4:7, :] @ effective.conj(),
            "magnetic_field": effective[7:10] @ covariance[7:10, :] @ effective.conj(),
        },
        "includes_all_cross_correlations": True,
    }
