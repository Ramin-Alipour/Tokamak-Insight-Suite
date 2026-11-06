"""Isothermal, momentum-conserving multispecies BGK longitudinal response.

This is a bounded collisional *linear-response kernel*, not a Landau operator
and not, by itself, a CTS dynamic-structure-factor implementation.  The
collision term conserves each species' particle number and total mixture
momentum.  It assumes a common equilibrium temperature and common bulk flow;
energy conservation and a fluctuation source/FDT closure are not included.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from scipy.special import roots_hermite


def conserving_bgk_longitudinal_susceptibility(
    wave_number_m_inverse: float,
    angular_frequency_rad_s,
    *,
    species_densities_m3: Sequence[float],
    species_masses_kg: Sequence[float],
    common_temperature_J: float,
    collision_frequencies_s_inverse: Sequence[float],
    bulk_flow_m_s: float = 0.0,
    quadrature_order: int = 96,
) -> dict:
    """Return the multispecies density-response matrix for an isothermal BGK mixture.

    The linearized operator is
      C_s = -nu_s [delta f_s - f0_s(delta n_s/n_s
                         + m_s (v-u) delta u/T)],
    where delta u is chosen so that sum_s int m_s(v-u) C_s dv = 0.
    Each species' number moment is conserved separately.  The model is
    isothermal (not energy conserving), nonrelativistic and one-dimensional
    in velocity.  A common equilibrium temperature and flow are mandatory.

    The returned matrix R[i,j] = delta(n_i)/delta(U_j), where U_j is an
    externally applied potential energy (J) coupled to species j. It is a
    bare response to prescribed external potentials; self-consistent
    electrostatic screening and the CTS fluctuation spectrum are not included.
    Do not convert it to CTS power without a separately derived and validated fluctuation-dissipation
    closure and Thomson-scattering normalization.
    """
    def positive_scalar(value, name):
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be positive and finite")
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
        return value

    k = positive_scalar(wave_number_m_inverse, "wave_number_m_inverse")
    temperature = positive_scalar(common_temperature_J, "common_temperature_J")
    omega = np.asarray(angular_frequency_rad_s, dtype=float)
    if omega.ndim != 1 or omega.size == 0 or not np.all(np.isfinite(omega)):
        raise ValueError("angular_frequency_rad_s must be a finite non-empty 1-D array")
    n = np.asarray(species_densities_m3, dtype=float)
    m = np.asarray(species_masses_kg, dtype=float)
    nu = np.asarray(collision_frequencies_s_inverse, dtype=float)
    if n.ndim != 1 or n.size == 0 or m.shape != n.shape or nu.shape != n.shape:
        raise ValueError("species density, mass and collision-frequency arrays must match")
    if (np.any(~np.isfinite(n)) or np.any(n <= 0) or np.any(~np.isfinite(m)) or np.any(m <= 0)
            or np.any(~np.isfinite(nu)) or np.any(nu < 0)):
        raise ValueError("species densities and masses must be positive and finite; collision frequencies must be finite and non-negative")
    if not np.any(nu > 0):
        # With every species collisionless, the conserving BGK correction is
        # identically absent. The same routine still evaluates the Vlasov
        # streaming response, but it must not divide by a zero total relaxation
        # weight when constructing the common momentum-restoring velocity.
        momentum_weights = np.zeros_like(nu)
    else:
        momentum_weights = nu * m * n
        momentum_weights = momentum_weights / np.sum(momentum_weights)
    if isinstance(quadrature_order, (bool, np.bool_)) or int(quadrature_order) != quadrature_order or quadrature_order < 24:
        raise ValueError("quadrature_order must be an integer >= 24")
    flow = float(bulk_flow_m_s)
    if not np.isfinite(flow):
        raise ValueError("bulk_flow_m_s must be finite")

    nodes, weights = roots_hermite(int(quadrature_order))
    ns = n.size
    response = np.empty((omega.size, ns, ns), dtype=complex)
    for iw, w in enumerate(omega):
        velocities = []
        denominators = []
        i0 = np.empty(ns, dtype=complex)
        i1 = np.empty(ns, dtype=complex)
        j0 = np.empty(ns, dtype=complex)
        j1 = np.empty(ns, dtype=complex)
        for s in range(ns):
            thermal_speed = np.sqrt(2.0 * temperature / m[s])
            v = flow + thermal_speed * nodes
            d = nu[s] - 1j * w + 1j * k * v
            if np.any(np.abs(d) <= np.finfo(float).tiny):
                raise FloatingPointError("singular BGK kinetic denominator")
            g0 = nu[s] / d
            dv = v - flow
            g1 = g0 * m[s] * dv / temperature
            velocities.append((v, dv, d))
            denominators.append(d)
            i0[s] = np.sum(weights * g0) / np.sqrt(np.pi)
            i1[s] = np.sum(weights * g1) / np.sqrt(np.pi)
            j0[s] = np.sum(weights * dv * g0) / np.sqrt(np.pi)
            j1[s] = np.sum(weights * dv * g1) / np.sqrt(np.pi)

        # Unknowns are all delta(n_s)/n_s followed by the common delta-u.
        matrix = np.zeros((ns + 1, ns + 1), dtype=complex)
        for s in range(ns):
            matrix[s, s] = 1.0 - i0[s]
            matrix[s, -1] = -i1[s]
        matrix[-1, :ns] = -momentum_weights * j0
        matrix[-1, -1] = 1.0 - np.sum(momentum_weights * j1)

        for driven in range(ns):
            rhs = np.zeros(ns + 1, dtype=complex)
            source_moments = np.zeros((ns, 2), dtype=complex)
            for s, (v, dv, d) in enumerate(velocities):
                # Force from U_driven exp[i(kx-wt)]: F=-ik U.
                source = np.zeros_like(v, dtype=complex)
                if s == driven:
                    source = -1j * k * dv / temperature
                g = source / d
                source_moments[s, 0] = np.sum(weights * g) / np.sqrt(np.pi)
                source_moments[s, 1] = np.sum(weights * dv * g) / np.sqrt(np.pi)
                rhs[s] = source_moments[s, 0]
            rhs[-1] = np.sum(momentum_weights * source_moments[:, 1])
            try:
                moments = np.linalg.solve(matrix, rhs)
            except np.linalg.LinAlgError as exc:
                raise ValueError("conserving BGK moment system is singular at the requested frequency") from exc
            response[iw, :, driven] = n * moments[:ns]

    return {
        "density_response_matrix_m3_per_J": response,
        "wave_number_m_inverse": k,
        "angular_frequency_rad_s": omega.copy(),
        "collision_frequencies_s_inverse": nu.copy(),
        "collisionless_species_indices": np.flatnonzero(nu == 0.0),
        "collision_regime": "collisionless_vlasov_limit" if np.all(nu == 0.0) else ("mixed_collisionless_collisional_bgk" if np.any(nu == 0.0) else "collisional_bgk"),
        "collision_operator": "isothermal_multispecies_momentum_conserving_BGK",
        "conservation": {
            "species_particle_number": True,
            "total_mixture_momentum": True,
            "total_energy": False,
        },
        "equilibrium_assumptions": {
            "common_temperature_J": temperature,
            "common_bulk_flow_m_s": flow,
            "nonrelativistic": True,
        },
        "quadrature_order": int(quadrature_order),
        "cts_integration_status": "response_kernel_only_no_FDT_fluctuation_source_or_Skomega",
        "limitations": [
            "BGK closure, not the velocity-dependent linearized Landau operator.",
            "Isothermal closure; energy conservation is not imposed.",
            "Common-temperature equilibrium only; unequal-temperature species are unsupported.",
            "No magnetic gyroharmonics, relativistic kinetics or Full-EM mode coupling.",
            "A CTS spectrum requires a consistent fluctuation source/FDT and scattering normalization.",
        ],
    }
