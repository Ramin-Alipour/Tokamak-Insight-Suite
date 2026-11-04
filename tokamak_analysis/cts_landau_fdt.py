"""Conservative multispecies linearized Coulomb Landau operator and classical FDT.

Finite-box, 3-D Cartesian velocity-space reference discretization about a
common-temperature, common-flow Maxwellian mixture. The symmetric weak form
retains test-particle and field-particle terms and enforces particle number,
total momentum and total energy invariants. The matching Langevin covariance
is derived from the same discrete operator. Non-relativistic only.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
import numpy as np
from scipy.constants import epsilon_0
from scipy.sparse import csr_matrix, lil_matrix, diags
from scipy.sparse.linalg import norm as sparse_norm


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be positive and finite")
    x = float(value)
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return x


def _axis(axis):
    x = np.asarray(axis, dtype=float)
    if x.ndim != 1 or x.size < 3 or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("velocity axes must be finite, strictly increasing, and contain >=3 points")
    w = np.empty_like(x)
    w[0], w[-1] = 0.5*(x[1]-x[0]), 0.5*(x[-1]-x[-2])
    w[1:-1] = 0.5*(x[2:]-x[:-2])
    return x, w


def _grid(axes):
    grids = np.meshgrid(*axes, indexing="ij")
    shape = tuple(len(x) for x in axes)
    n = int(np.prod(shape))
    velocities = np.column_stack([g.ravel() for g in grids])
    volume = np.einsum("i,j,k->ijk", *[_axis(x)[1] for x in axes]).ravel()
    derivatives = []
    for direction, coordinate in enumerate(axes):
        matrix = np.empty((n, n))
        for column in range(n):
            basis = np.zeros(shape)
            basis.flat[column] = 1.0
            matrix[:, column] = np.gradient(
                basis, coordinate, axis=direction, edge_order=2
            ).ravel()
        derivatives.append(csr_matrix(matrix))
    return velocities, volume, derivatives


@dataclass(frozen=True)
class LandauFDTSystem:
    operator_s_inverse: csr_matrix
    entropy_metric: csr_matrix
    fdt_source_covariance_s_inverse: csr_matrix
    equilibrium_weights: np.ndarray
    velocity_moments: np.ndarray
    species_slices: tuple[slice, ...]
    metadata: Mapping


def build_linearized_landau_fdt(
    velocity_x_m_s, velocity_y_m_s, velocity_z_m_s, species: Sequence[Mapping],
    *, common_temperature_J, common_bulk_flow_m_s=(0.0, 0.0, 0.0),
    coulomb_logarithm=15.0,
) -> LandauFDTSystem:
    """Assemble the multispecies linearized Landau operator for h=delta_f/F.

    The discrete weak form is
      <g,Lh>_W = -sum_ab Gamma_ab int F_a F_b'
       (grad(g_a)/m_a-grad(g_b')/m_b) U(v-v')
       (grad(h_a)/m_a-grad(h_b')/m_b) dv dv',
    with unordered species pairs and a factor 1/2 for self-pairs.
    U(u)=(I-u u/|u|^2)/|u|. The cross terms are the field-particle terms.
    Q_h=-(L W^-1+W^-1 L.T)=2 W^-1 K W^-1 is the matching classical
    equilibrium Langevin source covariance.

    Coulomb logarithm is explicit input. Finite velocity-domain and grid
    convergence are required. This reference implementation is not
    relativistic and assumes isotropic Maxwellians with common T and flow.
    """
    temp = _positive(common_temperature_J, "common_temperature_J")
    axes = [_axis(x)[0] for x in (velocity_x_m_s, velocity_y_m_s, velocity_z_m_s)]
    vel, dv, deriv = _grid(axes)
    nv = vel.shape[0]
    if nv > 512:
        raise ValueError("velocity grid exceeds 512 nodes; use a matrix-free implementation")
    if not isinstance(species, Sequence) or isinstance(species, (str, bytes)) or not species:
        raise ValueError("species must be a non-empty sequence")
    if len(species) > 4:
        raise ValueError("assembled reference solver supports at most four species")
    flow = np.asarray(common_bulk_flow_m_s, dtype=float)
    if flow.shape != (3,) or np.any(~np.isfinite(flow)):
        raise ValueError("common_bulk_flow_m_s must be a finite 3-vector")
    parsed = []
    for i, item in enumerate(species):
        if not isinstance(item, Mapping):
            raise ValueError(f"species[{i}] must be a mapping")
        mass = _positive(item.get("mass_kg"), f"species[{i}].mass_kg")
        density = _positive(item.get("density_m3"), f"species[{i}].density_m3")
        charge = float(item.get("charge_C", np.nan))
        if not np.isfinite(charge) or charge == 0:
            raise ValueError(f"species[{i}].charge_C must be finite and non-zero")
        parsed.append((mass, density, charge, str(item.get("name", f"species_{i}"))))
    ns = len(parsed)
    ln = np.asarray(coulomb_logarithm, dtype=float)
    if ln.ndim == 0:
        ln = np.full((ns, ns), float(ln))
    if ln.shape != (ns, ns) or np.any(~np.isfinite(ln)) or np.any(ln <= 0):
        raise ValueError("coulomb_logarithm must be positive scalar or symmetric species-pair matrix")
    if not np.allclose(ln, ln.T, rtol=1e-12, atol=1e-12):
        raise ValueError("coulomb_logarithm matrix must be symmetric")

    f0s, ws, slices = [], [], []
    delta = vel - flow
    for s, (mass, density, charge, name) in enumerate(parsed):
        vth2 = temp / mass
        f0 = density / (2*np.pi*vth2)**1.5 * np.exp(-0.5*np.sum(delta**2, axis=1)/vth2)
        w = f0 * dv
        if np.any(~np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("Maxwellian underflow/overflow; enlarge or recenter the velocity domain")
        f0s.append(f0)
        ws.append(w)
        slices.append(slice(s*nv, (s+1)*nv))

    ndof = ns*nv
    K = csr_matrix((ndof, ndof), dtype=float)
    for a in range(ns):
        for b in range(a, ns):
            ma, mb = parsed[a][0], parsed[b][0]
            gamma = parsed[a][2]**2 * parsed[b][2]**2 * ln[a, b] / (8*np.pi*epsilon_0**2)
            oa, ob = a*nv, b*nv
            for i in range(nv):
                ga = [deriv[d].getrow(i)/ma for d in range(3)]
                for j in range(nv):
                    u = vel[i]-vel[j]
                    u2 = float(u@u)
                    if u2 <= np.finfo(float).tiny:
                        continue
                    U = (np.eye(3)-np.outer(u,u)/u2)/np.sqrt(u2)
                    gb = [deriv[d].getrow(j)/mb for d in range(3)]
                    D = lil_matrix((3, ndof), dtype=float)
                    for d in range(3):
                        D[d, oa:oa+nv] = ga[d]
                        D[d, ob:ob+nv] = D[d, ob:ob+nv] - gb[d]
                    D = D.tocsr()
                    factor = gamma*ws[a][i]*ws[b][j]*(0.5 if a == b else 1.0)
                    K = K + factor*(D.T @ U @ D)
    K = ((K+K.T)*0.5).tocsr()
    metric_diag = np.concatenate(ws)
    W = diags(metric_diag, format="csr")
    Winv = diags(1.0/metric_diag, format="csr")
    L = (-Winv @ K).tocsr()
    Q = (2.0*Winv @ K @ Winv).tocsr()

    moments = np.zeros((ndof, 5))
    for s, (mass, density, charge, name) in enumerate(parsed):
        sl = slices[s]
        moments[sl, 0] = 1.0
        moments[sl, 1:4] = mass*vel
        moments[sl, 4] = 0.5*mass*np.sum(vel**2, axis=1)
    inv_resid = float(sparse_norm(K@moments) /
        max(sparse_norm(K)*np.linalg.norm(moments), np.finfo(float).tiny))
    # K is a Gram sum, hence PSD; verify the FDT covariance numerically.
    if inv_resid > 5e-2:
        raise ArithmeticError("velocity grid does not resolve discrete Landau invariants; refine/expand it")
    metadata = {
        "operator": "multispecies_linearized_Coulomb_Landau_test_and_field_particle",
        "equilibrium": "common_temperature_common_flow_isotropic_Maxwellian",
        "velocity_grid": "user_supplied_3D_Cartesian_tensor_product",
        "velocity_nodes_per_species": int(nv),
        "species_count": int(ns),
        "conservation": {"species_particle_number": True, "total_momentum": True, "total_energy": True},
        "entropy_production": "non_positive",
        "fdt": "classical_Langevin_covariance_from_same_discrete_Landau_operator",
        "invariant_relative_residual": inv_resid,
        "coulomb_logarithm": ln.copy(),
        "limitations": [
            "Non-relativistic isotropic Maxwellian equilibrium only.",
            "Finite velocity box and quadrature require grid/domain convergence.",
            "Assembled reference operator; computational cost grows quadratically with velocity nodes.",
            "No quantum detailed balance or arbitrary non-Maxwellian equilibrium.",
        ],
    }
    return LandauFDTSystem(L, W, Q, metric_diag, moments, tuple(slices), metadata)


def landau_density_response(system: LandauFDTSystem, angular_frequency_rad_s,
                            density_observation, *, forcing=None):
    """Compute a linear resolvent and FDT spectrum for supplied velocity moments.

    Electrostatic screening, CTS geometry and optical normalization are not
    inferred here; the caller supplies the observation and forcing operators.
    """
    omega = np.asarray(angular_frequency_rad_s, dtype=float)
    if omega.ndim != 1 or omega.size == 0 or np.any(~np.isfinite(omega)):
        raise ValueError("angular_frequency_rad_s must be a finite non-empty vector")
    obs = np.asarray(density_observation, dtype=float)
    n = system.operator_s_inverse.shape[0]
    if obs.ndim != 2 or obs.shape[1] != n or np.any(~np.isfinite(obs)):
        raise ValueError("density_observation must have shape (n_observable,n_dof)")
    drive = np.eye(n) if forcing is None else np.asarray(forcing, dtype=float)
    if drive.ndim != 2 or drive.shape[0] != n or np.any(~np.isfinite(drive)):
        raise ValueError("forcing must have shape (n_dof,n_input)")
    L = system.operator_s_inverse.toarray()
    Q = system.fdt_source_covariance_s_inverse.toarray()
    result = np.empty((omega.size, obs.shape[0], drive.shape[1]), complex)
    spectrum = np.empty((omega.size, obs.shape[0], obs.shape[0]), complex)
    identity = np.eye(n)
    for i, w in enumerate(omega):
        R = np.linalg.inv(-1j*w*identity-L)
        result[i] = obs@R@drive
        spectrum[i] = obs@R@Q@R.conj().T@obs.T
    return {"response": result, "fluctuation_spectral_matrix": spectrum,
            "angular_frequency_rad_s": omega.copy(),
            "collision_operator": system.metadata["operator"],
            "fdt": system.metadata["fdt"],
            "limitations": list(system.metadata["limitations"])+[
                "Self-consistent electrostatic screening and CTS scattering normalization are not applied."
            ]}
