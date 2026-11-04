"""Conservative multispecies linearized Coulomb Landau operator and classical FDT.

Finite-box, 3-D Cartesian normalized-velocity reference discretization about a
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
from scipy.sparse.linalg import norm as sparse_norm, splu


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
    velocity_grid_m_s: np.ndarray
    metadata: Mapping


def build_linearized_landau_fdt(
    normalized_velocity_x, normalized_velocity_y, normalized_velocity_z, species: Sequence[Mapping],
    *, common_temperature_J, common_bulk_flow_m_s=(0.0, 0.0, 0.0),
    coulomb_logarithm=15.0,
) -> LandauFDTSystem:
    """Assemble the multispecies linearized Landau operator for h=delta_f/F.

    The three input axes are dimensionless velocity coordinates in units of each
    species thermal speed sqrt(T/m); each species is mapped to its own physical
    velocity grid, avoiding the unresolved electron/ion thermal-speed mismatch.

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
    axes = [_axis(x)[0] for x in (normalized_velocity_x, normalized_velocity_y, normalized_velocity_z)]
    xi, dv_normalized, deriv_normalized = _grid(axes)
    nv = xi.shape[0]
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

    f0s, ws, slices, velocity_grids, derivative_grids = [], [], [], [], []
    for s, (mass, density, charge, name) in enumerate(parsed):
        vth = np.sqrt(temp/mass)
        velocities = flow + vth*xi
        f0 = density / (2*np.pi*vth**2)**1.5 * np.exp(-0.5*np.sum(xi**2, axis=1))
        w = f0 * (vth**3*dv_normalized)
        velocity_grids.append(velocities)
        derivative_grids.append([d/vth for d in deriv_normalized])
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
                ga = [derivative_grids[a][d].getrow(i)/ma for d in range(3)]
                for j in range(nv):
                    u = velocity_grids[a][i]-velocity_grids[b][j]
                    u2 = float(u@u)
                    if u2 <= np.finfo(float).tiny:
                        continue
                    U = (np.eye(3)-np.outer(u,u)/u2)/np.sqrt(u2)
                    gb = [derivative_grids[b][d].getrow(j)/mb for d in range(3)]
                    D = lil_matrix((3, ndof), dtype=float)
                    for d in range(3):
                        D[d, oa:oa+nv] = ga[d]
                        D[d, ob:ob+nv] = D[d, ob:ob+nv] - gb[d]
                    D = D.tocsr()
                    factor = gamma*ws[a][i]*ws[b][j]*(0.5 if a == b else 1.0)
                    K = K + factor*(D.T @ csr_matrix(U) @ D)
    K = ((K+K.T)*0.5).tocsr()
    metric_diag = np.concatenate(ws)
    W = diags(metric_diag, format="csr")
    Winv = diags(1.0/metric_diag, format="csr")
    L = (-Winv @ K).tocsr()
    Q = (2.0*Winv @ K @ Winv).tocsr()

    moments = np.zeros((ndof, ns + 4))
    for s, (mass, density, charge, name) in enumerate(parsed):
        sl = slices[s]
        moments[sl, s] = 1.0
        moments[sl, ns:ns+3] = mass*velocity_grids[s]
        moments[sl, ns+3] = 0.5*mass*np.sum(velocity_grids[s]**2, axis=1)
    inv_resid = float(sparse_norm(K@moments) /
        max(sparse_norm(K)*np.linalg.norm(moments), np.finfo(float).tiny))
    # K is a Gram sum, hence PSD; verify the FDT covariance numerically.
    # The metadata below advertises exact discrete conservation.  A loose
    # percent-level tolerance would let a poorly resolved grid pass while
    # still labelling number, momentum and energy as conserved.  The chosen
    # finite-difference stencil differentiates the invariant polynomials
    # exactly on supported grids; only round-off-scale residual is accepted.
    invariant_tolerance = 1.0e-8
    if not np.isfinite(inv_resid) or inv_resid > invariant_tolerance:
        raise ArithmeticError(
            "velocity grid does not resolve discrete Landau invariants "
            f"(relative residual={inv_resid:.3e}, tolerance={invariant_tolerance:.1e}); "
            "refine/expand the grid"
        )
    metadata = {
        "common_temperature_J": float(temp),
        "common_bulk_flow_m_s": flow.copy(),
        "species": tuple({"name": name, "mass_kg": float(mass), "density_m3": float(density), "charge_C": float(charge)} for mass, density, charge, name in parsed),
        "operator": "multispecies_linearized_Coulomb_Landau_test_and_field_particle",
        "equilibrium": "common_temperature_common_flow_isotropic_Maxwellian",
        "velocity_grid": "species_scaled_3D_Cartesian_normalized_thermal_coordinates",
        "velocity_nodes_per_species": int(nv),
        "species_count": int(ns),
        "conservation": {"species_particle_number": True, "total_momentum": True, "total_energy": True},
        "entropy_production": "non_positive",
        "fdt": "classical_Langevin_covariance_from_same_discrete_Landau_operator",
        "invariant_relative_residual": inv_resid,
        "invariant_relative_tolerance": invariant_tolerance,
        "invariants_numerically_verified": True,
        "coulomb_logarithm": ln.copy(),
        "limitations": [
            "Non-relativistic isotropic Maxwellian equilibrium only.",
            "Finite velocity box and quadrature require grid/domain convergence.",
            "Assembled reference operator; computational cost grows quadratically with velocity nodes.",
            "No quantum detailed balance or arbitrary non-Maxwellian equilibrium.",
        ],
    }
    return LandauFDTSystem(L, W, Q, metric_diag, moments, tuple(slices), np.stack(velocity_grids), metadata)


def landau_density_response(system: LandauFDTSystem, angular_frequency_rad_s,
                            density_observation, *, forcing=None,
                            wave_number_m_inverse=0.0,
                            propagation_direction=(1.0, 0.0, 0.0)):
    """Compute the Landau resolvent and matching FDT spectrum with sparse solves.

    The adjoint resolvent is solved only for the requested observation
    functionals. This evaluates O A^-1 B and O A^-1 Q A^-H O^H without forming
    a dense inverse or a dense velocity-space covariance. If forcing is None,
    the historical identity-forcing contract is preserved.
    """
    from scipy.sparse import diags as sparse_diags

    omega = np.asarray(angular_frequency_rad_s, dtype=float)
    if omega.ndim != 1 or omega.size == 0 or np.any(~np.isfinite(omega)):
        raise ValueError("angular_frequency_rad_s must be a finite non-empty vector")
    obs = np.asarray(density_observation, dtype=float)
    n = system.operator_s_inverse.shape[0]
    if obs.ndim != 2 or obs.shape[1] != n or np.any(~np.isfinite(obs)):
        raise ValueError("density_observation must have shape (n_observable,n_dof)")
    drive = None if forcing is None else np.asarray(forcing, dtype=complex)
    if drive is not None and (
        drive.ndim != 2 or drive.shape[0] != n
        or np.any(~np.isfinite(drive.real)) or np.any(~np.isfinite(drive.imag))
    ):
        raise ValueError("forcing must have shape (n_dof,n_input)")
    wave_number = float(wave_number_m_inverse)
    direction = np.asarray(propagation_direction, dtype=float)
    if not np.isfinite(wave_number) or wave_number < 0:
        raise ValueError("wave_number_m_inverse must be finite and non-negative")
    if direction.shape != (3,) or np.any(~np.isfinite(direction)) or np.linalg.norm(direction) == 0:
        raise ValueError("propagation_direction must be a finite non-zero 3-vector")
    direction = direction / np.linalg.norm(direction)
    streaming = system.velocity_grid_m_s @ direction
    generator = (
        system.operator_s_inverse
        - 1j * wave_number * sparse_diags(streaming, format="csr")
    ).tocsr()
    source_covariance = system.fdt_source_covariance_s_inverse.tocsr()
    observation_adjoint = obs.T.conj()
    result = np.empty(
        (omega.size, obs.shape[0], n if drive is None else drive.shape[1]),
        dtype=complex,
    )
    spectrum = np.empty((omega.size, obs.shape[0], obs.shape[0]), dtype=complex)
    identity = sparse_diags(np.ones(n), format="csc")
    for index, frequency in enumerate(omega):
        resolvent = (-1j * frequency * identity - generator).tocsc()
        try:
            # X=A^-H O^H; do not construct A^-1 in velocity space.
            adjoint_solution = splu(resolvent.getH().tocsc()).solve(observation_adjoint)
        except RuntimeError as exc:
            raise ValueError(
                f"Landau resolvent is singular at angular frequency {frequency:g} rad/s"
            ) from exc
        result[index] = (
            adjoint_solution.conj().T
            if drive is None else adjoint_solution.conj().T @ drive
        )
        spectrum[index] = (
            adjoint_solution.conj().T @ source_covariance @ adjoint_solution
        )
        spectrum[index] = 0.5 * (spectrum[index] + spectrum[index].conj().T)
    return {
        "response": result,
        "fluctuation_spectral_matrix": spectrum,
        "angular_frequency_rad_s": omega.copy(),
        "collision_operator": system.metadata["operator"],
        "fdt": system.metadata["fdt"],
        "resolvent_solver": "sparse_LU_adjoint_observation_projection",
        "dense_velocity_space_inverse_formed": False,
        "limitations": list(system.metadata["limitations"]) + [
            "Self-consistent electrostatic screening and CTS scattering normalization are not applied.",
            "Streaming is represented by the supplied Fourier wave number and direction; the velocity grid remains finite."
        ],
    }


def landau_cts_dynamic_structure_factor(
    system: LandauFDTSystem,
    angular_frequency_rad_s,
    species: Sequence[Mapping],
    *,
    common_temperature_J,
    wave_number_m_inverse,
    propagation_direction=(1.0, 0.0, 0.0),
) -> dict:
    """Build a screened multispecies CTS density spectrum from Landau--FDT.

    The bare density response is driven by the Fourier-space force associated
    with a species potential energy, U_s=q_s*phi. A self-consistent
    electrostatic Poisson closure is then applied with
    V_ab=q_a*q_b/(epsilon_0*k^2). The same Landau operator supplies the
    velocity-space Langevin covariance; its spectrum is screened by the same
    dielectric resolvent. The classical matrix FDT is evaluated independently
    as a passivity/consistency check.

    This is the electrostatic longitudinal CTS branch. It is not the
    magnetized Full-EM three-wave scattering vertex or optical absolute-power
    normalization.
    """
    from scipy.constants import epsilon_0
    from .cts_fdt import classical_fdt_dynamic_structure_factor

    temperature = _positive(common_temperature_J, "common_temperature_J")
    k = _positive(wave_number_m_inverse, "wave_number_m_inverse")
    omega = np.asarray(angular_frequency_rad_s, dtype=float)
    if omega.ndim != 1 or omega.size == 0 or np.any(~np.isfinite(omega)):
        raise ValueError("angular_frequency_rad_s must be a finite non-empty vector")
    if np.any(omega == 0.0):
        raise ValueError("zero frequency requires a separately converged static-limit extrapolation")
    if not isinstance(species, Sequence) or isinstance(species, (str, bytes)):
        raise ValueError("species must be a sequence of species mappings")
    if len(species) != len(system.species_slices) or not species:
        raise ValueError("species must match the species used to assemble the Landau system")
    charges = np.asarray([float(item["charge_C"]) for item in species], dtype=float)
    if np.any(~np.isfinite(charges)) or np.any(charges == 0.0):
        raise ValueError("every species charge_C must be finite and non-zero")
    direction = np.asarray(propagation_direction, dtype=float)
    if direction.shape != (3,) or np.any(~np.isfinite(direction)) or np.linalg.norm(direction) == 0:
        raise ValueError("propagation_direction must be a finite non-zero 3-vector")
    direction = direction / np.linalg.norm(direction)

    ndof = system.operator_s_inverse.shape[0]
    ns = len(system.species_slices)
    observation = np.zeros((ns, ndof), dtype=float)
    forcing = np.zeros((ndof, ns), dtype=complex)
    for s, sl in enumerate(system.species_slices):
        observation[s, sl] = system.equilibrium_weights[sl]
        projected_velocity = system.velocity_grid_m_s[s] @ direction
        # Linearized Vlasov force term for exp(i*k*x-i*omega*t), h=delta f/F.
        forcing[sl, s] = -1j * k * projected_velocity / temperature

    bare = landau_density_response(
        system, omega, observation, forcing=forcing,
        wave_number_m_inverse=k, propagation_direction=direction,
    )
    chi0 = np.asarray(bare["response"], dtype=complex)
    noise0 = np.asarray(bare["fluctuation_spectral_matrix"], dtype=complex)
    coulomb = np.outer(charges, charges) / (epsilon_0 * k * k)
    identity = np.eye(ns, dtype=complex)
    screened_response = np.empty_like(chi0)
    screened_noise = np.empty_like(noise0)
    dielectric = np.empty_like(chi0)
    for index in range(omega.size):
        dielectric[index] = identity - chi0[index] @ coulomb
        condition = np.linalg.cond(dielectric[index])
        if not np.isfinite(condition) or condition > 1e14:
            raise ValueError(
                "self-consistent CTS dielectric matrix is singular or ill-conditioned; "
                "increase velocity-grid resolution or inspect the requested frequency"
            )
        screening = np.linalg.solve(dielectric[index], identity)
        screened_response[index] = screening @ chi0[index]
        screened_noise[index] = screening @ noise0[index] @ screening.conj().T
        screened_noise[index] = 0.5 * (
            screened_noise[index] + screened_noise[index].conj().T
        )

    fdt = classical_fdt_dynamic_structure_factor(
        omega, screened_response, temperature_J=temperature
    )
    fdt_spectrum = fdt["dynamic_structure_factor_m3_s"]
    scale = np.maximum(
        np.linalg.norm(fdt_spectrum, axis=(1, 2)),
        np.finfo(float).tiny,
    )
    discrepancy = np.linalg.norm(
        screened_noise - fdt_spectrum, axis=(1, 2)
    ) / scale
    return {
        "bare_density_response_m3_per_J": chi0,
        "screened_density_response_m3_per_J": screened_response,
        "dielectric_matrix": dielectric,
        "coulomb_interaction_matrix_J_m3": coulomb,
        "dynamic_structure_factor_m3_s": fdt_spectrum,
        "landau_langevin_spectral_matrix_m3_s": screened_noise,
        "landau_fdt_relative_consistency_residual": discrepancy,
        "angular_frequency_rad_s": omega.copy(),
        "wave_number_m_inverse": k,
        "propagation_direction": direction,
        "collision_operator": system.metadata["operator"],
        "fdt_closure": fdt["closure"],
        "screening_model": "self_consistent_multispecies_electrostatic_Poisson_RPA",
        "cts_model": "electrostatic_longitudinal_multispecies_linearized_Landau_FDT",
        "positive_semidefinite_fdt": fdt["positive_semidefinite"],
        "limitations": list(system.metadata["limitations"]) + [
            "Electrostatic longitudinal response only; magnetized Full-EM CTS requires the electromagnetic current vertex and mode coupling.",
            "Classical FDT only; quantum detailed balance is not included.",
            "The finite-box Landau reference operator must pass velocity-domain and grid convergence before quantitative interpretation.",
            "Absolute scattered power still requires supplied calibrated optical/receiver geometry and Thomson normalization.",
        ],
    }
