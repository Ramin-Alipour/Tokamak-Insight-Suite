# Optical Thomson anisotropic projected electron VDF kernel

The kinetic-model module now exposes
`project_bi_maxwellian_electron_vdf` for the gyrotropic, non-relativistic
bi-Maxwellian reference distribution.

For magnetic-field unit vector `b` and scattering-vector unit vector `q`,
the one-dimensional projected temperature is

```text
T_q = T_perp + (T_parallel - T_perp) (b · q)^2
sigma_q^2 = e T_q / m_e
f_q(v_q) = exp[-(v_q-u_q)^2/(2 sigma_q^2)] / (sqrt(2 pi) sigma_q)
```

The function evaluates the analytic projected marginal on a supplied velocity
grid. It does not renormalize a truncated grid; instead it reports
`velocity_grid_probability_coverage`, so finite-grid loss is explicit.

**Identifiability:** one scattering direction identifies only `T_q`. It does
not separately identify `T_parallel` and `T_perpendicular`. Independent
scattering pitch angles are required to separate those temperatures, and the
result remains conditional on the gyrotropic bi-Maxwellian assumption.

**Scope:** this is a tested forward-distribution kernel for OPT-03. It is not
yet an end-to-end anisotropic optical Thomson spectral forward model, optimizer,
dashboard selection, or general anisotropic EVDF inversion. No claim is made
that OPT-03 is fully closed.

Regression coverage is in
`tests/test_thomson_anisotropic_projection.py`. GitHub Actions is responsible
for the test result on the commit that contains this note.
