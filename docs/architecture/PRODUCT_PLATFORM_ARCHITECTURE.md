# Product Platform Architecture

Status: proposed architecture baseline
Target branch: `main`
Scope: the 20 user-facing analysis families currently presented by TIS

## 1. Decision

Tokamak Insight Suite (TIS) will evolve toward a **platform plus a family of independently maintainable specialist products**.

This does not mean immediately creating 20 repositories, 20 deployments, or 20 duplicated applications. The first implementation is a modular platform in the existing repository. Each specialist product receives explicit ownership of its physics workflows, numerical solvers, input/output contracts, tests, documentation, API routes, and dashboard experience. Shared infrastructure remains in the platform.

A specialist product may later be extracted into its own distribution or repository when its dependency graph, release cadence, user base, and operational needs justify that cost.

## 2. Product taxonomy

TIS currently presents 20 primary analysis families: 19 physical/diagnostic families and one cross-cutting instrumentation-support family.

1. Thomson Scattering
2. Langmuir & Electrostatic Probes
3. Electron Cyclotron Emission (ECE)
4. Doppler & Charge-Exchange Spectroscopy
5. Molecular Spectroscopy
6. Magnetic Diagnostics & Equilibrium
7. Emission Spectroscopy
8. Interferometry & Reflectometry
9. Bolometry & Radiation Tomography
10. Imaging & IR Thermography
11. Neutron Diagnostics
12. Gamma-Ray Spectroscopy
13. Fast-Particle Diagnostics
14. Gas, Pressure & Fueling Diagnostics
15. Fluctuation, MHD & Transient Analysis
16. Transport, Confinement & Plasma Parameters
17. Heating & Power Analysis
18. Plasma-Facing Components & Thermal Analysis
19. X-Ray Diagnostics
20. Instrumentation & Measurement Support

Family 20 is a shared cross-diagnostic capability set, not a physical diagnostic in the same sense as families 1–19. Its services should be reusable by all products and may be surfaced in their workflows without duplicating implementations.

The family taxonomy is a navigation and product-ownership layer. It must not collapse distinct physical models or numerical workflows into a single generic solver.

## 3. Target architecture

### 3.1 Shared platform

The platform owns cross-cutting capabilities:

- calibrated measurement data models, units, metadata, and provenance;
- data import/export and quality inspection;
- common uncertainty and covariance primitives;
- reusable numerical utilities where semantics are genuinely shared;
- API infrastructure, authentication/authorization if introduced, job execution, and serialization;
- product discovery, capability metadata, and version compatibility;
- dashboard shell, navigation, plotting primitives, and report/export infrastructure;
- test utilities, CI, packaging, logging, and documentation conventions.

A shared utility must not silently preprocess or alter scientific measurements. Each physical workflow declares its own required measurement state and preprocessing contract.

### 3.2 Specialist products

Each physical family owns its domain-specific:

- physics equations and assumptions;
- forward models and inverse problems;
- numerical solvers and optimization strategy;
- identifiability and validity checks;
- uncertainty propagation specific to its measurement physics;
- public workflow/API contract;
- dashboard pages and result interpretation;
- analytic, synthetic-recovery, regression, and benchmark tests;
- scientific references, limitations, and reproducibility examples.

Products may depend on shared platform services and on explicitly declared cross-product interfaces. They must not import another product's private implementation modules.

### 3.3 Modular monolith first

The initial deployment remains one Python application and one repository. Enforce product boundaries through packages, public interfaces, dependency rules, and tests before considering separate processes or repositories.

Do not introduce microservices, distributed messaging, or separate databases merely to achieve conceptual independence. Extraction is justified only by concrete release, scaling, security, dependency, or collaboration requirements.

## 4. Product completion contract

A product capability is not complete merely because it has a registry row, formula record, placeholder, UI label, or smoke test.

Every claimed executable physical capability must provide:

1. **Physics definition:** equations, symbols, units, assumptions, validity domain, and references.
2. **Numerical implementation:** a callable solver or calculation that evaluates the declared model; no placeholder or unrelated function mapping.
3. **Input/output contract:** required calibrated measurements, optional auxiliary inputs, outputs, uncertainty semantics, and explicit non-identifiability behavior.
4. **Physical/software tests:** analytic limits or invariants, dimensional and boundary checks, synthetic recovery with known parameters, failure/invalid-domain tests, and reference benchmarks when available.
5. **Execution path:** connected through the product workflow, public API, registry/catalog, and dashboard where the capability is intended to be user-facing.
6. **Reproducibility:** parameters, units, model selection, software version, and provenance sufficient to reproduce a result.

Independent experimental validation must be reported distinctly from analytic and synthetic tests. Never describe a model as experimentally validated unless that evidence exists. The project's open-research policy remains applicable: independent experimental validation is not a blanket prerequisite for releasing research software, but limitations and evidence level must be explicit.

## 5. Product interface contract

Each specialist product should progressively expose a stable interface equivalent to:

- product metadata: identifier, name, version, supported capabilities;
- input schema: names, units, dimensions, required/optional status, validation rules;
- execution entry point: typed request to typed result;
- result schema: values, units, diagnostics, uncertainties/covariances, fit quality, warnings, and provenance;
- capability catalog: equation/model identifiers and maturity/evidence labels;
- dashboard registration: navigation, forms, plots, tables, and export;
- test entry points: fast unit tests and slower integration/benchmark tests.

The existing analysis registry and numeric IDs remain compatibility surfaces during migration. Do not renumber or silently repoint existing analysis IDs as part of architectural cleanup. Any intentional retirement or remapping requires a documented compatibility decision and migration tests.

## 6. Thomson Scattering as the pilot product

Thomson Scattering is the first product to receive explicit product boundaries because its scope spans optical non-collective scattering, collective scattering, kinetic distributions, spectral inference, and spatial profile reconstruction.

The pilot should be decomposed into domain subpackages only after a dependency and execution-path inventory. Candidate bounded domains include:

- optical non-collective forward models and inference;
- collective optical Thomson scattering and dielectric/kinetic response;
- microwave/frequency-domain CTS;
- EVDF models and reconstruction;
- spectral response, background, and covariance handling;
- multi-channel and multi-view inference;
- density/temperature/profile reconstruction;
- geometry and full-forward uncertainty propagation.

These are candidate boundaries, not permission to create empty packages or move files without understanding dependencies. Each extracted capability must retain its actual solver and be reachable from the supported workflow/API/dashboard path.

## 6.1 Current implementation boundary

The first product-layer implementation is now represented by:

- `tis/products/catalog.py`: derives product descriptors and ownership from the canonical family and analysis registries;
- `tis/products/api.py`: validates product ownership and Python callable signatures before dispatching unchanged arguments to the registered solver;
- `tis/products/dependencies.py`: statically traces local Python imports from registered solver modules and separates shared-domain reuse from product-local dependency closures;
- `tis/products/thomson/`: named facade for the Thomson family workflows;
- `tis/products/langmuir/`: first second-product facade for registered electrostatic-probe workflows;
- `tis/ui/product_workspace_page.py`: dashboard view of product ownership and scientific contracts;
- `app.py`: registered analysis execution now selects products and entry points through the product catalog.

The dependency inventory is an AST-based approximation. Dynamic imports, runtime plugins, non-Python resources, and third-party dependency constraints require additional review before any repository/package extraction.

The canonical callable registry currently contains 86 entry points (IDs 1–87, with retired ID 7 omitted). The diagnostic knowledge-base registry is a separate 80-record catalogue; these counts are not expected to match. See [PRODUCT_IMPLEMENTATION_MAP.md](PRODUCT_IMPLEMENTATION_MAP.md) for the current product-to-code-to-dashboard map.

## 7. Migration plan

### Phase 0 — Baseline and inventory

- Record the current `main` commit and test/CI state.
- Inventory the 20 family definitions, all registered entry points, formula/diagnostic catalogs, API routes, dashboard routes, and package dependencies.
- Map every entry point to its real callable and identify aliases, duplicate paths, dead paths, and unsupported claims.
- Record current behavior before refactoring.

Deliverable: a traceable product-to-code-to-UI map and a list of architectural risks.

### Phase 1 — Platform and product contracts

- Define the common product metadata, request/result, provenance, uncertainty, and capability contracts.
- Define dependency rules: platform may be used by products; product-private modules may not be imported across products.
- Add architecture tests that enforce these rules without replacing physical tests.
- Keep existing public APIs and numeric registry IDs compatible.

Deliverable: documented contracts plus executable contract tests.

### Phase 2 — Thomson pilot

- Select one bounded capability at a time based on the inventory, not on file names.
- Separate domain implementation from orchestration and presentation.
- Keep equations, solvers, validation, workflow/API, registry, and dashboard connected.
- Run targeted tests and the full relevant CI suite after each coherent change.
- Compare numerical outputs before/after refactoring to prevent scientific regressions.

Deliverable: a product boundary demonstrated by working scientific workflows, not directory structure alone.

### Phase 3 — Reuse and second product

- Extract only genuinely shared primitives into the platform.
- Apply the proven contracts to Langmuir & Electrostatic Probes as the second pilot.
- Revise contracts only through explicit compatibility changes.

### Phase 4 — Remaining products

- Migrate the remaining families incrementally.
- Consider separate distributions/repositories only after a product is independently buildable, testable, documented, and versionable.

## 8. Non-goals and safeguards

- No immediate split into 20 GitHub repositories.
- No rewrite of all current code before the inventory exists.
- No mass file moves that only change paths.
- No registry-only or dashboard-only claims of implementation.
- No silent calibration, denoising, resampling, synchronization, or measurement repair.
- No fabricated uncertainties or outputs for non-identifiable quantities.
- No claims of experimental validation based solely on software tests.
- No breaking changes to existing analysis IDs or user workflows without an explicit migration plan.

## 9. Acceptance criteria for the architecture

The architecture baseline is considered implemented only when:

- all 20 families have an explicit owner/product descriptor;
- every registered entry point maps to a real callable and declared product;
- API, registry/catalog, and dashboard exposure agree with the actual executable capability;
- product dependency boundaries are test-enforced;
- shared services have documented semantics and do not absorb domain-specific physics;
- the Thomson pilot demonstrates at least one end-to-end capability with equations, solver, validity checks, tests, reproducible outputs, and a working user execution path;
- existing supported workflows remain compatible or have documented migrations.

This document defines the direction and migration safeguards. It does not claim that these acceptance criteria are already satisfied.
