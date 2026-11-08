# Product implementation map

Generated against the canonical registry on the current `main` branch. This is an ownership/execution map, not a claim that every entry is scientifically complete.

## Registry reconciliation

- Primary products: 20
- Callable analysis entry points: 86
- Registered IDs: 1–87, with retired ID 7 omitted
- Diagnostic knowledge-base records: 80 (a separate catalogue layer; not required to be one-to-one with callable workflows)
- Product ownership: every callable entry point must occur in exactly one family.

## Shared platform services

| Shared concern | Current implementation surface | Boundary rule |
|---|---|---|
| Canonical workflow metadata | `tokamak_analysis/analysis_registry.py`, `ANALYSIS_FAMILIES.json` | Owns IDs and contracts; no physics solver logic |
| Product metadata and ownership API | `tis/products/catalog.py`, `tis/products/api.py` | Resolves ownership and dispatches; does not mutate input measurements |
| Units, signal and data utilities | `tis/core.py`, `tis/signals/`, `tis/data_sources.py` | Reusable primitives; diagnostic assumptions remain in product solvers |
| Measurement uncertainty | `tokamak_analysis/measurement_uncertainty.py` and domain covariance routines | Shared numerical conventions; covariance semantics remain domain-specific |
| Catalogues and formula metadata | `tis/catalog.py`, `DIAGNOSTIC_REGISTRY.json`, `FORMULA_REGISTRY.json` | Descriptive catalogue does not replace executable implementations |
| Dashboard shell and reporting | `app.py`, `tis/ui/`, `tis/report.py` | Presentation only; calculations remain in domain modules |
| Provenance and contribution export | `tis/contributions.py`, project persistence modules | Capture provenance; do not infer missing physical metadata |

## Product-to-code ownership

| # | Product | Stable product ID | Entry IDs | Registered solver modules |
|---:|---|---|---|---|
| 1 | Thomson Scattering | `thomson-scattering` | 1, 30, 46, 81, 82, 83, 84, 85, 86, 87 | `advanced_inference`, `cts_analysis`, `cts_full_physics`, `thomson_distribution`, `thomson_scattering`, `thomson_temperature_tensor`, `thomson_workflow` |
| 2 | Langmuir & Electrostatic Probes | `langmuir-electrostatic-probes` | 2, 16, 17, 18, 19, 20 | `ball_pen_probe`, `gundestrup_probe`, `langmuir_probe`, `multipin_probe`, `potential_analysis`, `probe_analysis` |
| 3 | Electron Cyclotron Emission (ECE) | `electron-cyclotron-emission` | 3 | `ece_analysis` |
| 4 | Doppler & Charge-Exchange Spectroscopy | `doppler-charge-exchange-spectroscopy` | 4, 5, 24 | `cxrs`, `impurity_concentration`, `magnetic_field` |
| 5 | Molecular Spectroscopy | `molecular-spectroscopy` | 6 | `molecular_spectroscopy` |
| 6 | Magnetic Diagnostics & Equilibrium | `magnetic-diagnostics-equilibrium` | 8, 9, 10, 11, 12, 13, 14, 15, 55 | `equilibrium_reconstruction`, `mhd_modes`, `plasma_geometry`, `polarimetry`, `poloidal_flux`, `shafranov_shift`, `toroidal_field_coils`, `x_point` |
| 7 | Emission Spectroscopy | `emission-spectroscopy` | 21, 22, 25 | `doppler_spectroscopy`, `doppler_temperature`, `spectral_broadening` |
| 8 | Interferometry & Reflectometry | `interferometry-reflectometry` | 28, 29, 31 | `interferometry`, `plasma_wave_spectrum`, `reflectometry` |
| 9 | Bolometry & Radiation Tomography | `bolometry-radiation-tomography` | 32, 33, 39 | `hard_xray`, `radiation_tomography`, `soft_xray` |
| 10 | Imaging & IR Thermography | `imaging-ir-thermography` | 36, 37, 38, 75 | `edge_field`, `line_radiation`, `radiation_tomography`, `visible_imaging` |
| 11 | Neutron Diagnostics | `neutron-diagnostics` | 40, 41, 42 | `fusion_rate`, `neutron_flux`, `neutron_spectrum` |
| 12 | Gamma-Ray Spectroscopy | `gamma-ray-spectroscopy` | 43 | `nuclear_gamma` |
| 13 | Fast-Particle Diagnostics | `fast-particle-diagnostics` | 44, 45, 47, 48 | `energetic_particles`, `runaway_electrons`, `runaway_energy` |
| 14 | Gas, Pressure & Fueling Diagnostics | `gas-pressure-fueling` | 49, 50, 51, 52, 53 | `fueling_rate`, `gas_puffing`, `neutral_gas`, `vacuum_pressure`, `wall_recycling` |
| 15 | Fluctuation, MHD & Transient Analysis | `fluctuation-mhd-transient-analysis` | 54, 56, 57, 58, 59, 60 | `fft_analysis`, `magnetic_mode_frequency`, `mirnov_analysis`, `svd_analysis`, `turbulence`, `wavelet_analysis` |
| 16 | Transport, Confinement & Plasma Parameters | `transport-confinement-plasma-parameters` | 61, 62, 63, 64, 65 | `confinement`, `heat_flux`, `transport` |
| 17 | Heating & Power Analysis | `heating-power-analysis` | 66, 67, 68, 69, 70 | `ecrh_icrh`, `heating_analysis`, `rf_interaction` |
| 18 | Plasma-Facing Components & Thermal Analysis | `plasma-facing-components-thermal-analysis` | 71, 72, 73, 74, 76 | `divertor_analysis`, `lh_transition` |
| 19 | X-Ray Diagnostics | `x-ray-diagnostics` | 23, 34, 35 | `bolometry`, `doppler_flow`, `xray_spectrum` |
| 20 | Instrumentation & Measurement Support | `instrumentation-measurement-support` | 26, 27, 77, 78, 79, 80 | `channel_synchronization`, `denoising`, `measurement_uncertainty`, `signal_calibration`, `spectral_line_identification`, `spectral_line_ratio` |

## Execution surfaces

- Generic Python API: `tis.products.execute_product_analysis(product_id, analysis_id, *args, **kwargs)`.
- Thomson specialist API: `tis.products.thomson` (named wrappers for the 10 Thomson family workflows).
- Langmuir specialist API: `tis.products.langmuir` exposes named facades for Langmuir I-V, compatibility I-V, ball-pen, Gundestrup, multi-pin, and plasma/floating-potential workflows; each uses ownership-checked dispatch to the canonical registry.
- Dashboard product workspace: `tis/ui/product_workspace_page.py`, reachable from **Advanced Analysis → Specialist Product Workspace**.
- Existing execution form: **Advanced Analysis → Registered analysis modules** now resolves its family and subanalysis list through the product catalog, then executes the original registry callable.
- The repository currently exposes a Python execution API and Streamlit dashboard; this change does not claim that a separate HTTP/REST server exists.

## Thomson pilot boundary

| Concern | Current domain modules | Product API |
|---|---|---|
| Optical measured-spectrum inference | `thomson_inference.py`, `thomson_workflow.py` | `analyze_spectra` |
| Collective / coherent scattering | `cts_analysis.py`, `cts_full_physics.py`, kinetic/EM CTS modules | `analyze_cts`, `analyze_landau_fdt_convergence` |
| Spectral distribution diagnostics | `thomson_distribution.py` | `analyze_spectral_distribution` |
| Spatial/profile inference | `thomson_workflow.py`, `thomson_joint_spatial_spectral.py` | `reconstruct_profiles` |
| Gyrotropic and tensor inference | `thomson_anisotropic_inference.py`, `thomson_temperature_tensor.py` | `fit_gyrotropic`, `fit_temperature_tensor`, `compare_temperature_tensor_models` |
| Relativistic 3-D EVDF | `advanced_inference.py` | `reconstruct_relativistic_evdf` |
| Correlated optical calibration density | `thomson_scattering.py` | `infer_path_averaged_density` |

The Thomson facade delegates to the canonical registered functions without reimplementing their equations or changing their inputs. The product boundary is therefore an executable API boundary, not a duplicate physics layer.

## Second-product pilot: Langmuir & Electrostatic Probes

The Langmuir facade exposes named entry points for IDs 2 and 16–20 while retaining each diagnostic's distinct physical model. Registry labels for the six workflows are aligned with their actual solver modules. The multi-pin solver propagates shared or time-resolved full cross-pin voltage covariance through its nonuniform finite-difference operator, C_E = D_x C_V D_x^T, and declares the unsupported cross-time and coordinate-uncertainty terms. The product does not assume that all probe types share one sheath model.

## Dependency audit scope and next refinement

The executable AST dependency inventory is available through `tis.products.build_dependency_inventory()`. It follows static local imports from every registered solver module and reports entry-point modules, transitive domain-module closures, modules reused by multiple product closures, product-local modules, and direct shared-platform imports.

The graph intentionally does not claim to resolve dynamic imports, runtime plugin loading, non-Python resources, or third-party package constraints. Before extracting products into separate distributions, review those limitations, classify every cross-product dependency as a public interface or shared platform service, and add import-boundary tests. Do not create microservices or duplicate shared services merely to satisfy the product taxonomy.
