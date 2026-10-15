# Implemented diagnostic workflows (audit branch)

This page records the 11 registry-only workflows advanced in this branch. "Implemented" means executable numerical processing with explicit input assumptions; it is not a claim of device-specific calibration or independent experimental validation.

| Diagnostic | Workflow / callable | Output and limits |
|---|---|---|
| Spectral wavelength calibration | `tis.workflows.wavelength_calibration_workflow` | Polynomial wavelength map, per-line residuals, RMS/MAE, monotonicity check; acceptance threshold is optional and user-supplied. |
| Instrumental line-shape characterization | `tis.line_shape.instrument_line_shape_workflow` | Observed FWHM, nominal resolving power, fit quality; intrinsic reference-line width is not automatically deconvolved. |
| Magnetic flux / poloidal flux change | `tis.magnetic.magnetic_flux_from_voltage` | Faraday-law integration with explicit polarity and optional baseline/drift correction; requires calibrated loop voltage and does not reconstruct equilibrium. |
| Gas flow meters | `tis.gas_flow.gas_flow_to_particle_inventory` | Standard-volume flow to molar/particle flow and integrated inventory; standard pressure, temperature and gas calibration are explicit. |
| Effective pumping speed | `tis.gas_flow.effective_pumping_speed_m3_s` | Computes `S=Q/p` for throughput in Pa m³/s and pressure in Pa. |
| Fast bolometry | `tis.fast_bolometry.fast_bolometry_timeseries` | Sampling/QC summary, optional low-pass filtering and Welch PSD; not an absolute radiated-power inversion. |
| Channel synchronization | `tis.channel_sync.estimate_channel_time_offset` | Relative cross-correlation lag and first-order time-axis correction; ambiguous for periodic/weakly coupled signals and not a hardware trigger calibration. |
| MHD mode / Mirnov analysis | `tis.mhd_modes.mhd_mode_workflow` | Coherent frequency and spatial mode-number estimate; geometry, sign convention, coherence and spatial aliasing remain important. |
| Hard X-ray detector | `tis.hard_xray.hard_xray_spectrum_workflow` | Counts, background-subtracted totals and optional count rates; no detector-response unfolding, dead-time correction or absolute photon-flux inference. |
| Bolometric tomography | `tis.tomography_workflow.bolometric_tomography_workflow` | Ray-length matrix plus regularized grid reconstruction; depends on supplied geometry, calibrated line integrals and regularization. |
| Visible / optical emission spectroscopy | Existing `tis.core.fit_spectral_line` plus `tis.workflows.wavelength_calibration_workflow` | Calibrated-axis line centroid/width/intensity fitting; no automatic species identification or atomic-population inference. |

## Validation status

Synthetic regression tests are added for the numerical contracts and known-by-construction examples. Passing software tests verifies implementation behavior only. Independent scientific validation remains separate and must use matched raw diagnostic data and an independently reported reference result where available; otherwise validation is marked pending.
