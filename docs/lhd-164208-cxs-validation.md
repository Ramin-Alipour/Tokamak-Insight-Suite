# LHD shot 164208: CXS7 raw-spectrum validation

## Source data

All three files were downloaded from the public NIFS LHD analyzed-data repository using its official search and file endpoints:

- `lhdcxs7_img`: measured CXS7 spectral images.
- `lhdcxs7_prep`: stripe-to-radius and background-stripe mapping.
- `lhdcxs7_cvi`: NIFS processed ion temperature, velocity, and intensity output.

The automated workflow `.github/workflows/lhd-164208-cxs-validation.yml` downloads the files again and emits a JSON report and full time-series CSV.

## Verified field semantics

The `lhdcxs7_cvi` data columns distinguish `Ti` (keV) from `Vc` (km/s). The two candidate values previously treated as ion temperatures are in fact matches to the velocity field:

- At 3.41 s and (R=3.5974\,\mathrm{m}), (V_c=2.7583\,\mathrm{km/s}), while (T_i=1.6328\,\mathrm{keV}). The value 2.76 matches (V_c), not (T_i).
- At 4.01 s and (R=4.6311\,\mathrm{m}), (V_c=4.3765\,\mathrm{km/s}), while (T_i=0.3462\,\mathrm{keV}). The value 4.38 matches (V_c), not (T_i); it is also at a different radius from the first comparison.

Therefore, 2.76 and 4.38 must not be used as (T_i\) references in keV. The actual processed (T_i) reference at the selected central radius (R=3.5974\,\mathrm{m}) is 1.6328 keV at 3.41 s and 3.8004 keV at 4.01 s.

## Raw spectrum fit versus NIFS output

A reproducible single-Gaussian-plus-linear-baseline fit was applied to the raw spectrum for stripe 36, subtracting its mapped background stripe 32. The observed width was corrected for the stated instrument width in quadrature before applying the carbon Doppler formula.

| Time | Raw-fit (T_i) | NIFS (T_i) | Relative difference | Status |
|---:|---:|---:|---:|---|
| 3.41 s | 1.5155 keV | 1.6328 keV | 7.19% | Pass at 10% tolerance |
| 4.01 s | 2.9833 keV | 3.8004 keV | 21.50% | Pending |

The second point remains pending; the script does not tune the fit window or scale to force agreement.

## Independently checked observable: line-intensity trend

The raw fitted line area was compared with the official `In` series over 80 frames with valid positive values:

- Pearson correlation: (r=0.94877), (p=8.91\times10^{-41}).
- Spearman rank correlation: (ho=0.96967), (p=1.75\times10^{-49}).

The intensity-trend consistency check passes the prespecified (r,\rho\geq0.90) threshold. This validates temporal tracking of the line-intensity signal, not the absolute Ti fit.

## Reproduction

```bash
python tools/validate_lhd_164208_cxs.py
```

The validation deliberately separates the passing intensity check, the single passing Ti point, and the pending Ti point. It does not claim that the entire Doppler-temperature method is validated.


## Cross-product consistency and schema check

The shared fields in the NIFS `lhdcxs7_cvi` and `cxsmap7` products were compared row-for-row across 40,400 time-radius points. Time, radius, and fit-quality code matched exactly. The 95th-percentile relative differences were:

- (T_i): 0.00154%.
- (T_i) uncertainty: 0.00144%.
- (V_c): 0.00135%.
- (V_c) uncertainty: 0.00164%.

This cross-product consistency check passes. It confirms the processed values are internally consistent; it does not independently validate the raw spectral-width reconstruction.

A schema defect was also detected in `cxsmap7`: its declared `ValName` omits the `In` and `Iner` fields present in the data rows after `Ver`. The validation therefore compares only the seven fields whose positions are confirmed by both products and does not interpret later columns from the stale names.
