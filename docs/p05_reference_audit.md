# P05 reference audit: limiter position and external bias

**Paper:** R. Alipour et al., “Effects of the location of a biased limiter on turbulent transport in the IR-T1 tokamak plasma,” *European Physical Journal D* 71, 228 (2017), DOI [10.1140/epjd/e2017-80215-6](https://doi.org/10.1140/epjd/e2017-80215-6).

## Verdict

The prose contains a genuine numerical inconsistency for the radial electric field at a limiter position 5 mm inside the plasma with +200 V applied:

- **Abstract:** radial electric field increases by **35–45%**.
- **Conclusion:** radial electric field increases by **20–30%**.

These ranges do not overlap. The paper provides no tabulated value, uncertainty, or explicit averaging rule that reconciles them. The software therefore preserves both claims and flags the disagreement; it must not silently select, average, or hard-code one as the definitive expected answer.

The reported increase in poloidal turbulent transport is consistent between abstract and conclusion at **25–35%**. Reynolds-stress reduction is described as a maximum of **5–10%** in the abstract/conclusion, while Results §3.6 says about **5%** for +200 V at 5 mm. These are compatible at the lower end.

## Claim-by-claim reconciliation

| Quantity | Condition | Abstract | Results text | Conclusion | Audit |
|---|---|---|---|---|---|
| Radial electric field | 5 mm, +200 V | +35–45% | Positive bias increases it at edge, 5 mm and 10 mm; no matching numeric range in prose | +20–30% | **Unresolved internal contradiction**; retain both ranges |
| Poloidal turbulent transport | 5 mm, +200 V | +25–35% | Positive bias increases it; effect depends on position and polarity | +25–35% | Consistent prose claim |
| Poloidal electric field | 5 mm, +200 V | — | Decreases at edge/5 mm; at 10 mm the paper says it rises nearly 30% under applied ±200 V | Decrease 30–40% | Condition-dependent; not contradictory once position is preserved |
| Radial turbulent transport | 5 mm, +200 V | — | Positive bias reduces it by almost 35% | Decrease 35–45% | Approximately compatible; no uncertainty supplied |
| Radial turbulent transport | Edge, +200 V | — | Reduction about 50% | — | Different position; not comparable to 5 mm |
| Reynolds-stress parameter | 5 mm, +200 V | Maximum reduction 5–10% | About 5% reduction | Maximum reduction 5–10% | Compatible at lower end; condition-specific |

## Qualitative findings preserved by figure/condition

The machine-readable registry also retains the non-percentage statements from Results §§3.1–3.6 so they are not lost behind the abstract's headline percentages:

- **Plasma current (Fig. 2):** +200 V at the edge slightly improves current duration; −200 V at the edge reduces current strongly. At 5 mm, the text says −200 V can sustain a discharge up to 31 ms while +200 V causes earlier disruption; the paper also reports a maximum current above 20 kA at 5 mm without tying that maximum to one polarity. At 10 mm, the current trace is described as the worst and more unstable.
- **Radial electric field (Fig. 3):** positive bias raises the field at the edge and at 5/10 mm; both polarities are said to raise it noticeably at 5 mm. At 10 mm, +200 V is described as more effective than −200 V, although −200 V also has an effect. These qualitative statements do not resolve the conflicting percentage ranges.
- **Poloidal electric field (Fig. 4):** +200 V gives the lowest edge value in the described comparison and reduces the field at 5 mm; at 10 mm, the paper describes a rise of nearly 30% under applied ±200 V, with no uncertainty/statistic specified.
- **Poloidal turbulent transport (Fig. 5):** at the edge, +200 V increases and −200 V decreases transport relative to no bias. At 5 mm, both polarities increase it, with +200 V having the larger effect. At 10 mm, −200 V causes the largest increase and +200 V produces more fluctuations.
- **Radial turbulent transport (Fig. 6):** at the edge, +200 V reduces it by about 50% while −200 V produces erratic changes. At 5 mm, +200 V reduces it by almost 35% but −200 V is not effective at reducing it. At 10 mm, both applied polarities increase it.
- **Reynolds-stress parameter (Fig. 7):** at the edge, +200 V reduces it by 15% while −200 V increases it. At 5 mm, both polarities reduce it and +200 V reduces it by about 5%. At 10 mm, the text describes wild fluctuations and a dramatic increase.

These are transcriptions of the paper's narrative descriptions of plotted traces, not independent digitization of the plotted pixels.

## What the plots establish—and what they do not

Figures 3–7 are time-evolution plots. The prose does not specify a common averaging window, whether percentages refer to mean, peak, envelope, or selected time interval, or the uncertainty of the percentage estimates. Digitizing a plot can estimate displayed curve coordinates, but cannot recover the original acquired signal or resolve the abstract/conclusion inconsistency without a declared comparison statistic.

1. The numbers above are **paper-reported prose benchmarks**, not digitized experimental datasets.
2. This paper does not supply the original IR-T1 raw probe time series.
3. Strict automated pass/fail against the radial-electric-field percentage is unjustified. A result can match the abstract, match the conclusion, match both, or match neither.
4. Edge, 5 mm, and 10 mm positions and +200 V versus −200 V remain separate conditions.
5. The paper's “Reynolds stress parameter” follows its operational definition; software should state the implemented formula and units rather than imply a fully normalized dimensionless Reynolds number.

## Software implementation

The **Paper Benchmarks & P05 Audit** workspace displays the source claim table, highlights the radial-electric-field contradiction, and lets researchers compare an observed percentage with the exact metric/condition. When source sections disagree, the result reports whether it matches one claim but conflicts with another. This is a reference-consistency check, not experimental validation.

Machine-readable claims live in `tis/paper_benchmarks.py` and are covered by tests.
