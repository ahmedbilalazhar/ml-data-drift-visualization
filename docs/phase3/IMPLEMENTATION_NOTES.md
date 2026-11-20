# Phase 3 implementation record, 15 November 2026

The submitted `docs/23i-2604_23i-2581DAV-phase2.docx` is the methodological basis.
No Phase 2 instructor feedback was received. The original submitted documents and
the existing plan/manifest are preserved. Legacy E0 outputs and the development
timing probe are excluded from all Phase 3 research results.

## Scope and interpretation

The existing modest plan is implemented in a new `drift_monitoring.phase3` package.
PCA replaces the proposed neural anchor; marginal KS ranks replace causal discovery;
three frozen clusters describe occupancy and estimated correctness. Causal discovery,
online adaptation/recalibration, selective labels, the 20-person study, and external
dataset evaluation are deferred. These are team scope choices, not instructor-approved
exemptions. Human benefit and causal responsibility are untested. RQ1 is evaluated
for current-batch realised deterioration (h=0), not forecasting; RQ2 is limited to
descriptive marginal ranking; RQ3 remains unanswered; RQ4 covers controlled regimes
of one generator family, not cross-domain transfer.

## Clarifications needed to make the plan executable

- Batch numbers and clocks start at 1. At clock t the input prediction is persisted
  in phase 0, then due labels are verified in phase 1. Delay zero therefore still
  predicts before observing its labels. Clock drainage ends at 24+d with no synthetic
  input batches. CSVs separate online evidence, release events and evaluator truth.
- C2 adds its shift to X0 after the baseline joint construction. X1 is retained.
  Labels are sampled from the modified X; the ramp's first changed row receives
  1.5/2000 and row 6000 reaches 1.5, following the declared endpoints.
- ATC-style threshold candidates include observed scores and endpoints 0, 1 and
  nextafter(1,+infinity), so zero accepted rows is attainable even if a score is 1.
  The largest threshold breaks exact objective ties. This is a disclosed adaptation.
- Harm is compared through exact integer/rational arithmetic. Concern is the
  plug-in probability of strictly crossing that same threshold. The point ablation
  uses the stated floating-point point estimate and a strict 0.05 comparison.
- Original-only KS recomputes Holm with 8 tests; full KS uses 11. This changes
  multiplicity allocation as well as the evidence representation. The impact cache
  is identical for the no-latent ablation. Cluster rates below n=20 are missing.
- Repeated seeds are independent random draws of one generator family. Conditions
  share innovations within seed; conditional-only cases share inputs exactly.
  Delay schedules and ablations are not independent replications. Bootstrap
  intervals resample seed vectors, with 2,000 deterministic replicates. Undefined
  per-seed precision/recall/AUROC stays missing; pooled numerators/denominators
  are supplied separately, and the number of defined seeds is reported.
- The broker is a software API boundary, not a cryptographic isolation mechanism.
  Feature-only scoring receives neither current labels nor a broker handle.
  Labels are kept in explicitly evaluator-only archives. The offline evidence
  viewer may contain eventual labels in its source data, but filters every displayed
  result by release time and labels its retrospective role.

## Schedule

Initial clock check: 15 November 2026, 14:14:02 Asia/Karachi; 2h45m58s to the
17:00 deadline. Implementation and tests take priority, then final experiments,
figures/report, independent reproduction and a versioned submission archive.
No upload or submission is claimed without a real upload confirmation.

## Completed implementation and measured validation

- All five declared development seeds completed and passed artifact validation.
- The final run used every seed 100-129: 210 streams, 2,520,000 monitoring rows,
  5,040 named batches and 15,120 scheduled verification records. No seed was
  replaced. The final runner took 430.511 seconds, with zero recorded KS warnings
  and no logistic convergence failures.
- All 39 automated tests passed (26 Phase 3 tests plus 13 legacy regression tests).
  The legacy tests passing does not validate legacy result files. JUnit evidence
  is in `reports/phase3/validation/pytest.xml`.
- A complete independent 30-seed reproduction matched exactly across 21
  scientific CSVs (1,683,110 cells) and 2,400 arrays (33,010,080 values) in 450 NPZ
  archives. Elapsed timings and ZIP metadata were excluded. Originally locked
  numerical sources match; later reporting/audit scripts are separately identified.
- The offline replay passed 488,880 release-boundary comparisons and interaction
  checks. A change-only select interaction initially left panels stale; this UI
  bug was corrected by handling both input and change events, then revalidated.
  Future accuracy/F1 remains hidden at delay 20 before the scheduled release.
- Four result figures and every page of the ten-page PDF were visually reviewed.
  Updated figure 1 plots the actual adjusted p-value on a log axis against .05.
- Measured results are retained even when unfavorable: calibrated overall MAE
  0.114189 versus raw AC 0.113387, calibrated coverage 0.695238, no calibrated
  concern flags in conditional-only C3-C5, and dependence onset matches 8/30 for
  full KS/PCA versus 4/30 for original KS. No tuning followed these final outcomes.
- Individual bootstrap intervals for identical channels can differ slightly from
  Monte Carlo resampling; their estimates and prediction intervals are exactly
  identical. Paired method differences use paired seed vectors directly.
- The 10-page report and versioned packages are prepared artifacts, not an upload
  confirmation. The original DOCX and requirements PDF still match their recorded
  SHA256 values. Use a new version for subsequent changes.
