# Day 1 research improvement checkpoint

Created 2026-11-15. Current time at session start: 13:58 UTC (18:58 Asia/Karachi). The user supplied no demonstration or handoff date; the staged default is one complete day. This is an audit checkpoint, not a submitted snapshot.

## Preservation and workspace state

The workspace is on branch `main` tracking `origin/main`, with extensive pre-existing modifications, deletions, and untracked files. No cleanup, reset, or commit was performed. The submitted Phase 2 basis is `docs/23i-2604_23i-2581DAV-phase2.docx`; its observed SHA-256 is `786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4`. The implementation brief PDF hash is `9a94654bdb772b03d5228993f2cc7eaf1b8552f41c462ea84a4649bb0a828115`.

Existing validation receipt `reports/phase3/validation/complete_package_validation.json` records the Phase 2 hash above, implementation PDF hash above, v1 submission archive SHA-256 `ddf91fa89ab9a9d45c20059c1dbea5cfdd966f3eea62a605769a371ae55e0ebd`, and v1 full reproduction archive SHA-256 `0baba912c5f3e48a8a08a606e2a77a303f67e884ddccfef0907046d94dca5519`. The same receipt reports completed research archive SHA-256 `2a055ce833894338f1a0e9da8b9008acfcc86d0deb6e5fdba89e61303d9edfbb` and full archive SHA-256 `660014248d913c0a2edd96485f7c60dad5e875f30e3824e96aabb97bd03f4bff`. Fresh hashes of these archives, the submitted Phase 2 DOCX, and implementation PDF match the receipt. The recorded submitted archive is `output/phase3/23I-2581_23I-2604_Phase3.zip`, freshly verified as `279d9867d1c75720a51df12caae2ac0b4d490641b83752e32642dcb87e1c8877`. Recheck these hashes at handoff.

## Existing evidence found

- The locked v1 manifest describes 30 final seeds, seven conditions, 500-row batches, frozen logistic regression, input KS tests, PCA evidence, isotonic correctness calibration, delayed verification, and declared baseline/ablation comparisons.
- The protocol lock receipt reports 210 streams, 2.52 million monitoring rows, and 5,040 online batches. Existing reproduction receipts claim exact CSV/array agreement and package tests. These have not been rerun in this session.
- Existing v2 report identifies itself as outcome-informed and separates the XOR synthetic extension from external validation.
- Existing adaptation report identifies itself as an exploratory, outcome-informed extension; it documents cumulative released-label refitting and seed-level uncertainty.
- Existing v1 report describes the coordinated evidence example as as-of clock 16, and explicitly limits live verified accuracy to released labels.
- Existing v2 report records rejection of the local ELEC2 candidate because chronology fields do not uniquely order observations and the processed parquet failed to read.

## Traceability (initial)

| Phase 2 aim | Phase 3 implementation/evidence located | Boundary or open audit item |
|---|---|---|
| Detect input distribution changes | Original-feature KS tests and frozen PCA anchor; v1 conditions C1/C2/C6 | Detection is not equivalent to predictive harm; inspect dependence generator and evaluator timing |
| Explain affected features | Feature rankings, clusters, coordinated evidence reports | Descriptive associations only; no causal attribution |
| Estimate model performance impact | Frozen classifier, separate correctness calibration, estimated accuracy and intervals | Calibration transfer and plug-in interval coverage are empirical limitations |
| Verify estimates with delayed labels | Scheduled label broker, release accounting, adaptation extension | Confirm labels cannot enter pre-release scoring and as-of views exclude retrospective truth |
| Communicate evidence through IDM-Viz | HTML evidence replay and figures | Recheck generated view claims against data timestamps and evaluator data |
| Evaluate controlled drift scenarios | Seven original synthetic conditions; v2 adds XOR/tree | Synthetic results do not establish external validity |
| Study response to feedback/adaptation | Separate cumulative-label adaptation protocol | Small selected scenario set; exploratory due outcome-informed selection |

## Material issues to resolve

1. C6 source uses `X1=.9 Z0+sqrt(1-.9^2) Z1` after onset, preserving unit marginal variance for independent standard normal innovations. This inspection supports the current generator, not any historical implementation.
2. `replay.py` persists predictions at phase 0 before releasing due labels at phase 1, including delay zero; `as_of` omits unreleased truth. `core.py` exposes a feature-only `FeatureBatch`. Source checks support the intended label-timing contract. Fresh inspection of evaluator aggregation, bootstrap resampling, and v2/adaptation label timing remains pending.
3. No LaTeX compiler was found on PATH in the initial check. Existing Phase 3 reports are Markdown/PDF; no editable LaTeX report source is confirmed. Check alternative local TeX installations and available compiler routes.
4. Build a single-column LaTeX report from verified evidence, include bibliography and source figures, compile and inspect every page, then validate a clean extracted package.
5. Recompute all preservation hashes at handoff and report actual checks only.

## Environment and validation status

The expected interpreter exists and reports Python 3.13.5. A previous environment receipt lists NumPy 2.1.3, SciPy 1.17.0, scikit-learn 1.8.0, pandas 2.3.3, matplotlib 3.10.8, pytest 8.3.4, and threadpoolctl 3.5.0; these versions were not independently checked in this session. No experiment was run in this session. Reported test/reproduction outcomes above are historical receipt contents, not fresh validation.

## Continuation priority

1. Inspect evaluator aggregation and bootstrap resampling for timing and dependence errors.
2. Inspect v2 and adaptation source label clocks and report context.
3. Check TeX compilation routes and report source evidence.
4. Correct only substantiated material defects in a new versioned implementation; run targeted validation.
5. Produce report, figures, PDF, coordinated evidence view, reproduction guide, and clean-package receipt for Day 1.
