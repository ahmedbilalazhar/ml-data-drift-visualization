# Phase 3 implementation hardening — 2026-11-15

This is a separately dated source-code hardening pass on the current workspace. It does not revise the submitted Phase 2 methodology or retroactively alter any Day 1–3 report, locked study, or archive. No instructor feedback was received; these are implementation corrections identified by code review and reproduction.

## Corrections

- Added input contracts to the Phase 3 fit and scoring APIs: non-empty aligned matrices and vectors, finite features, binary labels, compatible widths, unique IDs, no train/test ID reuse, and both classes in the training partition.
- Hardened Holm adjustment, harm-bound calculations, impact scoring, and ATC calibration against malformed, empty, non-finite, or out-of-domain inputs.
- Hardened delayed-label brokers and replay: valid batch/delay clocks, unique IDs, binary labels, complete batch coverage, aligned cached predictions, and unique cohort identities in as-of views.
- Added equivalent shape, label, identifier, and finite-value checks to v2 model fitting/scoring and cumulative adaptation/release processing.
- Hardened UCI 224 ingestion for CRC, malformed LIBSVM rows, invalid class/feature indices, duplicate features, and non-finite values. Added an explicit `--out` path that refuses existing result directories and records source-code hash, split indices, and running/completed status.
- Fixed the runner's final partition-coverage assertion, which used an undefined `indices` variable and aborted before the external evaluation.
- Added focused regression tests for these contracts and for the locked 267/89/89 historical split.

## Reproduced external result

The corrected runner completed on the verified local UCI 224 archive. Its ten cohort rows, all monitored numeric/boolean metrics, class counts, and class proportions matched `results/phase3/external_uci224_day2_v2` exactly. The recorded fit/calibration/reference sizes are 267/89/89 from batch 1. Across the nine later cohorts, accuracy-estimation MAE was 0.5446578894, interval coverage was 0/9, and all 9 cohorts triggered an input alarm. These results expose weak calibration/interval behavior and high alarm burden on this single external dataset; they do not establish general performance or known drift-onset recall.

## Verification

- `python -m pytest -q`: 76 passed.
- Cleanly extracted overlay regression tests: 28 passed.
- Overlay ZIP integrity check passed; the package contains 27 files.
- Fresh external runner: completed 10 cohorts; row-level comparison to the corrected v2 result passed exactly.
- Fresh run protocol lock records `status: complete`, source archive SHA-256, implementation SHA-256, and split indices/counts.
- Submitted-document and archive hashes are listed in `HASHES.sha256` and match the preserved Day 3 hash ledger.

Run from the repository root with the expected interpreter:

```powershell
& 'C:\Users\hp\anaconda3\python.exe' -m pytest -q
& 'C:\Users\hp\anaconda3\python.exe' scripts\evaluate_uci224_day2.py --out results\phase3\external_uci224_day2_hardening_check
& 'C:\Users\hp\anaconda3\python.exe' docs\phase3\implementation_hardening_20261115\build_overlay.py
& '.\docs\phase3\implementation_hardening_20261115\verify_hashes.ps1'
```

The second command intentionally fails if that result directory already exists. Choose a new versioned output directory for another run.

## Remaining limits

This pass adds defensive checks and reproduces existing evidence; it does not repair the external study's observed interval undercoverage, estimation error, or alarms. Those are findings requiring a separately locked methodological investigation. This pass did not re-run every historical seed/condition, inspect every dashboard interaction, or recompile the three reports. Existing daily packages remain the last report/PDF checkpoints and do not contain these later source changes. The Day 3 final package remains preserved as its own snapshot.
