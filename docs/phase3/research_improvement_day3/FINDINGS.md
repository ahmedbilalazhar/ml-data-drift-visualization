# Day 3 final findings

- The submitted Phase 2 document remains unchanged and is the basis. No instructor feedback is claimed.
- Existing Phase 3 synthetic evidence and all historical archives remain preserved in nested Day 1/Day 2 packages.
- External dataset admission passed: UCI 224 has 13,910 finite, 128-dimensional records across ten source batches, but cohorts are unequal and class composition changes. One source-batch history limits the external evaluation.
- Corrected protocol uses 267/89/89 batch-1 train/calibration/reference rows. An earlier implementation split bug was detected, its results were marked superseded, and corrected v2 was run without changing model or threshold.
- Corrected external results: mean observed accuracy 0.432, macro-F1 0.290, calibrated estimate 0.976, MAE 0.545, interval coverage 0/9 cohorts, and input alarms in all 9 cohorts.
- Exploratory sensitivity after the primary results shows confidence overstates observed correctness by 0.18-0.57 in future cohorts. Common-class standardization does not remove poor accuracy; it is not causal and excludes absent classes.
- Existing `tests/phase3`: 35 passed. All Day 2/Day 3 PDF pages were rendered and inspected. Day 2 and Day 3 packages passed clean extraction and CRC checks.
- The result is one external dataset evaluation, not independent replications, known-onset recall, human impact, causal explanation, or population safety evidence.

## Deviations

The extension is multiclass and whole-batch rather than binary and within-stream. It does not emulate delayed label release or claim exact event times. The source description says research-only and excludes commercial use even though its page also lists CC BY 4.0; use is restricted to research in this package. Day 3 class-mixture sensitivity was outcome-informed after calibration failure was observed. Each deviation is documented in the protocol and report.
