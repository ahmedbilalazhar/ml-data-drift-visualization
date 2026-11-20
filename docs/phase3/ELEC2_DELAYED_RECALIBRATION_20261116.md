# ELEC2 released-label recalibration extension

**Status:** outcome-informed extension locked before execution. It uses the completed fixed-model ELEC2 cache and does not retrain the classifier.

## Question

Can isotonic correctness recalibration using only previously released monitoring labels improve current-batch correctness estimates relative to the frozen historical calibrator under the same fixed classifier?

## Protocol

- Keep the ELEC2 source, approximate normalized-date order, 10,800/3,662/3,662 historical split, fixed logistic classifier, predictions, feature scores and outcomes exactly as in `ELEC2_TEMPORAL_EXTENSION_20261116.md`.
- The initial isotonic calibrator is the one fit on the original calibration partition. For each delay 0/5/20, before predicting batch `t`, refit isotonic correctness on initial calibration confidence/correctness pairs plus each monitoring instance whose label release clock `source_batch + delay` is strictly less than `t`. Then persist the current batch's recalibrated confidence-derived correctness scores. Release same-clock labels only after persistence, even for delay zero.
- Classifier, scaler and feature representation stay frozen. No batch may calibrate against its own or a future label. The only update is the correctness map. Delay-specific available monitoring-label counts and maximum release clock are audited.
- Compare frozen versus online-updated q with equal-batch absolute estimation error, bias, Brier score and fixed 10-bin ECE. Summarize empirical intervals descriptively; this small series has one domain and no independent stream replicates. Report all three delays; do not choose the best delay post hoc.
- Use every scheduled batch and label, including the final partial batch. Drain labels after the input horizon; draining labels cannot affect earlier forecasts.

This checks a narrow supervised mitigation on one observed stream. It does not establish general calibration repair, zero-delay deployability, accuracy improvement, future-harm prediction, domain transfer or an arbitrary-shift guarantee. An observed improvement can reflect adaptation to this sequence and is exploratory.
