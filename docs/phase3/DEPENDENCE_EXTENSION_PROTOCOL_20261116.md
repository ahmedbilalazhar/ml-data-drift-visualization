# Pairwise dependence detector extension

**Protocol version:** 1.0. **Status:** exploratory follow-up, because the weakness was identified after prior C6 results. This is not a prospective confirmation of the whole IDM-Viz method.

## Question and limits

Does a feature-only maximum pairwise-correlation-change statistic detect the known C6 dependence intervention more often than the frozen marginal KS/PCA alarm while retaining a measured stationary false-alarm rate? It addresses one known dependence mechanism only. It does not detect arbitrary changes in higher-order dependence, prove causality, identify harmfulness or replace natural-stream validation.

## Frozen statistic and threshold

Use the canonical core partitions: stage 0 for the 4,000 training rows and stage 2 for the 2,000 historical-reference rows; use the final 1,000 reference rows as the fixed comparator, matching the existing monitor. At each 500-row batch, calculate the absolute difference between its Pearson correlation matrix and the matrix from that reference. The score is the largest absolute off-diagonal difference; store the maximizing feature pair. Score calculation receives only historical and current feature matrices.

Use development seeds 930000–930029. Estimate one threshold as the empirical 95th percentile with the higher quantile rule over all 30 × 24 stationary C0 batch scores. Freeze this threshold before evaluation. The 95th percentile is a development operating point, not a guaranteed false-alarm rate. Report held-out false alarms directly.

Use evaluation seeds 930100–930129 and conditions C0–C6, 24 batches of 500. Do not replace seeds or change the statistic or threshold after inspecting evaluation output. Inputs and seed laws are inherited from `src/drift_monitoring/phase3/core.py`. In C6 the programmed onset is batch 9 (one-based). On every identical stream and batch, also recompute the existing eight-feature plus three-PCA-coordinate KS family, using training-only scaling/PCA and Holm correction at 0.05. The main comparison is the paired per-seed difference in C6 onset detection by batch 12.

## Outcomes

Primary: per-seed C6 detection by batches 9–12, paired against the recomputed marginal/PCA alarm on the same input stream. Also report C6 detection at any later batch, pre-onset false alarms, stationary C0 false alarms per 1,000 batches, and alarm frequencies in C1–C5. Report paired seed-bootstrap 95% intervals (2,000 resamples) for the difference in onset detection. Report the two detectors' stationary false-alarm burdens separately; the new detector's threshold is calibrated on stationary development batches, not guaranteed to attain a nominal rate.

The maximizing pair is descriptive localization. Score whether pair (0,1) is selected in C6 onset batches, but do not call it causal localization. No threshold optimization on evaluation seeds is allowed.

## Execution and provenance

Run development first, review only the development threshold and integrity checks, then run the frozen evaluation. Each output directory is new and contains the protocol copy, threshold, per-batch predictions, per-seed summaries, bootstrap contrasts, environment and SHA-256 hashes. Online detector outputs contain no labels or realized harm. The standalone follow-up does not alter earlier result files.
