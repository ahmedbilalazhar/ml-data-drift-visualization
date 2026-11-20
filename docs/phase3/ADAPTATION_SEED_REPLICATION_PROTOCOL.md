# Large-seed replication of released-label adaptation

**Status:** Outcome-informed replication protocol first locked 15 November 2026, 22:52 Asia/Karachi, before these seeds were run. A sample-size-only amendment was made at 23:11 Asia/Karachi, also before any final seed was run. It follows the completed pilot and is not a preregistration made before the earlier studies.

## Weakness and question

The adaptation pilot used 30 seeds per task and selected two conditional-only cases after inspecting prior evidence. Its zero-delay C3 late-post gain was larger than the full post-onset average, while delayed C3 and all X3 effects were small or uncertain. This replication asks whether those patterns persist on many fresh streams under exactly the same model, cumulative-refit rule, label budgets and selected conditions. It does not add an adaptation algorithm or condition and is not external validation.

## Frozen design

- Conditions: original synthetic C3 and XOR synthetic X3 only; same generators and onset at monitoring row 4,000.
- Development seeds: 999001-999005 per task. Final seeds: 10000-11499 per task (1,500 per task, all included, no replacement). Disjoint from all earlier seed sets.
- Initial train/calibration/reference roles remain 4,000/2,000/2,000; training-only StandardScaler; LogisticRegression(C=1, max_iter=500, random_state=seed). As in the pilot, calibration/reference labels are not used by adaptation.
- Arms: frozen initial logistic versus cumulative refit logistic on original training plus all unique monitoring labels available strictly before each forecast. No tuning, recalibration, feature engineering or alternative updater.
- Delays: 0, 5 and 20, unchanged. Before prediction batch t, exact available budget is `500 * min(24, max(0, t-delay-1))`. Persist feature-only online records before releasing same-clock labels. Drain each queue to clock 24+delay without new inputs. Compare at exact logged budgets; this is update access versus no update, not an equal-label adaptive algorithm competition.
- Primary metric: paired seed difference in accuracy over all post-onset batches 9-24. Late post (17-24), pre (1-8), macro F1 and per-batch paths are secondary. No confidence/harm/detector claim is added.
- Uncertainty: 2,000 paired percentile bootstrap replicates over complete seed vectors; no batch or delay pseudoreplication. Report denominators and per-seed data.
- Evidence contracts: `online_batches.csv` contains IDs, delay, method, pre-prediction label budget, update audit, predicted positive rate, mean score and persistence marker only. `evaluator_batches.csv` contains correctness, accuracy, macro-F1 and class counts in a separate retrospective file. `release_log.csv` contains availability clocks and ID digests. Prediction arrays are stored by task/seed after the stream for deterministic evaluator joins; they are not claimed to be a security boundary.
- Stop rules: fail on any skipped seed, future/same-clock fit label, budget mismatch, ID overlap, non-finite scores, missing prediction row, non-drained queue or exact-clean-reproduction mismatch. No seed or condition replacement.

## Outputs

`results/phase3/adaptation_seed_replication_development`, `..._final`, and the separate report under `reports/phase3/adaptation_seed_replication_final`. Earlier adaptation pilot files remain unchanged.

## Sample-size amendment record

The initial 22:52 lock proposed final seeds 10000-10499 (500 per task). The five-seed development run took 8.6 seconds across both tasks, or about 0.86 seconds per task-seed. With the available work window, the final seed count was increased before any new final outcomes were available to 1,500 per task (10000-11499). The estimate is approximately 43 minutes for 3,000 task-seed cases before evaluation/reporting overhead. Models, generators, conditions, metrics, thresholds, labels, budgets, uncertainty method and stop rules were unchanged. This amendment is based on measured development runtime, not favorable outcomes.
