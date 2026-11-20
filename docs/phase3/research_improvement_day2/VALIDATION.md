# Day 2 validation record

- UCI archive CRC passed; archive SHA-256 matches the locked source digest.
- Source-only audit: 13,910 rows across ten batches, six class IDs, 128 numeric feature positions, no non-finite values, no duplicate complete records; class counts and proportions are saved for every cohort.
- Primary corrected run completed on local CPU with 267/89/89 historical rows and whole batches 2-10 evaluated. Predictions, per-cohort metrics, per-class metrics, and 1,310 KS/Holm test records (ten cohorts including the batch-1 reference evaluation) are retained.
- The initial v1 run had a split-indexing defect and is superseded; it is retained for audit. Its values are excluded from conclusions. Corrected v2 implements the locked allocation.
- Day 3 exploratory sensitivity completed with a separately fixed question. It reuses the frozen v2 model and predictions; no refit or threshold selection.
- Existing `tests/phase3` suite: 35 passed in 17.87 seconds.
- Day 2 report compiled with Tectonic 0.17.0. All three PDF pages were rendered and visually inspected. No overfull/undefined-reference warnings; a non-fatal Fontconfig environment message remains.
- Day 3 report compiled; all three PDF pages were rendered and visually inspected. No overfull/undefined-reference warnings; a non-fatal Fontconfig environment message remains.
- Preservation hashes for submitted DOCX, implementation brief, and submitted archive match the Day 1 inventory. A clean extraction of the final Day 2 and Day 3 archives is recorded in their package JSON receipts.

The nine-batch interval coverage tally is descriptive on one dataset. It is not an inferential coverage guarantee.
