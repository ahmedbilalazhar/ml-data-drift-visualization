# Data card — Covertype (CANDIDATE, NOT ADMITTED)

- Role if admitted: Tier C semi-synthetic — multiclass tabular, realistic feature
  structure, DOCUMENTED replay order + controlled feature interventions (known
  intervention locations; natural chronological drift NOT claimed).
- Source: UCI record (https://archive.ics.uci.edu/dataset/31/covertype) —
  581,012 observations, 54 features, 7 cover types. Keep native multiclass labels
  (no silent binarization); preserve feature groups (one-hot back to source var).
- Subset rule: fixed documented subset ≤100k rows with manifest (selection seed,
  row IDs, class mapping). Chunked loading; RAM measured (pilot: 100k×200×4B≈80MB
  array only — process memory higher).
- Audit required: license/redistribution, checksum, subset manifest, replay-order
  rationale (explicitly constructed, NOT natural time), intervention spec per
  scenario card, fit ranges (encoders/models on train split only), imbalance +
  minority-support check per window (suppress small-group stats per rule).
- Ground-truth boundary: intervention truth known; natural drift NOT established.
  Never present induced replay as "found natural drift."
- Status: ADMITTED-WITH-NOTES (2026-09-28, laptop audit `data/interim/audit_report.json`).
  Measured: full 581,012 × 55 (sha256 in `data/manifests/covtype_manifest.json`);
  stratified subset 83,665 rows (seed 0, ~14.3k/class except 4: 2,747 and 5: 9,493 —
  capped by source rarity, recorded in `covtype_subset_manifest.json`), 0 missing,
  0 dup ids/rows, in `data/processed/covtype_subset100k.parquet` with 20/20/60 split
  manifest. Source incident: sklearn figshare mirror 403 → UCI direct
  (`docs/incidents/001_covtype_figshare_403.md`). Notes: minority-class suppression
  rule applies; replay order explicitly constructed; interventions still to be
  specified per scenario card before semi-synthetic runs.
