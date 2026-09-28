# Data card — Elec2 / Electricity (CANDIDATE, NOT ADMITTED)

- Role if admitted: Tier B natural temporal external-validity case (chronological replay).
- Source: River `Elec2` documentation (https://riverml.xyz/latest/api/datasets/Elec2/);
  binary electricity-price-direction task. Upstream: NSW electricity market.
- Target/classes: price direction (up/down). Native binary — keep as-is.
- Time semantics: ordered market intervals; MUST verify timestamp column + row order
  from the downloaded manifest before any temporal claim.
- Audit required before admission: download stability, version/date, license/
  redistribution terms, checksum, schema, target definition, duplicate policy,
  missingness, class balance over time, subgroup support, prediction-time feature
  availability. Record fitting period of any normalized upstream copy.
- Ground-truth boundary: NO verified drift-event catalog exists for our evaluation.
  If admitted: report alarm burden + performance associations ONLY; never call an
  alarm false-positive for lack of label, never tune thresholds on the test chronology.
- Status: ADMITTED-WITH-NOTES (2026-09-28, laptop audit `data/interim/audit_report.json`).
  Measured: 45,312 rows, 0 missing, 0 dup ids/rows, classes {0: 26075, 1: 19237}.
  Raw immutable in `data/raw/` (sha256 in `data/manifests/elec2_manifest.json`);
  ordered parquet in `data/processed/elec2_ordered.parquet`; chronological 20/20/60
  split manifest in audit report. Notes: no drift-event catalog (burden/association
  only). VERIFIED 2026-09-28: correct variant — 45,312 rows = River Elec2 count;
  schema (nswprice/nswdemand/vicprice/vicdemand/transfer/day/period/date/class)
  = elecNormNew; task = binary NSW price up/down at 5-min intervals (SPLICE-2,
  matches River spec). File order chronological except 5 documented out-of-order
  rows [25488, 34896, 35232, 36240, 40704]; `date` normalized 0–1, excluded from
  the feature universe. See verification block in `data/manifests/elec2_manifest.json`.
