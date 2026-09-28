# Incident 001 — Covertype figshare mirror 403 (2026-09-28)

- Affected: `scripts/download_datasets.py` via `sklearn.datasets.fetch_covtype`
  (mirror `https://ndownloader.figshare.com/files/5976039` → HTTP 403, retry then fail).
- Detection: immediate download traceback on laptop; no partial artifact kept.
- Root cause: mirror access forbidden from this network (UCI origin reachable, HEAD 200).
- Correction: switched to UCI direct
  `https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz`
  with sha256 in `data/manifests/covtype_manifest.json`. No scientific run IDs affected
  (no confirmatory inventory started; G4 unsigned).
- Prevention: manifests record the exact URL used; rerun verifies checksum before use.
