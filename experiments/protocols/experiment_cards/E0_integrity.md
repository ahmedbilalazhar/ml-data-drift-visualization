# E0 — Integrity & Feasibility (confirmatory: NO; gate G3)

- RQ: E0. Stream generator S0–S7 valid? Frozen replay + detectors run in budget?
- Units: 8 templates x 1 pilot seed, window 500, ref 5000-rule (scaled to pilot rows).
- Frozen: ScenarioSpec defaults; tree-d6 + logreg-C1; KS + Wasserstein; delay 2.
- Accept: all 8 evidence CSVs validate (window counts, unique ids, finite scores,
  label joins consistent); peak RAM < 8GB target; resume from checkpoint == fresh run.
- Budget: pilot rows only (2k/2k/6k); full 10k/10k/30k inventory NOT spent here.
