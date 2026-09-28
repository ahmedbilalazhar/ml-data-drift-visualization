# Supervisor handoff — one page (2026-09-28)

## Question being tested
Does an aligned view of drift scores + feature explanations + delayed performance
help analysts diagnose change and act appropriately (RQ4)? → INFEASIBLE solo;
fallback locked pre-results: empirical monitoring study (RQ1–RQ3 + proxy), viz as
prototype + heuristic review only. Charter amendment 2026-09-28.

## Done (laptop CPU, all regenerable)
E0 8/8×12w 25s/33MB (G3 pass) · E1 thr≈0.094, S1–S6 delay 0–1, S7 silent ·
E2 recall/AP 1.0, P@3 proxy-diluted · E3 proxy + delay (S6 update@d0→investigate@d2;
S7@d5 MISS) · E6 windows stable; ref-size stable (0.24–0.25) · KS≡Wasserstein on
easy shifts · 5-seed pilot, CIs · failure panels + results draft + G4 draft + cards.

## To sign (G4 freeze + heuristic reviewer)
`experiments/protocols/freeze_g4_draft.md` (threshold plan, seeds, matching,
degradation rule, exclusions) and `studies/rubrics/heuristic_checklist.md`
(30–45 min walkthrough). Two questions: (1) technical-only thesis acceptable?
(2) confirm 5 seeds/template + ≤24 ablations for Colab?

## Next (Colab only): full 10k/10k/30k ×5 seeds, independent calibration,
1000-window stationary, Elec2/Covertype audits (`notebooks/colab/01_heavy_inventory.ipynb`).
No natural-data or user-performance claims made until then.
Evidence: `results/aggregates/` (8 CSVs) · `results/figures/` · `reports/manuscript/`.
