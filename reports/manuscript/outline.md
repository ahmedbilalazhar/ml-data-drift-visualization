# Manuscript / thesis outline (practical track; claims per claim_to_evidence.md)

1. **Intro:** deployed tabular classifier, features now / labels later; alarm ≠
   failure ≠ cause ≠ action. Contribution = evaluated diagnosis workflow +
   auditable protocol (no new detector, no user-performance claim).
2. **Related:** DriftLens (monitor+viz), DDE (selective localization), PHT
   (predictive impact) + DriftVis/ConceptExplorer (viz prior art kills "viz is new").
   Gap = no open reproducible study combining detection + localization + delayed
   harm + one visual layer on identical windows.
3. **Method:** S0–S7 generator; frozen tree/logreg; fixed reference; KS/Wasserstein;
   replay clock (predict-then-release, delays 0/2/5); evidence records; fixed proxy rule.
4. **Evaluation:** event matching (±5w), P/R/F1@3+AP with empty-truth policy,
   macro-F1 degradation (−0.05×2w), delay replay from cache, window sensitivity,
   heuristic checklist. Paired streams, inclusion manifests, failures kept.
5. **Results:** E1 table + tradeoff fig; E2 localization table; E3 proxy + delay
   table; E6 window table; resource table (laptop pilot; Colab gated).
6. **Failure panels figure** (`Panels_failure.png`): S1 false-alarm trap, S7 blind
   spot, S7@d5 miss — the paper's honesty core.
7. **Limitations:** pilot scale/seeds, no natural-data claim yet, proxy≠humans,
   H4 not attempted, threshold needs independent calibration.
8. **Repro:** inspect / smoke / full(Colab) entry points; manifests; G4 freeze.
