# Week-1 + E0 pilot report (G3)

- Scope executed: E0 pilot + harness, local Windows (Colab reserved for heavy runs).
- Tests: `pytest -q` → 4 passed (leakage, predict-before-label, S7 contract, train-only fit).
- Smoke: `python scripts/smoke_e0.py` → 8/8 templates x 12 windows, 25.5 s, peak 33 MB.
- Pilot signals (NOT confirmatory): S0 KS_max 0.098; S1/S5 ~0.71; S2–S4 ~0.54–0.58;
  S6 0.19 (marginal-test limit, joint detector needed); S7 0.098 with F1 0.66→0.54
  (feature-blind conditional harm — must be reported, never tuned away).
- Gates: G3 engineering feasibility PASSES on pilot scale. G1/G2/G4+ remain open:
  full 10k/10k/30k inventory, stationary calibration (1000 windows), natural-data
  cards, frozen protocol, and H4 study decision still required before confirmatory runs.
