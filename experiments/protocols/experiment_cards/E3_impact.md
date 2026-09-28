# E3 — Predictive relevance + delay (pilot: DONE laptop; confirmatory: GATED)

- Questions: (a) contemporaneous alarm↔harm association; (b) future warning value
  over history-only baseline at fixed horizon, chronological, no future leakage.
- Pilot proxy: fixed rule on cached evidence → S0 monitor, S1/S5 investigate
  (not retrain), S6 update@d0, S7 request-labels (silent detector ≠ fine).
- Delay finding: S6 update@d0→investigate@d2; S7 harm hidden@d5→monitor (MISS).
- Scripts: scripts/e3_proxy.py, scripts/e3_delay_sensitivity.py (replay, no retrain).
- Confirmatory: fixed horizon h (3 windows candidate), history baseline, block-aware
  uncertainty, purged overlaps, per-class secondaries.
