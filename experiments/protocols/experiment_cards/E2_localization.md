# E2 — Explanation localization (pilot: DONE laptop; confirmatory: GATED)

- Question: does raw distance ranking locate manipulated vars (vs proxies)?
- Pilot: recall 1.0/AP 1.0 everywhere with truth; P@3 diluted by correlated
  proxies; S0/S7 empty-truth flagged. Script: scripts/e2_localization.py.
- Confirmatory: held-out seeds, group-level scoring (S4/S6), stability resamples,
  no-change policy kept. No DDE inference in core (E5 extension only).
