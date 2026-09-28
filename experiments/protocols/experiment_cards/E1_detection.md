# E1 — Detection (pilot: DONE laptop; confirmatory: GATED on G4 + Colab)

- Question: at matched false-alert burden, do KS/Wasserstein find controlled
  input changes within useful delays (incl. gradual, recurrent, joint)?
- Pilot: S0-calibrated thr 0.097 → S1–S6 delay 0–1; S7 silent (control).
- Confirmatory needs: independent calibration streams, 5 held-out seeds/template,
  long stationary validation, inclusion manifest. Script: scripts/e1_detection.py.
