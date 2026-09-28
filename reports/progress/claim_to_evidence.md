# Claim-to-evidence (practical track, pilot-scale; roadmap Sec. 14.3)

| Claim | Evidence | Limit / disconfirming case |
|---|---|---|
| KS/Wasserstein detect controlled input changes | E1: S1–S6 detected delay 0–1 at S0-calibrated thr 0.097 | S0 1/12 false alert (small-sample noise); needs independent 1000-window calibration (Colab) |
| Raw distance ranking localizes manipulated vars | E2: recall 1.0, AP 1.0; proxies dilute P@3 | Correlated proxies unavoidable; S6 joint truth is group-level |
| Feature-only detectors are blind to conditional-only harm | E1+S7: silent (same as S0) while dF1 0.19 | Core honest-monitoring result; interface must say unknown |
| Delay changes actionability, not detector output | E3 delay: S6 update@d0 → investigate@d2; S7 harm hidden@d5 → monitor (FALSE reassurance) | Long delay + silent detector = missed harm; must be a thesis failure panel |
| Windows 250/500/1000 all fire on S2 pilot | E6: KS_max 0.61/0.54/0.54 | Main 500 locked; sensitivity does not replace main result |
| Prototype fits student CPU | E0 25.5s/33MB; E6 ≤4.1s | Pilot rows only; FULL 120-job + stationary inventory is Colab-gated |
| Visualization improves analysts | NOT CLAIMED (H4 not attempted; heuristic checklist only) | Requires future 12–20 person study to claim |

Regenerate: `python scripts/smoke_e0.py` (laptop) → E1/E2/E3/E6 scripts → figures in `results/figures/`. Heavy: `notebooks/colab/01_heavy_inventory.ipynb`.
