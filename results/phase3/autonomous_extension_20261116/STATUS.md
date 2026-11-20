# Dependence stability extension run ledger

- Started: 2026-11-16T13:19:24.255363+05:00
- Stopped: 2026-11-16T14:21:48.148775+05:00 at user request.
- Status: partial; held-out evaluation did not complete.
- Completed: protocol frozen, 5 focused tests passed, and development run completed across 30 seeds / 2,160 stationary batch records. Frozen thresholds: n=250 0.2098204758; n=500 0.1563885832; n=1,000 0.1213968712. Each produced 36/720 development alarms (1.2 per 24-batch stream).
- Stopped process: PID 32824; CPU at stop 289.94 seconds; no held-out CSV/checkpoints were written. The empty v1 evaluation directory is retained. This is not a negative or positive detector result.
- Exact safe resume command (new output suffix; original empty v1 directory is preserved): `C:\Users\hp\anaconda3\python.exe scripts\run_dependence_stability.py --mode evaluation --threshold-file results/phase3/dependence_stability_20261116_v1_development/threshold.json --out results/phase3/dependence_stability_20261116_v2_evaluation`
