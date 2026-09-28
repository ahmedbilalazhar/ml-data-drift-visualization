# Reporting checklist (adapted NeurIPS Paper Checklist + FAIR + OSF, roadmap Sec. 13.1)

- Claims: every manuscript claim → row in `reports/progress/claim_to_evidence.md`.
  H4 (analyst improvement) explicitly NOT claimed (not attempted).
- Limitations: `failure_catalog.md` + results-chapter §7; S7/S1/delay/transfer cases kept.
- Methods: splits, reference rule, matching policy, delays, tuning cap (≤6), seeds,
  exclusion handling — frozen in `freeze_g4_draft.md` (unsigned = pilot only).
- Uncertainty: paired-CI over stream seeds; windows never treated as independent.
- Reproducibility: inspect (saved evidence) / smoke (`smoke_e0.py`) / full
  (`run_full_inventory.py`, Colab-gated); manifests + checksums; drill-verified resume.
- Compute: measured per-job seconds/peak MB in drill + budget worksheet (populated post-Colab).
- Data: raw immutable + gitignored; subset manifests with seeds/IDs; licenses in cards.
- Human study: none conducted; heuristic checklist filed as formative only.
- Assistance: AI coding assistant used for harness implementation; all outputs
  executed and verified locally by the author (see `docs/governance/contributions.md`).
