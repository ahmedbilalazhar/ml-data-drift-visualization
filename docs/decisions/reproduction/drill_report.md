# Interruption/recovery drill report (week-4 requirement, 2026-09-28, laptop)

- Target: DRILL-STAT, S0 full-scale rows (10k/10k/120k), 240 windows, background job.
- Kill: job Running at 40 s → stopped. Checkpoint `{"next_window": 229}` survived;
  resume restarted exactly at 229, contiguous, no duplicates. Resume offset: PASS.
- FINDING (fail → fixed): resumed evidence file held only the 11 post-kill windows.
  Pre-fix runner buffered records in memory and wrote the CSV once at completion,
  so a killed run's finished windows existed only as a checkpoint number.
- Fix (`orchestration/runner.py`, code v2-incremental): per-window append BEFORE
  checkpoint write; resume reconciles `max(checkpoint, file_max+1)` (covers a kill
  landing between append and checkpoint); fresh start deletes stale files.
- Verification: pytest 4 passed; E0 re-ran byte-equivalent values; MERGE-TEST
  (3-window tiny run, simulated kill after window 1) → 3 rows, no dups, values
  identical to uninterrupted run.
- Old evidence (v1) remains valid: same per-window values, only write timing changed.
- Budget datum: ~229 full-scale windows in <40 s (≈0.2 s/window incl. fit) →
  a 60-window short job ≈ 15–25 s; 147-job detector total ≈ 0.5–1.5 CPU-hours on
  this laptop class — well under the roadmap's 11.5–46 h illustration. Colab
  still owns confirmatory (isolation + Post-G4 discipline), but cost is not a blocker.
- Drill artifacts: `results/runs/_drill/` (excluded from all aggregates).
