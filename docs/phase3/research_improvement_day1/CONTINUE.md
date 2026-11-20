# Continuation checkpoint

Current state: Day 1 report source, bibliography, findings, demonstration guide, and figures are in this directory. Tectonic 0.17.0 compiled the report; all three pages were rendered and inspected; a clean extraction check passed. Existing source run receipts are historical evidence and were not rerun in this pass.

## Next actions

1. Optional: run the full numerical reproduction commands in fresh output directories.
2. Recompute source and archive hashes before any later-day work.
3. Continue Day 2 work only as a separately versioned package.

## Commands

```powershell
tectonic docs/phase3/research_improvement_day1/REPORT.tex --outdir docs/phase3/research_improvement_day1 --keep-logs
```

Use the bundled runtime Python executable noted in `docs/phase3/REPRODUCTION.md` for PDF page rendering. Full study reproduction commands are in `docs/phase3/23I-2581_23I-2604_Phase3_Reproduction.md` and the adaptation reproduction script.
