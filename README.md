# Explainable Visualization and Monitoring of ML Data Drift

Explainable drift monitoring, delayed-label evaluation and visualization, with pilot experiments and Phase 3 research extensions.

Start with the [AI agent handoff](prompts/PROJECT_HANDOFF_FOR_AI_AGENT.md) for the project map, reported findings and open questions. The [Phase 3 run index](prompts/PHASE3_RUN_INDEX.md) links to retained result evidence.

## Quickstart (Windows, CPU)

```powershell
pip install -e ".[dev]"
python scripts\smoke_e0.py
pytest -q
```

Colab heavy tests: open `notebooks/colab/00_e0_smoke.ipynb`
and run cells top-to-bottom (uses same `src/` package).

## Layout

- `src/drift_monitoring/` — simulation (S0–S7), windows/replay clock,
  detectors (KS/Wasserstein/JS/ADWIN-inspired/RBF-MMD/joint-logreg),
  explanations (ranking + DDE lift/scores), frozen models, evaluation,
  visualization (aligned views + heatmaps), orchestration, real-data loader
- `configs/` — dataset (incl. elec2/covtype/weather-pending/TierA) /
  scenario (E0–E6/full) / method / study YAML — all runners read these
- `scripts/` — smoke_e0, run_multiseed, run_real, e1_detection/compare,
  e2_localization, e3_proxy/delay, e6_reference/sensitivity,
  export_heatmap, build_dashboard, compare_offline, demo_dde,
  materialize_tierA, audit/download_datasets
- `experiments/protocols/` — protocol outline + E0 card; `experiments/registries/` — run log
- `docs/` — charter, problem statement, data cards, decisions, architecture roadmap
- `reports/literature/` — evidence matrix + deep-research review (Week-1)
- `reports/manuscript/phase1/` — Phase 1 LaTeX report + compiled PDF
- `studies/tasks/` — course requirement briefs; `studies/rubrics/` — evaluation checklists
- `prompts/` — review/audit prompt templates
- `data/` — `raw/` (immutable, gitignored), `processed/`, `manifests/`, `interim/`
- `tests/` — scientific-invariant checks (leakage, label timing, resume)

Roadmap: `docs/architecture/research-project-architecture-and-roadmap.md` (planning doc, not results).
