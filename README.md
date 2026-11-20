# Explainable Visualization and Monitoring of ML Data Drift

Explainable drift monitoring, delayed-label evaluation and visualization, with pilot experiments and Phase 3 research extensions.

Start with [experiment results and reproduction](EXPERIMENTS.md). For project context, scientific limits and open work, read the [AI agent handoff](prompts/PROJECT_HANDOFF_FOR_AI_AGENT.md).

## Quickstart (Windows, CPU)

```powershell
pip install -e ".[dev]"
python scripts\smoke_e0.py
pytest -q
```

Colab heavy tests: open `notebooks/colab/00_e0_smoke.ipynb`
and run cells top-to-bottom (uses same `src/` package).

## Repository structure

| Folder | What belongs here |
| --- | --- |
| [experiments/](experiments/) | Pilot protocols, experiment cards and the run registry |
| [docs/phase3/](docs/phase3/) | Phase 3 protocols, dataset provenance, validation and decision records |
| [configs/](configs/) | Dataset, scenario, method and study settings |
| [src/drift_monitoring/](src/drift_monitoring/) | Monitoring, simulation, evaluation and visualization package |
| [scripts/](scripts/) | Run, evaluate and report entry points |
| [results/aggregates/](results/aggregates/) | Published pilot result tables |
| [results/phase3/](results/phase3/) | Retained run results, validation receipts and selected replay inputs |
| [reports/](reports/) | Interpretation, manuscript material, figures and reports |
| [data/](data/) | Source manifests and selected external inputs |
| [tests/](tests/) | Unit tests and scientific contracts |
| [prompts/](prompts/) | Agent handoff and experiment navigation |

For an experiment, follow its protocol to the runner and configuration, then compare its tables under `results/` with its interpretation under `reports/`. Write new run outputs to a new folder so the published evidence stays available.

See the [architecture roadmap](docs/architecture/research-project-architecture-and-roadmap.md) for planned work.
