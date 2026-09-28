# Explainable Visualization and Monitoring of ML Data Drift

E0 pilot + reproducible harness (roadmap Sec. 6, 11–13, 16).

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
  detectors, explanations, frozen models, evaluation, visualization, orchestration
- `configs/` — dataset / scenario / method / study YAML
- `experiments/protocols/` — protocol outline + E0 card; `experiments/registries/` — run log
- `docs/` — charter, problem statement, data cards, decisions
- `reports/literature/` — evidence matrix (Week-1)
- `tests/` — scientific-invariant checks (leakage, label timing, resume)

Roadmap: `Research_Project_Architecture_and_Roadmap.md` (planning doc, not results).
