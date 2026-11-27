# Phase 4 demo

1. Open the [submitted paper](submissions/phase4/23I-2581_23I-2604_Phase4_Research_Paper.pdf). Use the architecture diagram to introduce input drift detection, accuracy estimation, and delayed label verification.
2. Run the CPU pilot from the repository root in a Python environment with the project installed.

   ```powershell
   python -m pip install -e ".[dev]"
   python scripts/smoke_e0.py --out tmp/phase4-demo
   ```

3. Open the [Phase 3 evidence replay](../reports/phase3/v2_final/23I-2581_23I-2604_Phase3_Evidence_Replay.html) and show the stationary control and conditional shift. Input evidence can remain quiet while accuracy drops. Then show the natural stream and prospective forecast results in the submitted paper.
4. Open the [experiment evidence map](../EXPERIMENTS.md) for the code and results behind each claim. The [paper source](../paper/) can build using its bundled tables and plots.
