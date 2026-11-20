"""TierA scale probe: one full-scale S2 stream (10k/10k/30k) through the runner.

Ungated single-stream check (the 40-stream inventory stays G4-gated).
Reports windows, seconds, peak MB. Expected envelope from 2026-09-29 probe:
~60 windows in ~32s at ~15 MB on laptop CPU.
"""
import sys, time, tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
from drift_monitoring.models.frozen_models import fit_frozen
from drift_monitoring.orchestration.runner import run_stream_experiment

OUT = ROOT / "results" / "runs" / "_tierA_probe"


def main(template="S2", seed=0):
    OUT.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    t0 = time.time()
    df, truth = generate_stream(ScenarioSpec(
        template_id=template, seed=seed, n_train=10000, n_calib=10000,
        n_test=30000))
    models = fit_frozen(df[df.phase == "train"])
    csv = run_stream_experiment(f"TIERA-{template}-s{seed}", df, truth,
                                models, OUT, window=500, seed=seed)
    import pandas as pd
    n = len(pd.read_csv(csv))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"TierA {template}: {len(df)} rows -> {n} windows in "
          f"{time.time()-t0:.1f}s, peak {peak/1e6:.1f} MB -> {csv.name}")


if __name__ == "__main__":
    main()
