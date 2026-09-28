"""E6 sensitivity (roadmap Sec. 8.4, 8.6): window-size + resource probe — LAPTOP-light.

- Window sensitivity: S2 pilot rows at windows 250/500/1000 (detector behavior).
- NO full-scale run here: full 10k/10k/30k inventory + long stationary validation
  are Colab jobs (see notebooks/colab/01_heavy_inventory.ipynb). This script only
  writes the laptop-side sensitivity table so the protocol freeze has pilot data.
"""
import sys, time, tracemalloc
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
from drift_monitoring.models.frozen_models import fit_frozen
from drift_monitoring.orchestration.runner import run_stream_experiment

OUT = ROOT / "results" / "aggregates"


def main():
    tracemalloc.start()
    rows = []
    for w in [250, 500, 1000]:
        t0 = time.time()
        df, truth = generate_stream(ScenarioSpec(template_id="S2", seed=0,
                                                 n_train=2000, n_calib=2000,
                                                 n_test=6000))
        models = fit_frozen(df[df.phase == "train"])
        tmp = ROOT / "results" / "runs" / f"_sens_w{w}"
        csv = run_stream_experiment(f"E6-S2-w{w}", df, truth, models, tmp,
                                    window=w, seed=0)
        ev = pd.read_csv(csv)
        secs = time.time() - t0
        rows.append({"window": w, "n_windows": len(ev),
                     "ks_max": round(float(ev.ks_global_D.max()), 3),
                     "seconds": round(secs, 1)})
        print(rows[-1])
    cur, peak = tracemalloc.get_traced_memory()
    print(f"peak RAM {peak/1e6:.1f} MB (laptop pilot; heavy inventory -> Colab)")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "e6_window.csv", index=False)
    print(f"-> {OUT/'e6_window.csv'}")


if __name__ == "__main__":
    main()
