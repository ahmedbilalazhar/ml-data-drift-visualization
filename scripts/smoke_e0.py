"""E0 integrity + feasibility smoke (roadmap Sec. 8.2 E0, gate G3).

Checks: valid data flow, affordable execution, replay contracts
(predict-before-label, frozen reference), checkpoint resume.
Writes evidence CSVs + one aligned figure per template into results/runs/.
"""
import sys, time, tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
from drift_monitoring.models.frozen_models import fit_frozen
from drift_monitoring.orchestration.runner import run_stream_experiment

OUT = ROOT / "results" / "runs"

def main():
    tracemalloc.start()
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    for tid in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
        spec = ScenarioSpec(template_id=tid, seed=0,
                            n_train=2000, n_calib=2000, n_test=6000)
        df, truth = generate_stream(spec)
        # leakage guard: models see train only
        models = fit_frozen(df[df.phase == "train"])
        csv = run_stream_experiment(f"E0-{tid}-s0", df, truth, models, OUT,
                                    window=500, seed=0)
        print(f"{tid}: windows={len(__import__('pandas').read_csv(csv))} -> {csv.name}")
    from drift_monitoring.visualization.aligned_views import aligned_figure
    for tid in ["S0", "S2", "S7"]:
        aligned_figure(OUT / f"E0-{tid}-s0.evidence.csv",
                       ROOT / "results" / "figures" / f"E0-{tid}-s0.png")
    cur, peak = tracemalloc.get_traced_memory()
    print(f"E0 done in {time.time()-t0:.1f}s, peak RAM {peak/1e6:.1f} MB "
          f"(cap: investigate >10GB, stop <15GB)")

if __name__ == "__main__":
    main()
