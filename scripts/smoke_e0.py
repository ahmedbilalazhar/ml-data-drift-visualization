"""E0 integrity + feasibility smoke (roadmap Sec. 8.2 E0, gate G3).

Checks: valid data flow, affordable execution, replay contracts
(predict-before-label, frozen reference), checkpoint resume.
Writes evidence CSVs + one aligned figure per template into results/runs/.
"""
import argparse
import sys, time, tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
from drift_monitoring.models.frozen_models import fit_frozen
from drift_monitoring.orchestration.runner import run_stream_experiment

_FALLBACK_TIDS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]


def load_templates():
    import yaml
    p = ROOT / "configs" / "scenarios" / "e0_core.yaml"
    if p.exists():
        t = (yaml.safe_load(p.read_text()) or {}).get("templates")
        if t:
            return list(t)
    return list(_FALLBACK_TIDS)

def main(output_dir: Path = ROOT / "results"):
    tracemalloc.start()
    t0 = time.time()
    run_dir = output_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    for tid in load_templates():
        spec = ScenarioSpec(template_id=tid, seed=0,
                            n_train=2000, n_calib=2000, n_test=6000)
        df, truth = generate_stream(spec)
        # leakage guard: models see train only
        models = fit_frozen(df[df.phase == "train"])
        csv = run_stream_experiment(f"E0-{tid}-s0", df, truth, models, run_dir,
                                    window=500, seed=0)
        print(f"{tid}: windows={len(__import__('pandas').read_csv(csv))} -> {csv.name}")
    from drift_monitoring.visualization.aligned_views import aligned_figure
    for tid in ["S0", "S2", "S7"]:
        aligned_figure(run_dir / f"E0-{tid}-s0.evidence.csv",
                       output_dir / "figures" / f"E0-{tid}-s0.png")
    cur, peak = tracemalloc.get_traced_memory()
    print(f"E0 done in {time.time()-t0:.1f}s, peak RAM {peak/1e6:.1f} MB "
          f"(cap: investigate >10GB, stop <15GB)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results",
                        help="Directory for the runs and figures subfolders.")
    main(parser.parse_args().out)
