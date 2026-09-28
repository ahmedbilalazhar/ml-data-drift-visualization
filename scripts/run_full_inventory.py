"""Full-inventory runner (roadmap Sec. 8.3): G4-GATED. Do NOT run before freeze sign-off.

Modes (sequential, one job at a time, resumable — safe for Colab):
  short      : 8 templates x N seeds at full 10k/10k/30k (default seeds 0-4 → 40
               streams; x3 detector configs counted analytically — KS + W scored
               in the same pass, joint-classifier reserved)
  stationary : 3 S0 seeds x 1000 windows each, chunked generation (long null)
  all        : short + stationary

Budget worksheet: per-job seconds + artifact bytes -> results/aggregates/budget.csv
with median/p80, projected totals, and 25% contingency (roadmap Sec. 8.3).
Registry: appends to experiments/registries/runs.csv (never rewrites history).

Usage (Colab, after mounting Drive at /content/drive and cd to checkout):
  !python scripts/run_full_inventory.py --mode short --seeds 0 1 2 3 4
"""
import sys, time, tracemalloc, argparse, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
from drift_monitoring.models.frozen_models import fit_frozen
from drift_monitoring.orchestration.runner import run_stream_experiment, Registry, config_hash

FULL = ROOT / "results" / "runs" / "_full"
STAT = ROOT / "results" / "runs" / "_stationary"
AGG = ROOT / "results" / "aggregates"
REG = ROOT / "experiments" / "registries" / "runs.csv"
TIDS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]


def job(tid: str, seed: int, outdir: Path, n_train: int, n_calib: int,
        n_test: int, reg: Registry) -> dict:
    rid = f"FULL-{tid}-s{seed}"
    if (outdir / f"{rid}.evidence.csv").exists():
        return {"run_id": rid, "status": "skipped-complete", "seconds": 0.0, "bytes": 0}
    tracemalloc.start()
    t0 = time.time()
    try:
        df, truth = generate_stream(ScenarioSpec(template_id=tid, seed=seed,
                                                 n_train=n_train, n_calib=n_calib,
                                                 n_test=n_test))
        models = fit_frozen(df[df.phase == "train"])
        csv = run_stream_experiment(rid, df, truth, models, outdir, window=500, seed=seed)
        secs = time.time() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        b = csv.stat().st_size
        reg.mark(rid, config_hash({"t": tid, "s": seed}), "complete",
                 int(pd.read_csv(csv).__len__()))
        return {"run_id": rid, "status": "complete", "seconds": round(secs, 1),
                "peak_mb": round(peak / 1e6, 1), "bytes": b}
    except Exception as e:  # failure is a logged outcome, never silent
        tracemalloc.stop()
        reg.mark(rid, config_hash({"t": tid, "s": seed}), f"failed: {type(e).__name__}",
                 -1)
        return {"run_id": rid, "status": f"failed: {e}", "seconds": 0.0, "bytes": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["short", "stationary", "all"], default="short")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--templates", nargs="+", default=TIDS)
    a = ap.parse_args()
    print("G4 gate check: confirm experiments/protocols/freeze_g4_draft.md is SIGNED before use.")
    reg = Registry(REG)
    recs = []
    if a.mode in ("short", "all"):
        FULL.mkdir(parents=True, exist_ok=True)
        for tid in a.templates:
            for s in a.seeds:
                r = job(tid, s, FULL, 10_000, 10_000, 30_000, reg)
                recs.append(r)
                print(r, flush=True)
    if a.mode in ("stationary", "all"):
        STAT.mkdir(parents=True, exist_ok=True)
        for s in [0, 1, 2]:  # 3 independent generator seeds; 1000 windows x 500 rows
            r = job("S0", 1000 + s, STAT, 10_000, 10_000, 500_000, reg)
            r["run_id"] = f"STAT-S0-{s}"
            recs.append(r)
            print(r, flush=True)
    df = pd.DataFrame(recs)
    done = df[df.status == "complete"]
    if len(done):
        med, p80 = float(done.seconds.median()), float(done.seconds.quantile(0.8))
        proj = p80 * 147  # proposed detector total (roadmap Sec. 8.3)
        budget = pd.DataFrame([{"jobs_done": len(done), "median_s": med, "p80_s": p80,
                                "projected_147jobs_h": round(proj / 3600, 1),
                                "with_25pct_contingency_h": round(proj * 1.25 / 3600, 1),
                                "total_artifact_mb": round(done.bytes.sum() / 1e6, 1)}])
        AGG.mkdir(parents=True, exist_ok=True)
        budget.to_csv(AGG / "budget.csv", index=False)
        print(budget.to_string(index=False))


if __name__ == "__main__":
    main()
