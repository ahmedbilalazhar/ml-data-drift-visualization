"""Multi-seed pilot: 8 templates x seeds 0-4, laptop rows (roadmap Sec. 8.3/9.5).

Purpose: measure cross-seed variability -> informs confirmatory seed-count
decision (precision needs). NOT confirmatory (shared pilot rows, pooled S0
calibration). Paired seeds across templates; streams are the independent units.
Writes results/aggregates/multiseed_detection.csv + multiseed_localization.csv.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
from drift_monitoring.models.frozen_models import fit_frozen
from drift_monitoring.orchestration.runner import run_stream_experiment
from drift_monitoring.detectors.drift_detectors import calibrate_threshold
from drift_monitoring.evaluation.metrics import match_events, paired_ci
from drift_monitoring.detectors.drift_detectors import ks_window_score
from drift_monitoring.explanations.localization import rank_features, localization_scores

MS = ROOT / "results" / "runs" / "_multiseed"
AGG = ROOT / "results" / "aggregates"
TIDS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
SEEDS = [0, 1, 2, 3, 4]
WINDOW = 500


def main():
    t0 = time.time()
    MS.mkdir(parents=True, exist_ok=True)
    # 1) run all streams (cached evidence reused on resume)
    for tid in TIDS:
        for s in SEEDS:
            rid = f"MS-{tid}-s{s}"
            if (MS / f"{rid}.evidence.csv").exists():
                continue
            df, truth = generate_stream(ScenarioSpec(template_id=tid, seed=s,
                                                     n_train=2000, n_calib=2000,
                                                     n_test=6000))
            models = fit_frozen(df[df.phase == "train"])
            run_stream_experiment(rid, df, truth, models, MS, window=WINDOW, seed=s)
    # 2) calibrate on S0 pool (pilot simplification — confirmatory uses independent streams)
    pool = pd.concat([pd.read_csv(MS / f"MS-S0-s{s}.evidence.csv") for s in SEEDS])
    thr = calibrate_threshold(pool.ks_global_D.tolist(), 10.0)
    # 3) detection stats per template over seeds
    det_rows = []
    for tid in TIDS:
        dets, delays, pre, bur = [], [], [], []
        for s in SEEDS:
            ev = pd.read_csv(MS / f"MS-{tid}-s{s}.evidence.csv")
            truth = json.loads((MS / f"MS-{tid}-s{s}.truth.json").read_text())
            alerts = ev[ev.ks_global_D >= thr].window_id.tolist()
            on = (None if truth.get("onset_test_row") is None
                  else int(truth["onset_test_row"] // WINDOW))
            end = (on + truth.get("transition_rows", 0) // WINDOW
                   if on is not None and tid == "S2" else None)
            m = match_events(alerts, on, end)
            dets.append(float(m["detected"]))
            delays.append(np.nan if m["delay_windows"] is None else float(m["delay_windows"]))
            pre.append(float(m["false_pre"]))
            bur.append(float(m["burden"]))
        lo, hi = paired_ci(np.array(dets), seed=0)
        det_rows.append({"template": tid, "threshold": round(thr, 4),
                         "detect_rate": round(float(np.mean(dets)), 2),
                         "detect_rate_ci95": f"[{lo:.2f},{hi:.2f}]",
                         "mean_delay_w": round(float(np.nanmean(delays)), 2)
                         if not all(np.isnan(delays)) else "NaN",
                         "mean_false_pre": round(float(np.mean(pre)), 2),
                         "mean_burden": round(float(np.mean(bur)), 2)})
        print(det_rows[-1])
    pd.DataFrame(det_rows).to_csv(AGG / "multiseed_detection.csv", index=False)
    # 4) localization means over seeds (drifted window 6, same as E2)
    loc_rows = []
    for tid in [t for t in TIDS if t not in ("S0", "S7")]:
        ps, rs, aps = [], [], []
        for s in SEEDS:
            df, truth = generate_stream(ScenarioSpec(template_id=tid, seed=s,
                                                     n_train=2000, n_calib=2000,
                                                     n_test=6000))
            num = [c for c in df.columns if c.startswith("x")]
            ref = df[df.phase == "train"].sample(1000, random_state=s)
            seg = df[df.phase == "test"].reset_index(drop=True).iloc[6*500:7*500]
            rk = rank_features(ks_window_score(ref, seg, num)["per_feature"], k=3)
            ts = {a for a in truth["affected_original_vars"] if a in num}
            sc = localization_scores(rk["ranking"], ts, k=3)
            ps.append(sc["p@k"]); rs.append(sc["r@k"]); aps.append(sc["ap"])
        loc_rows.append({"template": tid, "mean_P3": round(float(np.mean(ps)), 2),
                         "mean_R3": round(float(np.mean(rs)), 2),
                         "mean_AP": round(float(np.mean(aps)), 2)})
        print(loc_rows[-1])
    pd.DataFrame(loc_rows).to_csv(AGG / "multiseed_localization.csv", index=False)
    print(f"done in {time.time()-t0:.0f}s, thr={thr:.4f}")

if __name__ == "__main__":
    main()
