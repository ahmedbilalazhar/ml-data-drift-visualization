"""E1 comparator + reference stability (roadmap Sec. 8.1 fairness, 8.4).

(a) Wasserstein vs KS on the SAME multiseed evidence: calibrate W threshold on
    pooled S0 (10/1000), match events identically, report detect rates side by side.
(b) Reference-subsample stability: S2/seed0 reference at 1000/2500/5000 rows ->
    KS_max on the fixed drifted window (tests the 'frozen summaries' assumption).
Laptop-light: (a) is pure replay; (b) is 3 small recomputes.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.detectors.drift_detectors import calibrate_threshold, ks_window_score
from drift_monitoring.evaluation.metrics import match_events
from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream

MS = ROOT / "results" / "runs" / "_multiseed"
AGG = ROOT / "results" / "aggregates"
TIDS = ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]
SEEDS = [0, 1, 2, 3, 4]


def main():
    # (a) comparator
    pool_k = pd.concat([pd.read_csv(MS / f"MS-S0-s{s}.evidence.csv") for s in SEEDS])
    thr_k = calibrate_threshold(pool_k.ks_global_D.tolist(), 10.0)
    thr_w = calibrate_threshold(pool_k.w_global_norm.tolist(), 10.0)
    rows = []
    for tid in TIDS:
        rk = rw = 0
        for s in SEEDS:
            import json
            ev = pd.read_csv(MS / f"MS-{tid}-s{s}.evidence.csv")
            truth = json.loads((MS / f"MS-{tid}-s{s}.truth.json").read_text())
            on = (None if truth.get("onset_test_row") is None
                  else int(truth["onset_test_row"] // 500))
            end = (on + truth.get("transition_rows", 0) // 500
                   if on is not None and tid == "S2" else None)
            ak = ev[ev.ks_global_D >= thr_k].window_id.tolist()
            aw = ev[ev.w_global_norm >= thr_w].window_id.tolist()
            rk += float(match_events(ak, on, end)["detected"])
            rw += float(match_events(aw, on, end)["detected"])
        rows.append({"template": tid, "ks_rate": rk / 5, "w_rate": rw / 5,
                     "ks_thr": round(thr_k, 4), "w_thr": round(thr_w, 4)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv(AGG / "e1_compare.csv", index=False)
    # (b) reference stability
    df, _ = generate_stream(ScenarioSpec(template_id="S2", seed=0,
                                         n_train=2000, n_calib=2000, n_test=6000))
    num = [c for c in df.columns if c.startswith("x")]
    train = df[df.phase == "train"]
    seg = df[df.phase == "test"].reset_index(drop=True).iloc[6*500:7*500]
    srows = []
    for n in [1000, 2000, 2500]:
        ref = train.sample(min(n, len(train)), random_state=0)
        ks = ks_window_score(ref, seg, num)
        srows.append({"ref_rows": len(ref), "ks_max": round(ks["global_D"], 3)})
        print(srows[-1])
    pd.DataFrame(srows).to_csv(AGG / "e6_refstability.csv", index=False)
    print(f"-> e1_compare.csv + e6_refstability.csv")


if __name__ == "__main__":
    main()
