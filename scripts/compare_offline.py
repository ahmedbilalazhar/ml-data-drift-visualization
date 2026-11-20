"""Offline 5-detector comparison (P2 verification, roadmap E1 extension).

Same reference/test windows per template; reports JS / ADWIN-inspired /
RBF-MMD / joint-logreg scores side by side. S0 stationary and S7
conditional-only must read identical (input-blind by design); S2/S4 must
separate; S6 joint case is the honest hard case. Writes
results/aggregates/offline_compare.csv.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
from drift_monitoring.detectors.drift_detectors import (
    js_window_score, adwin_window_score, mmd_window_score, joint_window_score)

OUT = ROOT / "results" / "aggregates"


def main(templates=("S0", "S2", "S4", "S6", "S7"), seed=0):
    rows = []
    for tid in templates:
        df, _ = generate_stream(ScenarioSpec(
            template_id=tid, seed=seed, n_train=2000, n_calib=2000, n_test=6000))
        num = [c for c in df.columns if c.startswith("x")]
        ref = df[df.phase == "train"].sample(1000, random_state=0)
        seg = df[df.phase == "test"].iloc[3000:3500]
        rows.append({
            "template": tid,
            "js": round(js_window_score(ref, seg, num)["global_JS"], 3),
            "adwin": round(adwin_window_score(ref, seg, num)["global_ADWIN"], 3),
            "mmd": round(mmd_window_score(ref, seg, num)["global_MMD"], 3),
            "joint_auc": round(joint_window_score(ref, seg, num)["auc"], 3),
        })
        print(rows[-1])
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "offline_compare.csv", index=False)
    print(f"-> {OUT / 'offline_compare.csv'}")


if __name__ == "__main__":
    main()
