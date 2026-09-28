"""E2 localization pilot (roadmap Sec. 9.2) — no humans.

For each template: reference (train) vs drifted test window -> KS ranking ->
P/R/F1@3 + AP vs generator truth. S7 truth is relationship-level (no single
feature cause); S6 truth is joint — both reported with limits, never forced.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
from drift_monitoring.detectors.drift_detectors import ks_window_score
from drift_monitoring.explanations.localization import rank_features, localization_scores

OUT = ROOT / "results" / "aggregates"

def truth_set(truth: dict, numeric: list[str]) -> set[str]:
    aff = [a for a in truth["affected_original_vars"] if a in numeric]
    if truth["template"] == "S7":
        return set()  # conditional-only: no feature cause by design
    if truth["template"] == "S0":
        return set()
    return set(aff)

def main():
    rows = []
    for tid in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
        df, truth = generate_stream(ScenarioSpec(template_id=tid, seed=0,
                                                 n_train=2000, n_calib=2000,
                                                 n_test=6000))
        num = [c for c in df.columns if c.startswith("x")]
        ref = df[df.phase == "train"].sample(1000, random_state=0)
        test = df[df.phase == "test"].reset_index(drop=True)
        seg = test.iloc[6 * 500:7 * 500]  # clearly post-onset window
        ks = ks_window_score(ref, seg, num)
        rk = rank_features(ks["per_feature"], k=3)
        ts = truth_set(truth, num)
        sc = localization_scores(rk["ranking"], ts, k=3)
        rows.append({"template": tid, "top3": ",".join(rk["top3"]),
                     "truth": ",".join(sorted(ts)) or "(none)",
                     **{k: round(v, 3) if isinstance(v, float) else v
                        for k, v in sc.items()}})
        print(f"{tid}: top3={rk['top3']} truth={sorted(ts)} -> {sc}")
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT / "e2_localization.csv", index=False)
    print(f"-> {OUT/'e2_localization.csv'}")

if __name__ == "__main__":
    main()
