"""E6 reference-policy ablation (roadmap Sec. 8.4): fixed vs past-only refresh.

- Fixed: reference = frozen train sample (current runs) — measures cumulative
  distance from deployment start.
- Refresh (past-only): reference = trailing 5000 rows immediately BEFORE the
  current window — measures local change rate. No future rows ever used;
  every refresh event is logged (history never silently erased: fixed scores
  are recomputed alongside for the comparison table).
- Same windows, same KS+Holm, same illustrative threshold. Score distributions
  reported; no transferred-threshold claims.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.detectors.drift_detectors import ks_window_score
from drift_monitoring.windows.replay import make_windows

PROC = ROOT / "data" / "processed"
AGG = ROOT / "results" / "aggregates"
THR = float(pd.read_csv(AGG / "multiseed_detection.csv").threshold.iloc[0])


def scores(df_test: pd.DataFrame, ref_fixed: pd.DataFrame, num: list[str],
           window: int = 500, ref_rows: int = 5000) -> pd.DataFrame:
    rows = []
    for w in make_windows(len(df_test), window, window):
        if w.get("partial"):
            continue
        seg = df_test.iloc[w["start"]:w["end"]]
        kf = ks_window_score(ref_fixed[num], seg[num], num)["global_D"]
        past = df_test.iloc[max(0, w["start"] - ref_rows):w["start"]]
        if len(past) >= 500:
            kr = ks_window_score(past[num], seg[num], num)["global_D"]
        else:
            kr = float("nan")  # warm-up: no past reference yet, NOT zero
        rows.append({"window_id": w["window_id"], "ks_fixed": kf, "ks_refresh": kr})
    return pd.DataFrame(rows)


def main():
    out = []
    for name, drop in [("elec2_ordered.parquet", ("date",)),
                       ("covtype_subset100k.parquet", ())]:
        df = pd.read_parquet(PROC / name).sort_values("event_id").reset_index(drop=True)
        num = [c for c in df.columns if c not in ("event_id", "label")
               and not any(d in c.lower() for d in drop)
               and pd.api.types.is_numeric_dtype(df[c])]
        n = len(df)
        tr, te = df.iloc[:int(n * 0.2)], df.iloc[int(n * 0.4):].reset_index(drop=True)
        ref = tr.sample(min(5000, len(tr)), random_state=0)
        ev = scores(te, ref, num)
        ev.to_csv(AGG / f"e6_refpolicy_{name.split('_')[0]}.csv", index=False)
        f = (ev.ks_fixed >= THR).sum()
        r = int((ev.ks_refresh >= THR).sum())
        out.append({"stream": name.split("_")[0], "n_windows": len(ev),
                    "mean_fixed": round(float(ev.ks_fixed.mean()), 3),
                    "mean_refresh": round(float(ev.ks_refresh.mean()), 3),
                    "alerts_fixed": int(f), "alerts_refresh": r,
                    "note": "thr illustrative; refresh warm-up NaN kept"})
        print(out[-1])
    pd.DataFrame(out).to_csv(AGG / "e6_refpolicy.csv", index=False)
    print(f"-> {AGG/'e6_refpolicy.csv'}")


if __name__ == "__main__":
    main()
