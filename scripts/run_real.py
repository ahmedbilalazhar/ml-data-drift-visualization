"""Real-data runs (roadmap Tier B + C): Elec2 burden, Covertype intervention.

Honesty rules enforced here:
- Elec2: NO event truth -> alarm burden + score/F1 association ONLY. The pilot
  S0 threshold is shown ILLUSTRATIVELY (not calibrated for this stream).
- Covertype: intervention truth known (shift point + cols); natural drift NOT
  claimed. Replay order explicitly constructed; labels native multiclass.
- F1 retrospective (full labels, marked) — operational delayed view stays E3.
Laptop-OK: Elec2 test ~27k rows / Covertype test ~50k rows, window 500.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.detectors.drift_detectors import ks_window_score
from drift_monitoring.evaluation.metrics import window_macro_f1
from drift_monitoring.windows.replay import make_windows
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROC = ROOT / "data" / "processed"
OUT = ROOT / "results" / "runs" / "_real"
AGG = ROOT / "results" / "aggregates"


def load_split(name: str, drop_like: tuple = ("date",)):
    df = pd.read_parquet(PROC / name).sort_values("event_id").reset_index(drop=True)
    feats = [c for c in df.columns if c not in ("event_id", "label")
             and not any(d in c.lower() for d in drop_like)]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    rep = json.loads((ROOT / "data" / "interim" / "audit_report.json").read_text())
    key = "elec2" if "elec2" in name else "covtype_subset"
    sp = rep["splits"][key]
    n = len(df)
    tr = df.iloc[:int(n * 0.2)];
    ca = df.iloc[int(n * 0.2):int(n * 0.4)];
    te = df.iloc[int(n * 0.4):].reset_index(drop=True)
    return df, tr, ca, te, num


def fit_models(tr: pd.DataFrame, num: list[str]):
    X, y = tr[num].to_numpy(float), tr.label.to_numpy()
    tree = Pipeline([("sc", StandardScaler()),
                     ("clf", DecisionTreeClassifier(max_depth=6, random_state=0))]).fit(X, y)
    lr = Pipeline([("sc", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=1000, C=1.0))]).fit(X, y)
    return {"tree": tree, "logreg": lr}


def run_stream(tag: str, te: pd.DataFrame, tr: pd.DataFrame, num: list[str],
               models: dict, truth: dict, window: int = 500) -> pd.DataFrame:
    ref = tr.sample(min(5000, len(tr)), random_state=0)
    recs = []
    for w in make_windows(len(te), window, window):
        if w.get("partial"):
            continue
        seg = te.iloc[w["start"]:w["end"]]
        ks = ks_window_score(ref[num], seg[num], num)
        pred = models["tree"].predict(seg[num].to_numpy(float))
        recs.append({"run_id": tag, "window_id": w["window_id"],
                     "ks_global_D": ks["global_D"],
                     "macro_f1": window_macro_f1(seg.label.to_numpy(), pred),
                     "label_status": "retrospective-full", "n": len(seg),
                     "evidence_status": "measured"})
    ev = pd.DataFrame(recs)
    OUT.mkdir(parents=True, exist_ok=True)
    ev.to_csv(OUT / f"{tag}.evidence.csv", index=False)
    (OUT / f"{tag}.truth.json").write_text(json.dumps(truth, indent=1, default=str))
    return ev


def main():
    t0 = time.time()
    pilot_thr = float(pd.read_csv(AGG / "multiseed_detection.csv")
                      .threshold.iloc[0])  # ILLUSTRATIVE on real streams
    rows = []

    # ---- Elec2: burden only ----
    _, tr, ca, te, num = load_split("elec2_ordered.parquet")
    models = fit_models(tr, num)
    ev = run_stream("REAL-elec2", te, tr, num, models,
                    {"stream": "elec2", "event_truth": None,
                     "note": "burden/association only; threshold illustrative"})
    n_alert = int((ev.ks_global_D >= pilot_thr).sum())
    rows.append({"stream": "elec2", "n_windows": len(ev),
                 "alerts_at_pilot_thr": n_alert, "thr": pilot_thr,
                 "mean_f1": round(float(ev.macro_f1.mean()), 3),
                 "detect_rate": "N/A (no event truth)"})
    print(f"elec2: {len(ev)} windows, {n_alert} alerts @illustrative thr, meanF1={rows[-1]['mean_f1']}")

    # ---- Covertype: controlled intervention in test segment ----
    df, tr, ca, te, num = load_split("covtype_subset100k.parquet")
    quant = ["Elevation", "Aspect", "Slope"]
    onset_frac, shift = 0.4, 1.5
    onset_row = int(len(te) * onset_frac)
    scale = te[quant].std()
    te = te.copy()
    te[quant] = te[quant].astype(float)  # intervention adds fractional shift
    te.loc[onset_row:, quant] = te.loc[onset_row:, quant] + shift * scale
    models = fit_models(tr, num)
    ev = run_stream("REAL-covtype-S2like", te, tr, num, models,
                    {"stream": "covtype_subset", "intervention": "synthetic shift",
                     "affected": quant, "onset_test_row": onset_row,
                     "note": "intervention truth known; natural drift not claimed"})
    from drift_monitoring.evaluation.metrics import match_events
    alerts = ev[ev.ks_global_D >= pilot_thr].window_id.tolist()
    on_w = onset_row // 500
    m = match_events(alerts, on_w, None)
    # localization on first drifted window
    ref = tr.sample(min(5000, len(tr)), random_state=0)
    seg = te.iloc[(on_w + 1) * 500:(on_w + 2) * 500]
    per = ks_window_score(ref[num], seg[num], num)["per_feature"]
    ranked = sorted(per, key=lambda c: -per[c]["D"])
    hits = len(set(ranked[:3]) & set(quant))
    rows.append({"stream": "covtype-S2like", "n_windows": len(ev),
                 "alerts_at_pilot_thr": len(alerts), "thr": pilot_thr,
                 "mean_f1": round(float(ev.macro_f1.mean()), 3),
                 "detect_rate": m["detected"], "delay_w": m["delay_windows"],
                 "loc_P3": round(hits / 3, 2)})
    print(f"covtype: {m}, loc_top3={ranked[:3]}")

    pd.DataFrame(rows).to_csv(AGG / "real_burden.csv", index=False)
    print(f"done in {time.time()-t0:.0f}s -> {AGG/'real_burden.csv'}")


if __name__ == "__main__":
    main()
