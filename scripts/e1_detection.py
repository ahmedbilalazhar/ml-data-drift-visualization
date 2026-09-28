"""E1 detection pilot (roadmap Sec. 9.1, 8.5) — local, cached E0 evidence.

- Calibrate KS threshold on S0 stationary scores (10 alerts / 1000 windows).
- Apply SAME threshold to S1-S7; match events (abrupt: onset+5; S2 gradual:
  start through end+5); one-to-one; S7 reported as blind-spot control.
- Writes results/aggregates/e1_detection.csv + results/figures/E1_tradeoff.png.
Pilot scale only — NOT confirmatory (needs independent calibration + held-out seeds).
"""
import sys, json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.detectors.drift_detectors import calibrate_threshold
from drift_monitoring.evaluation.metrics import match_events

RUNS = ROOT / "results" / "runs"
AGG = ROOT / "results" / "aggregates"
FIG = ROOT / "results" / "figures"
WINDOW = 500

def onset_window(truth: dict) -> int | None:
    o = truth.get("onset_test_row")
    return None if o is None else int(o // WINDOW)

def main():
    s0 = pd.read_csv(RUNS / "E0-S0-s0.evidence.csv")
    thr = calibrate_threshold(s0.ks_global_D.tolist(), 10.0)
    rows = []
    for tid in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
        ev = pd.read_csv(RUNS / f"E0-{tid}-s0.evidence.csv")
        truth = json.loads((RUNS / f"E0-{tid}-s0.truth.json").read_text())
        alerts = ev[ev.ks_global_D >= thr].window_id.tolist()
        on = onset_window(truth)
        end = (on + truth.get("transition_rows", 0) // WINDOW
               if on is not None and truth["template"] == "S2" else None)
        m = match_events(alerts, on, end)
        rows.append({"template": tid, "mechanism": truth["mechanism"],
                     "threshold": round(thr, 4), "n_alerts": len(alerts),
                     "detected": m["detected"], "delay_w": m["delay_windows"]})
        print(f"{tid} ({truth['mechanism']}): alerts={alerts} -> {m}")
    AGG.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(AGG / "e1_detection.csv", index=False)
    # figure: max KS per template vs threshold (S7/S0 below = blind-spot demo)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([r["template"] for r in rows],
           [float(pd.read_csv(RUNS / f"E0-{r['template']}-s0.evidence.csv").ks_global_D.max())
            for r in rows])
    ax.axhline(thr, ls="--")
    ax.text(7, thr, " S0-calibrated threshold", va="bottom")
    ax.set_ylabel("max KS global D (pilot)")
    ax.set_xlabel("scenario template")
    fig.suptitle("E1 pilot: detection signal vs stationary-calibrated threshold")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "E1_tradeoff.png", dpi=150)
    print(f"threshold={thr:.4f} -> {AGG/'e1_detection.csv'} + {FIG/'E1_tradeoff.png'}")

if __name__ == "__main__":
    main()
