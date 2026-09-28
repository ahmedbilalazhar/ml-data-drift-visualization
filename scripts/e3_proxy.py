"""E3 action-policy proxy (no humans, roadmap Sec. 9.3 + 9.4 fallback).

Fixed evidence-appropriate rule applied to cached E0 evidence:
- no alert -> continue monitoring
- alert + observed macro-F1 drop >=0.05 for 2 eligible windows -> evaluate update
- alert + F1 unavailable/no drop -> investigate data quality / request labels
- S7 (no alert by design): correct answer is request-labels/monitor, NEVER
  'no problem proven' — silent detector must not become false reassurance.

Scored against mechanism + realized F1 (pilot only, not confirmatory).
"""
import sys, json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "aggregates"

def decide(alert: bool, f1_base: float, f1_late: float) -> str:
    import math
    if not alert:
        if isinstance(f1_late, float) and not math.isnan(f1_late) \
           and not math.isnan(f1_base) and (f1_base - f1_late) >= 0.05:
            return "request-labels"  # S7 case: harm visible only via labels
        return "monitor"
    if isinstance(f1_late, float) and not math.isnan(f1_late) \
       and not math.isnan(f1_base) and (f1_base - f1_late) >= 0.05:
        return "evaluate-update"
    return "investigate/request-labels"

def main():
    e1 = pd.read_csv(OUT / "e1_detection.csv").set_index("template")
    rows = []
    for tid in ["S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7"]:
        ev = pd.read_csv(RUNS / f"E0-{tid}-s0.evidence.csv")
        base = float(ev.macro_f1.iloc[:3].mean())
        late = float(ev.macro_f1.iloc[6:9].mean())
        alert = bool(e1.loc[tid, "detected"])
        action = decide(alert, base, late)
        rows.append({"template": tid, "alert": alert,
                     "f1_base": round(base, 3), "f1_late": round(late, 3),
                     "dF1": round(base - late, 3), "action": action})
        print(f"{tid}: alert={alert} dF1={base-late:.3f} -> {action}")
    pd.DataFrame(rows).to_csv(OUT / "e3_proxy.csv", index=False)
    print(f"-> {OUT/'e3_proxy.csv'}")

if __name__ == "__main__":
    main()
