"""E3 delay sensitivity (roadmap Sec. 7.4, 9.3) — cached-evidence replay, no retraining.

Frozen core: labels never affect detectors, so KS outputs are reused verbatim.
Only performance AVAILABILITY shifts with delay d: at decision window 8, the
analyst sees F1 only for windows <= 8-d. Rule identical to e3_proxy.
Proves the S7 point: harm is actionable only after labels arrive.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "aggregates"
DECISION_W = 8


def decide(alert: bool, base: float, late: float) -> str:
    import math
    if not alert:
        if not math.isnan(base) and not math.isnan(late) and (base - late) >= 0.05:
            return "request-labels"
        return "monitor"
    if not math.isnan(base) and not math.isnan(late) and (base - late) >= 0.05:
        return "evaluate-update"
    return "investigate/request-labels"


def main():
    e1 = pd.read_csv(OUT / "e1_detection.csv").set_index("template")
    rows = []
    for tid in ["S0", "S1", "S2", "S6", "S7"]:
        ev = pd.read_csv(RUNS / f"E0-{tid}-s0.evidence.csv")
        base = float(ev.macro_f1.iloc[:3].mean())
        for d in [0, 2, 5]:
            avail = ev[ev.window_id <= DECISION_W - d]
            late = float(avail.macro_f1.tail(3).mean()) if len(avail) else float("nan")
            action = decide(bool(e1.loc[tid, "detected"]), base, late)
            rows.append({"template": tid, "delay": d,
                         "f1_base": round(base, 3),
                         "f1_late_avail": round(late, 3) if late == late else "NaN",
                         "dF1": round(base - late, 3) if late == late else "NaN",
                         "action": action})
            print(f"{tid} d={d}: dF1={base-late:.3f} -> {action}")
    pd.DataFrame(rows).to_csv(OUT / "e3_delay.csv", index=False)
    print(f"-> {OUT/'e3_delay.csv'}")


if __name__ == "__main__":
    main()
