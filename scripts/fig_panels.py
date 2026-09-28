"""Failure-panel figures for manuscript (roadmap Sec. 14.3): one honest panel.

Three annotated cases from cached pilot evidence (no recompute):
(a) S1 harmless shift — alert WITHOUT harm (must not retrain);
(b) S7 conditional harm — harm WITHOUT alert (must not reassure);
(c) S7 delay sweep — harm visible @d0/d2, hidden @d5 (delay cost).
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
FIG = ROOT / "results" / "figures"


def panel(ax, tid: str, note: str):
    ev = pd.read_csv(RUNS / f"E0-{tid}-s0.evidence.csv")
    ax.plot(ev.window_id, ev.ks_global_D, marker="o", ms=3, label="KS D")
    ax.plot(ev.window_id, ev.macro_f1, marker="s", ms=3, label="macro-F1")
    ax.set_title(f"{tid}: {note}")
    ax.set_xlabel("window")
    ax.legend(fontsize=8)


def main():
    fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    panel(ax[0], "S1", "alert, no harm → investigate, NOT retrain")
    panel(ax[1], "S7", "no alert, harm (dF1≈0.19) → request labels")
    ev = pd.read_csv(RUNS / "E0-S7-s0.evidence.csv")
    ax[2].plot(ev.window_id, ev.macro_f1, marker="s", ms=3, color="tab:green")
    ax[2].axvspan(8 - 5, 8, alpha=0.2, label="hidden @d5")
    ax[2].set_title("S7 delay: visible @d0/d2, hidden @d5")
    ax[2].set_xlabel("window")
    ax[2].legend(fontsize=8)
    fig.suptitle("Failure panels — where naive monitoring misleads (pilot)")
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "Panels_failure.png", dpi=150)
    print(f"-> {FIG/'Panels_failure.png'}")


if __name__ == "__main__":
    main()
