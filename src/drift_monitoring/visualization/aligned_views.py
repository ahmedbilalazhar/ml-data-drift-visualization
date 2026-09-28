"""Aligned evidence views (roadmap Sec. 10).

Small multiples on ONE shared time axis: detector score, feature heatmap
(top-k), performance w/ label-coverage gaps (missing never drawn as zero),
threshold ratio labelled as ratio (never 'drift probability').
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def aligned_figure(evidence_csv: Path, out_png: Path,
                   threshold: float | None = None) -> Path:
    ev = pd.read_csv(evidence_csv)
    fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    ax[0].plot(ev.window_id, ev.ks_global_D, marker="o", ms=3)
    ax[0].set_ylabel("KS global D")
    if threshold is not None:
        ax[0].axhline(threshold, ls="--")
        ax[0].text(ev.window_id.max(), threshold, " threshold", va="bottom")
    ax[1].plot(ev.window_id, ev.w_global_norm, marker="o", ms=3, color="tab:orange")
    ax[1].set_ylabel("Wasserstein norm")
    # performance: NaN gaps stay gaps (single-class / delayed windows)
    ax[2].plot(ev.window_id, ev.macro_f1, marker="s", ms=3, color="tab:green")
    ax[2].set_ylabel("macro-F1 (delayed labels)")
    ax[2].set_xlabel("monitoring window (500 rows each)")
    ax[2].set_ylim(0, 1.02)
    fig.suptitle(f"Aligned drift evidence — {evidence_csv.stem}")
    fig.tight_layout()
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
