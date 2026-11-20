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
    has_w = "w_global_norm" in ev.columns
    has_js = "js_global_JS" in ev.columns
    has_aw = "adwin_global" in ev.columns
    panels = ["ks"] + (["w"] if has_w else []) + (["js"] if has_js else []) \
        + (["adwin"] if has_aw else []) + ["f1"]
    n_panels = len(panels)
    fig, ax = plt.subplots(n_panels, 1, figsize=(10, 2.6 * n_panels + 1),
                           sharex=True)
    if n_panels == 1:
        ax = [ax]
    ax[0].plot(ev.window_id, ev.ks_global_D, marker="o", ms=3)
    ax[0].set_ylabel("KS global D")
    if threshold is not None:
        ax[0].axhline(threshold, ls="--")
        ax[0].text(ev.window_id.max(), threshold, " threshold", va="bottom")
    i = 1
    if has_w:
        ax[i].plot(ev.window_id, ev.w_global_norm, marker="o", ms=3,
                   color="tab:orange")
        ax[i].set_ylabel("Wasserstein norm")
        i += 1
    if has_js:
        ax[i].plot(ev.window_id, ev.js_global_JS, marker="o", ms=3,
                   color="tab:purple")
        ax[i].set_ylabel("JS distance [0,1]")
        ax[i].set_ylim(0, 1.02)
        i += 1
    if has_aw:
        ax[i].plot(ev.window_id, ev.adwin_global, marker="o", ms=3,
                   color="tab:red")
        ax[i].set_ylabel("ADWIN mean-shift")
        i += 1
    # performance: NaN gaps stay gaps (single-class / delayed windows)
    ax[i].plot(ev.window_id, ev.macro_f1, marker="s", ms=3, color="tab:green")
    ax[i].set_ylabel("macro-F1 (delayed labels)")
    ax[i].set_xlabel("monitoring window (500 rows each)")
    ax[i].set_ylim(0, 1.02)
    fig.suptitle(f"Aligned drift evidence — {evidence_csv.stem}")
    fig.tight_layout()
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
