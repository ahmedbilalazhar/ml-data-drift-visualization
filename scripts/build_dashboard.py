"""Unified drift dashboard (Gap G1/G2 build target, Phase 2 prototype).

One figure, shared time axis, for a single template/seed:
  (1) detector scores (KS + Wasserstein + JS + ADWIN where present) with threshold,
  (2) top-k feature x time heatmap (KS D),
  (3) classifier macro-F1 under delayed labels.
Inputs: results/runs/E0-{tid}-s0.evidence.csv + results/aggregates/heatmap_{tid}_s0.csv
  (+ e1_detection.csv threshold). Falls back gracefully when heatmap is absent.
Writes results/figures/Dashboard_{tid}_s0.png
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "results" / "figures"
AGG = ROOT / "results" / "aggregates"
RUNS = ROOT / "results" / "runs"


def main(template="S2", seed=0, top_k=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ev = pd.read_csv(RUNS / f"E0-{template}-s{seed}.evidence.csv")
    try:
        thr = float(pd.read_csv(AGG / "e1_detection.csv")
                    .set_index("template").loc[template, "threshold"])
    except Exception:
        thr = None
    hm_path = AGG / f"heatmap_{template}_s{seed}.csv"
    has_hm = hm_path.exists()
    if has_hm:
        mat = pd.read_csv(hm_path).set_index("window_id")
        top = mat.max().sort_values(ascending=False).head(top_k).index.tolist()
        n_panels, hm_idx = 3, 1
    else:
        n_panels, hm_idx = 2, -1
    fig, ax = plt.subplots(n_panels, 1, figsize=(11, 3.0 * n_panels + 1),
                           sharex=True)
    # (1) scores
    ax[0].plot(ev.window_id, ev.ks_global_D, marker="o", ms=3, label="KS D")
    if "w_global_norm" in ev.columns:
        ax[0].plot(ev.window_id, ev.w_global_norm, marker="o", ms=3,
                   label="Wasserstein norm")
    if "js_global_JS" in ev.columns:
        ax[0].plot(ev.window_id, ev.js_global_JS, marker="o", ms=3,
                   label="JS [0,1]")
    if "adwin_global" in ev.columns:
        ax[0].plot(ev.window_id, ev.adwin_global, marker="o", ms=3,
                   label="ADWIN shift")
    if thr is not None:
        ax[0].axhline(thr, ls="--", label="S0 threshold")
    ax[0].set_ylabel("drift score")
    ax[0].legend(loc="upper left", fontsize=8, ncol=2)
    # (2) heatmap
    if has_hm:
        im = ax[hm_idx].imshow(mat[top].T.values, aspect="auto", origin="lower")
        ax[hm_idx].set_yticks(range(len(top)))
        ax[hm_idx].set_yticklabels(top)
        ax[hm_idx].set_ylabel(f"top-{top_k} features")
        fig.colorbar(im, ax=ax[hm_idx], label="KS D")
    # (3) model impact under delayed labels
    f1 = ax[-1]
    f1.plot(ev.window_id, ev.macro_f1, marker="s", ms=3, color="tab:green")
    f1.set_ylabel("macro-F1 (delayed-2w)")
    f1.set_xlabel("monitoring window")
    f1.set_ylim(0, 1.02)
    fig.suptitle(f"Unified drift dashboard — {template} s{seed} "
                 f"(detection + localization + model impact)")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"Dashboard_{template}_s{seed}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"-> {out}")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2] or ["S2"])
