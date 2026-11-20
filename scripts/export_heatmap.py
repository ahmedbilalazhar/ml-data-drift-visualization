"""Top-k feature heatmap export (P4-1): time x feature drift-score matrix.

Recomputes per-feature KS D per monitoring window for one template (default
S2 gradual-relevant) and writes a matrix CSV + PNG heatmap. Self-contained:
does not change E0 evidence schema. Proves the 'ranked drifting-feature
panel / feature x time heatmap' missing layer from Gap G1/G3.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.simulation.generator import ScenarioSpec, generate_stream
from drift_monitoring.detectors.drift_detectors import ks_window_score
from drift_monitoring.windows.replay import ReplayConfig, ReplayClock

OUT = ROOT / "results" / "aggregates"
FIG = ROOT / "results" / "figures"


def main(template="S2", seed=0, window=500, top_k=6):
    df, truth = generate_stream(ScenarioSpec(template_id=template, seed=seed,
                                             n_train=2000, n_calib=2000,
                                             n_test=6000))
    num = [c for c in df.columns if c.startswith("x")]
    ref = df[df.phase == "train"].sample(1000, random_state=0)
    test = df[df.phase == "test"].reset_index(drop=True)
    clock = ReplayClock(test, ReplayConfig(window=window, stride=window,
                                           label_delay_windows=2))
    rows = []
    for w, seg, _ in clock.iter_steps():
        ks = ks_window_score(ref, seg, num)
        rec = {"window_id": w["window_id"]}
        for c in num:
            rec[c] = ks["per_feature"][c]["D"]
        rows.append(rec)
    mat = pd.DataFrame(rows).set_index("window_id")
    # top-k by max over time (most drifted features overall)
    top = mat.max().sort_values(ascending=False).head(top_k).index.tolist()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    mat.to_csv(OUT / f"heatmap_{template}_s{seed}.csv")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mat[top].T.values, aspect="auto", origin="lower")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top)
    ax.set_xlabel("monitoring window")
    ax.set_ylabel("feature (top-k by max KS D)")
    fig.colorbar(im, ax=ax, label="KS D")
    fig.suptitle(f"Feature x time drift heatmap — {template} s{seed} "
                 f"(truth={truth.get('affected_original_vars')})")
    fig.tight_layout()
    fig.savefig(FIG / f"Heatmap_{template}_s{seed}.png", dpi=150)
    plt.close(fig)
    print(f"top-{top_k}: {top}")
    print(f"-> {OUT / f'heatmap_{template}_s{seed}.csv'}")
    print(f"-> {FIG / f'Heatmap_{template}_s{seed}.png'}")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2] or ["S2"])
