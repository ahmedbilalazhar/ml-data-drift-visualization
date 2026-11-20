"""Summarize the fresh-seed, exploratory capacity-control sensitivity."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/phase3/adaptation_capacity_control_confirmation_20261118_v1"
OUT = ROOT / "reports/phase3/adaptation_capacity_confirmation_20261118_v4"
TASKS = ("linear_probabilistic", "xor_nonlinear", "five_region")
LABEL = {"linear_probabilistic": "Linear", "xor_nonlinear": "XOR", "five_region": "Five-region"}
CONTRASTS = (("matched_local", "matched_global", "Local - global", "#087E8B"),
             ("matched_interaction", "matched_global", "Interaction - global", "#4263EB"),
             ("matched_local", "matched_interaction", "Local - interaction", "#D97706"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(run: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    metrics = pd.read_csv(run / "seed_metrics.csv")
    receipt = json.loads((run / "audit_receipt.json").read_text(encoding="utf-8"))
    wide = metrics.pivot(index=["seed", "task", "mechanism", "delay"], columns="method", values="accuracy")
    rng = np.random.default_rng(714001)
    rows = []
    for task in TASKS:
        for left, right, label, _ in CONTRASTS:
            d = wide[left] - wide[right]
            seed_values = d.xs(task, level="task").groupby(level="seed").mean().to_numpy()
            boot = seed_values[rng.integers(0, len(seed_values), size=(50_000, len(seed_values)))].mean(axis=1)
            rows.append({"task": task, "contrast": label, "seeds": len(seed_values),
                         "mean_pp": float(seed_values.mean() * 100),
                         "bootstrap_lower_pp": float(np.quantile(boot, .025) * 100),
                         "bootstrap_upper_pp": float(np.quantile(boot, .975) * 100)})
    effects = pd.DataFrame(rows)
    effects.to_csv(out / "independent_seed_effects.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titleweight": "bold", "figure.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(9.3, 4.5), constrained_layout=True)
    offsets = {"Local - global": -.20, "Interaction - global": 0.0, "Local - interaction": .20}
    colors = {label: color for _, _, label, color in CONTRASTS}
    labels = ["Local - global", "Interaction - global", "Local - interaction"]
    base = np.arange(len(TASKS))
    for label in labels:
        s = effects[effects.contrast == label].set_index("task")
        mean = np.array([s.loc[t, "mean_pp"] for t in TASKS])
        low = mean - np.array([s.loc[t, "bootstrap_lower_pp"] for t in TASKS])
        high = np.array([s.loc[t, "bootstrap_upper_pp"] for t in TASKS]) - mean
        ax.errorbar(mean, base + offsets[label], xerr=[low, high], fmt="o", ms=5, capsize=3,
                    lw=1.5, color=colors[label], label=label)
    ax.axvline(0, color="#334155", lw=1, ls="--")
    ax.set_yticks(base, [LABEL[t] for t in TASKS]); ax.invert_yaxis()
    ax.set_xlabel("Mean accuracy difference (percentage points)\n95% seed-bootstrap percentile interval; 10 seeds")
    ax.set_title("Fresh-seed capacity-control sensitivity")
    ax.grid(axis="x", alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.savefig(out / "independent_seed_effects.pdf", bbox_inches="tight")
    fig.savefig(out / "independent_seed_effects.svg", bbox_inches="tight")
    plt.close(fig)

    body = []
    for task in TASKS:
        sub = effects[effects.task == task].set_index("contrast")
        cells = []
        for label in labels:
            z = sub.loc[label]
            digits = 3 if label == "Local - interaction" else 2
            cells.append(f"{z.mean_pp:+.{digits}f} [{z.bootstrap_lower_pp:+.{digits}f}, {z.bootstrap_upper_pp:+.{digits}f}]")
        body.append(LABEL[task] + " & " + " & ".join(cells) + r" \\")
    rows_tex = "\n".join(body)
    tex = r'''\documentclass[10pt]{article}
\usepackage[margin=.74in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref}
\definecolor{navy}{HTML}{16324F}\definecolor{teal}{HTML}{087E8B}
\hypersetup{colorlinks=true,linkcolor=navy,urlcolor=teal}
\title{\textcolor{navy}{Fresh-Seed Check of the Capacity Control}\\[3pt]
\large An exploratory replication of cluster-conditional adaptation effects}
\author{23I-2581 and 23I-2604\\Data Analysis and Visualization --- Phase 3 research addendum}
\date{18 November 2026}
\begin{document}\maketitle
\begin{abstract}
We reran the corrected matched-capacity protocol with ten evaluation seeds absent from the original diversity/capacity seed banks. This independent-seed sensitivity reproduces the direction of the main task pattern: cluster-conditional models are modestly worse than global refitting on the linear task and better on XOR and five-region tasks. Local experts and the pooled interaction control remain nearly indistinguishable in mean accuracy. Ten seeds provide lower precision; these estimates are exploratory and are not pooled with the 30-seed primary run.
\end{abstract}
\section{Protocol and independence}
The run uses the same locked generator, historical split, 96-batch stream, batch size, frozen three-cluster mapping, four update clocks, feedback delays 0/5/20, shared released-row windows and 8,000-row cumulative repeated-use cap as the corrected 30-seed experiment. Only the evaluation seed bank changes (73116401--73116410). The seed audit scanned 3,097 text files and 4,248 archive metadata members and found no exact seed collisions. Comparisons remain paired within seed. The seed is the bootstrap unit; rows below average outcomes within each seed over four mechanisms and three delays. This narrow follow-up is a sensitivity check, not an adequately powered confirmatory replication.
\section{Results}
\begin{figure}[ht]
\centering\includegraphics[width=.96\linewidth]{independent_seed_effects.pdf}
\caption{Fresh-seed paired accuracy differences, averaged across mechanisms and delays. Bars show percentile bootstrap intervals over ten seeds. Positive values favor the left-hand model.}
\end{figure}
\begin{table}[ht]
\centering\small\begin{tabular}{@{}lrrr@{}}\toprule
Task & Local $-$ global & Interaction $-$ global & Local $-$ interaction\\\midrule
''' + rows_tex + r'''\bottomrule\end{tabular}
\caption{Mean differences in percentage points with 95\% seed-bootstrap percentile intervals. Four mechanisms and three delays are averaged within each seed before resampling.}
\end{table}
\section{Interpretation and limitations}
The independent seeds reproduce the main qualitative pattern: local and interaction differences from global refitting are negative on linear data and positive on XOR and five-region data. Local-minus-interaction differences are near zero. This supports the capacity/representation interpretation as a pattern worth investigating. The 10-seed intervals are much less stable than the 30-seed analysis, are not multiplicity-adjusted across task contrasts, and should not be treated as independent real-world datasets. Both runs remain synthetic and use one logistic classifier and one fixed cluster count. No claim about causal mechanism, drift-monitoring benefit, natural-domain transfer or analyst benefit follows.
\section{Reproduction}
Reproduce this sensitivity with {\small\url{scripts/reproduce_adaptation_capacity_confirmation.ps1}}. The separate config and run receipt preserve the 10-seed results; do not overwrite the locked 30-seed study.
\end{document}
'''
    texpath = out / "adaptation_capacity_confirmation.tex"
    texpath.write_text(tex, encoding="utf-8")
    subprocess.run(["tectonic", "-X", "compile", str(texpath)], cwd=ROOT, check=True)
    names = ["independent_seed_effects.csv", "independent_seed_effects.pdf", "independent_seed_effects.svg",
             "adaptation_capacity_confirmation.tex", "adaptation_capacity_confirmation.pdf"]
    report_receipt = {"status": "report_compiled", "run_receipt_sha256": sha(run / "audit_receipt.json"),
                      "collision_audit_sha256": sha(ROOT / "docs/phase3/SEED_COLLISION_AUDIT_ADAPTATION_CAPACITY_CONFIRMATION_20261118.json"),
                      "source_script_sha256": sha(Path(__file__)),
                      "reproduction_script_sha256": sha(ROOT / "scripts/reproduce_adaptation_capacity_confirmation.ps1"),
                      "output_sha256": {n: sha(out / n) for n in names}}
    (out / "report_receipt.json").write_text(json.dumps(report_receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Compiled independent-seed report at {out}; seeds={len(receipt['seed_bank'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    build(args.run.resolve(), args.out.resolve())
