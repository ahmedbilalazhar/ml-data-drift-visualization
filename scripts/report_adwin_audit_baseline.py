"""Create a paired, label-timing-aware ADWIN baseline report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/phase3/adwin_audit_baseline_20261117_v1"
POLICY_RUN = ROOT / "results/phase3/audit_policy_matrix_20261117_v1"
COLORS = {"exact_quota": "#087E8B", "strict_cap": "#7C3AED"}
MODE_LABELS = {"exact_quota": "Exact quota", "strict_cap": "Strict cap"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out: Path) -> None:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    adwin = pd.read_csv(RUN / "adwin_per_seed.csv")
    policy = pd.read_csv(POLICY_RUN / "policy_per_seed.csv")
    lock = json.loads((RUN / "protocol_lock.json").read_text(encoding="utf-8"))
    keys = ["seed", "condition", "delay", "budget_cohorts", "cost_mode"]
    adwin_idx = adwin.set_index(keys)
    comparisons = []
    rng = np.random.default_rng(20261118)
    for baseline_name in ("uniform", "released_error"):
        base = policy[policy.policy == baseline_name].set_index(keys)
        join = adwin_idx.join(base[["verified_harmful_episodes_by_stream_end", "audited_batches"]],
                              how="left", rsuffix="_baseline")
        join["episode_delta"] = (join.verified_harmful_episodes_by_stream_end -
                                 join.verified_harmful_episodes_by_stream_end_baseline)
        join["audit_delta"] = join.audited_batches - join.audited_batches_baseline
        collapsed = join.reset_index().groupby(
            ["seed", "cost_mode", "delay"], as_index=False
        )[["episode_delta", "audit_delta"]].mean()
        for (mode, delay), group in collapsed.groupby(["cost_mode", "delay"], sort=True):
            for metric in ("episode_delta", "audit_delta"):
                values = group.sort_values("seed")[metric].to_numpy(float)
                draws = values[rng.integers(0, len(values), size=(2_000, len(values)))].mean(axis=1)
                comparisons.append({
                    "contrast": f"river_adwin_minus_{baseline_name}", "cost_mode": mode,
                    "delay": int(delay), "metric": metric,
                    "mean_difference": float(values.mean()),
                    "lower": float(np.quantile(draws, .025)), "upper": float(np.quantile(draws, .975)),
                    "paired_seeds": len(values), "bootstrap_replicates": 2_000,
                    "bootstrap_unit": "seed after averaging seven conditions and three budgets",
                })
    paired = pd.DataFrame(comparisons)
    paired.to_csv(out / "adwin_paired_aggregate_effects.csv", index=False)

    per_seed = adwin.copy()
    per_seed["detected"] = per_seed.adwin_detection_events.gt(0)
    detection = per_seed.groupby(["delay", "budget_cohorts", "cost_mode"], as_index=False).agg(
        detection_fraction=("detected", "mean"), detection_events=("adwin_detection_events", "mean"),
        labels_processed=("adwin_row_updates", "mean"), audits=("audited_batches", "mean"),
        verified_episodes=("verified_harmful_episodes_by_stream_end", "mean"))
    detection.to_csv(out / "adwin_detection_summary.csv", index=False)
    aggregate = per_seed.groupby(["delay", "cost_mode"], as_index=False).agg(
        detection_cells=("detected", "sum"), policy_cells=("detected", "size"),
        detection_fraction=("detected", "mean"), audits=("audited_batches", "mean"),
        labels_processed=("adwin_row_updates", "mean"), verified_episodes=("verified_harmful_episodes_by_stream_end", "mean"))
    aggregate.to_csv(out / "adwin_delay_summary.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.9, 4.7), constrained_layout=True)
    forest = paired[(paired.contrast == "river_adwin_minus_released_error") &
                    (paired.metric == "episode_delta")]
    offsets = {"strict_cap": -0.08, "exact_quota": 0.08}
    for mode in ("strict_cap", "exact_quota"):
        group = forest[forest.cost_mode == mode].sort_values("delay")
        y = np.arange(len(group)) + offsets[mode]
        axes[0].errorbar(group.mean_difference, y,
                         xerr=[group.mean_difference - group.lower, group.upper - group.mean_difference],
                         fmt="o", ms=5, capsize=3, lw=1.5, color=COLORS[mode], label=MODE_LABELS[mode])
    axes[0].axvline(0, color="#334155", lw=1, ls="--")
    axes[0].set_yticks(np.arange(4), ["0 batches", "5 batches", "20 batches", "100 batches"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Verified harmful episodes vs released-error policy\n(95% paired seed-bootstrap interval)")
    axes[0].set_title("ADWIN under matched quotas", weight="bold")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="x", alpha=.22)
    axes[0].spines[["top", "right"]].set_visible(False)

    heat = detection.pivot(index=["cost_mode", "budget_cohorts"], columns="delay",
                           values="detection_fraction")
    row_names = [f"{MODE_LABELS[m]} · B={b}" for m, b in heat.index]
    values = heat.to_numpy()
    image = axes[1].imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    axes[1].set_yticks(np.arange(len(row_names)), row_names, fontsize=7.5)
    axes[1].set_xticks(np.arange(4), ["0", "5", "20", "100"])
    axes[1].set_xlabel("Label delay (batches)")
    axes[1].set_title("Fraction of seed-condition cells\nwith at least one ADWIN detection", weight="bold")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            axes[1].text(j, i, f"{100*v:.0f}%", ha="center", va="center",
                         color="white" if v > .50 else "#16324f", fontsize=8)
    fig.colorbar(image, ax=axes[1], shrink=.84, label="Detection fraction")
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out / f"adwin_effects.{ext}", dpi=220 if ext == "png" else None)
    plt.close(fig)

    rows = []
    for row in aggregate.sort_values(["cost_mode", "delay"]).itertuples():
        rows.append(f"{MODE_LABELS[row.cost_mode]} & {row.delay} & {row.audits:.2f} & "
                    f"{row.labels_processed:.0f} & {int(row.detection_cells)}/{int(row.policy_cells)} & "
                    f"{row.verified_episodes:.3f}" + r" \\")
    table = "\n".join(rows)
    tex = r'''\documentclass[10pt]{article}
\usepackage[margin=0.78in]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref,placeins}
\definecolor{navy}{HTML}{16324F}\definecolor{teal}{HTML}{087E8B}
\hypersetup{colorlinks=true,linkcolor=navy,urlcolor=teal}
\title{\textcolor{navy}{A Stronger Drift Baseline}\\[3pt]
\large ADWIN with Delayed, Budgeted Prediction-Error Feedback}
\author{23I-2581 and 23I-2604\\Data Analysis and Visualization --- Phase 3 addendum}
\date{17 November 2026}
\begin{document}\maketitle
\begin{abstract}
We add River's canonical ADWIN detector as a supervised baseline on the exact same frozen predictions, 30 evaluation seeds, seven synthetic conditions, audit cohorts, budgets and label delays used in the policy matrix. ADWIN receives row-level binary prediction errors only from sampled cohorts after their labels are released. The comparison demonstrates that detector choice is constrained by the feedback schedule: under strict caps, a single bootstrap cohort supplies no second labeled sample from which to detect a change. This is a scoped baseline evaluation, not a claim of ADWIN's general performance.
\end{abstract}
\section{Method and timing}
We use River 0.27.0's implementation of ADWIN/ADWIN2. Its significance value is $\delta=0.002$, the documented default. We set \texttt{clock=1} so checks occur after each received error; the documented defaults for maximum buckets (5), minimum subwindow length (5) and grace period (10) are retained. River describes ADWIN as comparing the means of two subwindows in an adaptive window; see the [\href{https://riverml.xyz/latest/api/drift/ADWIN/}{River ADWIN API}]. The underlying method is due to Bifet and Gavald\`a~\cite{bifet2007}.

The detector receives a zero/one error sequence from the 500 rows in each selected audit cohort, in row order. It receives nothing from unselected batches. A selected source batch $s$ is first ingested at decision clock $t$ only when $s+d<t$. A fixed initial probe at $\lceil24/B\rceil$ is shared with the released-error policy. A new audit is requested when ADWIN signals and the strict budget has room; exact-quota mode fills unused quota at the end and marks those forced acquisitions. Each clock predicts before its current batch's labels can be used.

ADWIN is a supervised error-change detector; it does not detect feature-only drift without labels. Uncertainty resamples the 30 seeds after averaging the seven matched conditions and three budget levels. Neither conditions nor batches are treated as independent replicates.
\section{Results}
\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{adwin_effects.pdf}
\caption{Left: paired ADWIN-minus-released-error differences in harmful episodes verified by stream end; seed-bootstrap intervals average over matched conditions and budgets. Right: proportion of seed-condition cells with at least one ADWIN event. A detector event is not itself evidence that harmful performance was correctly identified.}
\end{figure}
\begin{table}[ht]
\centering\small\begin{tabular}{@{}llrrrr@{}}\toprule
Cost mode & Delay & Audits & Error rows ingested & Detection cells & Verified episodes\\\midrule
''' + table + r'''\bottomrule\end{tabular}
\caption{Descriptive averages across budgets, seeds and conditions. The event count is out of 630 seed-condition-budget cells for each delay and cost mode.}
\end{table}
\FloatBarrier
\section{Interpretation and limitations}
ADWIN produced at least one event in 212 of 5,040 policy cells; all such cells were exact-quota runs. No strict-cap cell fired after its initial probe. This is a feedback deadlock in the declared strict-cap rule: after one sampled cohort ADWIN has no second selected cohort to compare, so it cannot trigger the next audit. Exact-quota tail acquisitions supply further data, but those acquisitions are mandated by budget completion and arrive late; their event counts are not an online detection success rate. At delays 20 and 100, the finite 24-batch horizon leaves little or no released feedback.

The exact-quota paired contrast against released-error is small in the reported aggregate; it does not show ADWIN is generally better or worse. The strict-cap comparison is also constrained by fewer audits. This implementation demonstrates that a named baseline can be faithfully placed inside the same delayed feedback contract and exposes how a bootstrap schedule affects detector viability. A future operational ADWIN policy needs a predeclared exploration schedule that yields at least two informative audited cohorts without exceeding the same budget; this would be a new policy and requires a separately locked evaluation.

All experiments reuse the cached synthetic predictions and truth from the locked evaluation. The detector is not tested on new seeds or natural streams. It does not estimate causal feature importance, calibration quality, analyst benefit, or monetary acquisition cost. River parameters are fixed and were not tuned on final seeds.
\section{Reproduction}
From the project root, install the declared dependencies, then run:
\begin{verbatim}
python scripts/run_adwin_audit_baseline.py
python scripts/report_adwin_audit_baseline.py
\end{verbatim}
The protocol lock records the River version, parameters, source hashes and output hashes. The editable report command compiles this TeX with Tectonic and hashes the resulting PDF.
\begin{thebibliography}{1}
\bibitem{bifet2007} A. Bifet and R. Gavald\`a, ``Learning from time-changing data with adaptive windowing,'' in \emph{Proc. 7th SIAM Int. Conf. on Data Mining}, 2007, pp. 443--448, doi: \href{https://doi.org/10.1137/1.9781611972771.42}{10.1137/1.9781611972771.42}.
\end{thebibliography}
\end{document}
'''
    tex_path = out / "adwin_baseline.tex"
    tex_path.write_text(tex, encoding="utf-8")
    subprocess.run(["tectonic", "-X", "compile", str(tex_path)], cwd=ROOT, check=True)
    output_names = ["adwin_paired_aggregate_effects.csv", "adwin_detection_summary.csv",
                    "adwin_delay_summary.csv", "adwin_effects.png", "adwin_effects.pdf",
                    "adwin_effects.svg", "adwin_baseline.tex", "adwin_baseline.pdf"]
    receipt = {"status": "report_sources_complete", "protocol_lock_sha256": sha(RUN / "protocol_lock.json"),
               "run_protocol": lock["protocol"], "outputs_sha256": {n: sha(out / n) for n in output_names}}
    (out / "report_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote and compiled ADWIN report in {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "reports/phase3/adwin_audit_baseline_20261117_v3")
    args = parser.parse_args()
    build(args.out.resolve())
