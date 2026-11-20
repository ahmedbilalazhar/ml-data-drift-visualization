"""Seed-bootstrap summaries, publication figures and TeX source for localization audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COLORS = {"marginal": "#176B87", "dependence": "#D06A45", "region": "#6A6C9B"}


def bootstrap_mean(values: np.ndarray, seed: int, reps: int = 2000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[draws].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975))


def summarise(group: pd.DataFrame, keys: list[str], metric: str,
              reps: int = 2000) -> pd.DataFrame:
    rows = []
    for i, (key, part) in enumerate(group.groupby(keys, dropna=False, sort=True)):
        key = key if isinstance(key, tuple) else (key,)
        by_seed = part.groupby("seed")[metric].mean()
        mean, lower, upper = bootstrap_mean(by_seed.to_numpy(), 90000 + i, reps)
        row = dict(zip(keys, key, strict=True))
        row.update(mean=mean, lower=lower, upper=upper, seeds=int(by_seed.notna().sum()))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/phase3/localization_audit_20261117_v3")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase3/localization_audit_20261117_v5")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    args.run = args.run.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    raw_region = pd.read_csv(args.run / "region_localization.csv")
    cluster = raw_region[raw_region.batch.isna()].copy()
    batch = raw_region[raw_region.batch.notna()].copy()
    pair = pd.read_csv(args.run / "dependence_pair_localization.csv")

    cluster_summaries = []
    for metric in ("ari", "mapped_coverage", "mapped_accuracy_on_covered", "macro_f1_on_covered",
                   "region2_precision", "region2_recall"):
        frame = summarise(cluster, ["clusters"], metric, args.bootstrap)
        frame.insert(1, "metric", metric)
        cluster_summaries.append(frame)
    cluster_summary = pd.concat(cluster_summaries, ignore_index=True)
    cluster_summary.to_csv(args.output / "cluster_mapping_seed_bootstrap.csv", index=False)

    active_region = batch[batch.target_region_intervention_active.astype("boolean").fillna(False)].copy()
    region_seed = (active_region.groupby(["clusters", "mechanism", "seed"], as_index=False)
                   .agg(precision=("region2_precision", "mean"), recall=("region2_recall", "mean"),
                        coverage=("mapped_assignment_coverage", "mean")))
    region_summary = pd.concat([
        summarise(region_seed, ["clusters", "mechanism"], metric, args.bootstrap).assign(metric=metric)
        for metric in ("precision", "recall", "coverage")], ignore_index=True)
    region_summary.to_csv(args.output / "intervened_region_seed_bootstrap.csv", index=False)

    pair_seed = (pair.groupby(["task", "mechanism", "intervention_active", "seed"], as_index=False)
                 .agg(alarm_rate=("alarm", "mean"), top_pair_accuracy=("target_pair_top1", "mean"),
                      top_pair_accuracy_when_alarm=("target_pair_top1", lambda x: float(x[pair.loc[x.index, "alarm"]].mean())
                                                    if pair.loc[x.index, "alarm"].any() else np.nan)))
    pair_metrics = []
    for metric in ("alarm_rate", "top_pair_accuracy", "top_pair_accuracy_when_alarm"):
        summary = summarise(pair_seed, ["task", "mechanism", "intervention_active"], metric, args.bootstrap)
        summary.insert(3, "metric", metric)
        pair_metrics.append(summary)
    pair_summary = pd.concat(pair_metrics, ignore_index=True)
    pair_summary.to_csv(args.output / "pair_localization_seed_bootstrap.csv", index=False)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(8.6, 4.5), constrained_layout=True)
    task_names = {"linear_probabilistic": "Linear", "xor_nonlinear": "Nonlinear XOR", "five_region": "Five-region"}
    statuses = [("M0_stationary", False, "Stationary"),
                ("M6_dependence_marginals_preserved", False, "Dependence shift, before"),
                ("M6_dependence_marginals_preserved", True, "Dependence shift, active")]
    offsets = [-.18, 0, .18]
    for task_i, task in enumerate(task_names):
        for (mechanism, active, label), offset, color in zip(statuses, offsets,
                                                             ("#78818D", "#A7AFB7", COLORS["dependence"]), strict=True):
            sub = pair_summary[(pair_summary.task == task) & (pair_summary.mechanism == mechanism)
                               & (pair_summary.intervention_active == active)
                               & (pair_summary.metric == "alarm_rate")]
            if len(sub):
                row = sub.iloc[0]
                ax.errorbar(task_i + offset, row["mean"],
                            yerr=[[row["mean"]-row["lower"]], [row["upper"]-row["mean"]]],
                            fmt="o", color=color, capsize=3, markersize=6)
    ax.set_xticks(range(len(task_names)), [task_names[t] for t in task_names])
    ax.set_ylim(-.04, 1.05); ax.set_ylabel("Dependence alarm rate (95% seed bootstrap CI)")
    ax.set_title("Pair detector responds to the designed correlation intervention")
    ax.grid(axis="y", alpha=.2)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([], [], marker="o", color=c, linestyle="", label=l)
                       for (_, _, l), c in zip(statuses, ("#78818D", "#A7AFB7", COLORS["dependence"]), strict=True)],
              frameon=False, loc="upper left")
    fig.savefig(args.output / "dependence_detection.svg", bbox_inches="tight")
    fig.savefig(args.output / "dependence_detection.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1), constrained_layout=True)
    for metric, ax, title in (("ari", axes[0], "Cluster agreement with five regions"),
                              ("recall", axes[1], "Region 2 recall during intervention")):
        d = (cluster_summary if metric == "ari" else region_summary)
        d = d[d.metric == metric]
        for i, row in enumerate(d.sort_values("clusters").itertuples(index=False)):
            ax.errorbar(row.clusters, row.mean, yerr=[[row.mean-row.lower], [row.upper-row.mean]],
                        fmt="o", color=COLORS["region"], capsize=3)
        ax.set_title(title); ax.set_xlabel("Frozen detector clusters")
        ax.set_xticks([3, 5, 10]); ax.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Adjusted Rand index")
    axes[1].set_ylabel("Recall for generating region 2")
    fig.suptitle("Cluster count does not reliably recover the intervened region", weight="bold")
    fig.savefig(args.output / "region_localization.svg", bbox_inches="tight")
    fig.savefig(args.output / "region_localization.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    def fmt_summary(frame, metric, filters):
        sub = frame[frame.metric == metric]
        for key, value in filters.items():
            sub = sub[sub[key] == value]
        return sub.iloc[0]
    pair_active = pair_summary[(pair_summary.task == "linear_probabilistic")
                               & (pair_summary.mechanism == "M6_dependence_marginals_preserved")
                               & (pair_summary.intervention_active == True)]
    pair_alarm = pair_active[pair_active.metric == "alarm_rate"].iloc[0]
    pair_top = pair_active[pair_active.metric == "top_pair_accuracy"].iloc[0]
    ari5 = fmt_summary(cluster_summary, "ari", {"clusters": 5})
    recall5 = region_summary[(region_summary.clusters == 5)
                             & (region_summary.mechanism == "M4_abrupt_conditional_reversal")
                             & (region_summary.metric == "recall")].iloc[0]
    report = r'''\documentclass[10pt]{article}
\usepackage[margin=0.82in]{geometry}
\usepackage{graphicx,booktabs,xcolor,microtype,parskip}
\definecolor{navy}{HTML}{18334B}\definecolor{teal}{HTML}{176B87}
\usepackage[colorlinks=true,linkcolor=teal,urlcolor=teal]{hyperref}
\title{\vspace{-2em}\color{navy}\bfseries IDM-Viz localization audit\\[-.2em]\large 30-seed held-out synthetic evaluation}
\author{Phase 3 empirical addendum}\date{17 November 2026}
\begin{document}\maketitle\vspace{-1em}
\section*{Question and protocol}
We separately tested (i) dependence-pair identification and (ii) mapping frozen clusters to five known generating regions. All cluster-to-region maps were fitted using the historical training rows only. Pair thresholds used historical reference and calibration inputs only. We then applied the frozen mappings and thresholds to 30 evaluation seeds, each with 96 consecutive 500-row batches. Monitoring labels and region IDs were evaluator-only. Confidence intervals resample seeds (2,000 bootstrap replicates); batches are not treated as independent.
\section*{Dependence pair}
During the designed marginal-preserving correlation change, the linear-task detector alarmed on \PAIRALARM\ of active batches (seed-bootstrap 95\% CI: \PAIRLO--\PAIRHI); its top pair matched the intervention pair in \PAIRTARGET\ of batches (CI: \PAIRTOPLO--\PAIRTOPHI). This favorable result applies to one specified intervention law. On the stationary control, the empirically calibrated detector alarmed at approximately its intended 5\% operating rate. It does not establish localization under arbitrary dependence shifts.
\begin{center}\includegraphics[width=.95\linewidth]{dependence_detection.png}\end{center}
\section*{Generating-region assignment}
The five-cluster model achieved seed-mean adjusted Rand index \ARI (CI: \ARILO--\ARIHI). During the abrupt local reversal, its recall for the truly intervened region was \R2 (CI: \R2LO--\R2HI). Other cluster counts have similar weaknesses and may leave regions unmapped. The selected cluster count is not a reliable substitute for region identity; the visualization should expose uncertainty and occupancy rather than label these clusters as known regions.
\begin{center}\includegraphics[width=.95\linewidth]{region_localization.png}\end{center}
\section*{Interpretation}
The audit closes a measurement gap by providing target-specific localization metrics, and it finds that regional localization is weak. The pair result is a targeted synthetic positive control, not broad validation. No causal responsibility, operational benefit, or analyst decision improvement follows from these measurements. Full raw seed and batch rows, mappings, thresholds and checksums are in the accompanying result directory. The current complete project test suite reports 176 passing tests; this report is an exploratory post-hoc diagnostic on the locked evaluation seed bank and was not used to tune thresholds or models.
\end{document}
'''
    replacements = {
        "\\PAIRALARM": f"{pair_alarm['mean']:.3f}", "\\PAIRLO": f"{pair_alarm['lower']:.3f}",
        "\\PAIRHI": f"{pair_alarm['upper']:.3f}", "\\PAIRTARGET": f"{pair_top['mean']:.3f}",
        "\\PAIRTOPLO": f"{pair_top['lower']:.3f}", "\\PAIRTOPHI": f"{pair_top['upper']:.3f}",
        "\\ARI": f"{ari5['mean']:.3f}", "\\ARILO": f"{ari5['lower']:.3f}",
        "\\ARIHI": f"{ari5['upper']:.3f}", "\\R2": f"{recall5['mean']:.3f}",
        "\\R2LO": f"{recall5['lower']:.3f}", "\\R2HI": f"{recall5['upper']:.3f}",
    }
    for old, new in sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True):
        report = report.replace(old, new)
    (args.output / "LOCALIZATION_AUDIT.tex").write_text(report, encoding="utf-8")
    def file_sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = [args.run / "audit_receipt.json", args.run / "region_localization.csv",
              args.run / "historical_cluster_mappings.json",
              args.run / "dependence_pair_localization.csv"]
    outputs = [p for p in sorted(args.output.iterdir()) if p.is_file()]
    receipt = {"status": "complete", "run_dir": args.run.resolve().relative_to(ROOT).as_posix(),
               "input_sha256": {p.relative_to(ROOT).as_posix(): file_sha(p) for p in inputs},
               "summarizer_sha256": file_sha(Path(__file__)),
               "output_sha256": {p.relative_to(ROOT).as_posix(): file_sha(p) for p in outputs},
               "bootstrap_replicates": args.bootstrap,
               "bootstrap_unit": "seed; batch-level observations aggregated within seed first"}
    (args.output / "REPORT_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n",
                                                     encoding="utf-8")
    print(f"Wrote bootstrap summaries, vector/raster figures and TeX source to {args.output}")


if __name__ == "__main__":
    main()
