"""Build a checksum-gated figure and editable TeX report for NOAA Weather."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.contracts import atomic_write_json, sha256_file  # noqa: E402


def _latex(value: str) -> str:
    return (str(value).replace("\\", r"\textbackslash{}").replace("&", r"\&")
            .replace("%", r"\%").replace("_", r"\_").replace("#", r"\#"))


def build(run_dir: Path, data_path: Path, manifest_path: Path, output: Path) -> None:
    lock = json.loads((run_dir / "protocol_lock.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if lock.get("state") != "complete" or len(lock.get("completed_cells", {})) != 4:
        raise ValueError("NOAA run is incomplete")
    for item in lock["completed_cells"].values():
        path = run_dir / item["artifact"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError("run artifact checksum mismatch")
    if sha256_file(data_path) != summary["data_sha256"] or sha256_file(data_path) != manifest["processed"]["sha256"]:
        raise ValueError("processed NOAA data checksum mismatch")
    metrics = pd.read_csv(run_dir / "evaluator/batch_metrics.csv")
    online = pd.read_csv(run_dir / "online/batch_scores.csv")
    if len(metrics) != 21 or len(online) != 21 or set(metrics.batch) != set(online.batch):
        raise ValueError("unexpected complete-batch metrics")

    output.mkdir(parents=True, exist_ok=True)
    figs = output / "figures"
    figs.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 11, "axes.labelsize": 9,
                         "figure.facecolor": "#F4F7FA", "axes.facecolor": "white",
                         "savefig.facecolor": "#F4F7FA", "axes.edgecolor": "#D6DEE8",
                         "axes.labelcolor": "#344054", "text.color": "#172B4D",
                         "xtick.color": "#475467", "ytick.color": "#475467",
                         "grid.color": "#E7ECF2", "grid.linewidth": .7})
    x = np.arange(1, len(metrics) + 1)
    dates = metrics.timestamp_end.astype(str).to_list()
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(11.4, 6.6), sharex=True,
                                 gridspec_kw={"height_ratios": [2.0, 1.0]}, constrained_layout=True)
    ax.fill_between(x, online.cal_lower.to_numpy(), online.cal_upper.to_numpy(),
                    color="#137C8B", alpha=.12, label="Plug-in interval")
    ax.plot(x, metrics.accuracy, color="#172B4D", marker="o", lw=2.0, ms=4,
            label="Verified accuracy")
    ax.plot(x, online.cal_estimate, color="#137C8B", marker="s", lw=1.6, ms=3.5,
            label="Frozen isotonic estimate")
    ax.plot(x, online.constant_estimate, color="#E76F51", lw=1.4, ls="--",
            label="Historical constant")
    ax.set_ylabel("Accuracy / estimated accuracy")
    ax.set_ylim(0, 1); ax.grid(axis="y"); ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    bx.bar(x, metrics.rain_prevalence, color="#6C63A8", alpha=.85, label="Rain prevalence")
    bx.scatter(x, metrics.input_alarm.astype(int), marker="|", s=90, linewidths=2.1,
               color="#E76F51", label="Input alarm (1 = raised)")
    bx.set_ylabel("Prevalence / alarm"); bx.set_ylim(-.06, 1.06); bx.set_xlabel("Monitoring batch (end date)")
    bx.set_xticks(x[::2], dates[::2], rotation=35, ha="right")
    bx.grid(axis="y"); bx.legend(frameon=False, ncol=2, loc="upper right")
    bx.spines[["top", "right"]].set_visible(False)
    fig.suptitle("NOAA Offutt weather: chronological holdout", x=.01, ha="left",
                 fontsize=15, weight="bold", color="#172B4D")
    fig.savefig(figs / "weather_holdout_timeline.png", dpi=260, bbox_inches="tight")
    fig.savefig(figs / "weather_holdout_timeline.pdf", bbox_inches="tight")
    plt.close(fig)

    estimators = [
        ("Frozen isotonic AC", "calibrated_accuracy_estimate"),
        ("Raw Average Confidence", "raw_average_confidence_estimate"),
        ("ATC-style threshold", "atc_estimate"),
        ("Historical constant", "historical_constant_estimate"),
    ]
    table_rows = []
    for label, col in estimators:
        err = metrics[col].to_numpy(float) - metrics.accuracy.to_numpy(float)
        table_rows.append("{} & {:.3f} & {:+.3f} \\\\".format(_latex(label), np.abs(err).mean(), err.mean()))
    table = "\n".join(table_rows)
    r = summary["row_counts"]
    m = summary["primary_holdout_summary"]
    tex = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=22mm,headheight=14pt]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,tabularx,graphicx,xcolor,hyperref,fancyhdr,titlesec,tcolorbox,caption}
\definecolor{ink}{HTML}{172B4D}\definecolor{teal}{HTML}{137C8B}\definecolor{coral}{HTML}{E76F51}\definecolor{mist}{HTML}{F1F6F8}
\hypersetup{colorlinks=true,linkcolor=teal,urlcolor=teal}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}\setlength{\emergencystretch}{3em}
\pagestyle{fancy}\fancyhf{}\lhead{\textcolor{teal}{IDM-Viz}}\rhead{Phase 3 -- Natural-stream addendum}\cfoot{\thepage}
\titleformat{\section}{\Large\bfseries\color{ink}}{\thesection}{.6em}{}
\captionsetup{font=small,labelfont={bf,color=ink}}
\begin{document}
\begin{titlepage}\color{ink}\vspace*{15mm}
{\large\textcolor{teal}{DATA ANALYSIS AND VISUALIZATION -- PHASE 3}}\\[18mm]
{\Huge\bfseries IDM-Viz}\\[4mm]
{\LARGE NOAA weather-stream transfer addendum}\\[7mm]
{\large A source-verified chronological holdout on Offutt AFB daily records}\\[18mm]
\begin{tcolorbox}[colback=mist,colframe=teal,boxrule=0pt,leftrule=2mm]
\textbf{Evidence boundary}\\
The source is NOAA GSOD, station 72554014949, 1949--1999. The target is same-day rain/drizzle occurrence. This is retrospective classification, not a forward weather forecast.
\end{tcolorbox}
\vfill\textbf{Ahmed Bilal (23I-2581) -- Umar Behram (23I-2604)}\\
FAST-NUCES Islamabad\\[4mm]
Supplementary report -- 17 November 2026\\Editable source: \texttt{REPORT.tex}
\end{titlepage}
\begin{abstract}
This supplement adds a checksum-verified NOAA natural-stream exercise to the controlled IDM-Viz evaluation. A frozen logistic classifier, historical-only scaler/PCA/clusters and isotonic correctness map are fitted on chronological training, calibration and reference segments. The monitoring segment is scored in 21 complete batches of 500 days; 426 remaining rows are disclosed and excluded. Delay-zero labels are replayed only after each batch's scores are persisted. Results are descriptive for one station and one temporal split. They do not identify drift onset, forecast future rainfall, or establish general transfer.
\end{abstract}
\section{Data provenance and admission}
The source is the NOAA/NCEI Global Surface Summary of the Day (GSOD), station \texttt{72554014949}, Bellevue Offutt AFB, Nebraska. The official station-year files cover 1949-01-01 through 1999-12-31. The source manifest records SHA-256 for each of the 51 raw CSV files and for the processed stream. NOAA documents daily summaries and the GSOD fields at \href{https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00516/html}{the GSOD metadata page}.

Eight numeric predictors are TEMP, DEWP, SLP, VISIB, WDSP, MXSPD, MAX and MIN. The target is the rain/drizzle bit in FRSHTT. Precipitation and weather-event flags are excluded from predictors. Values are retained in GSOD's reported units. Rows with the documented 999.9 missing code in any selected predictor are dropped. The output has \textbf{""" + str(manifest["processed"]["row_count"]) + r""" days}, from """ + manifest["processed"]["timestamp_start"] + r""" through """ + manifest["processed"]["timestamp_end"] + r"""; positive-label prevalence is """ + f"{manifest['processed']['positive_rate']:.3f}" + r""".

The public scikit-multiflow benchmark mirror is retained only as a comparison artifact. Its row count and labels differ from this current timestamped GSOD release, and an exact row-by-row equivalence was not established. It is therefore not used to define timestamps or truth. NOAA's daily summaries include same-day aggregates; consequently this protocol makes no ahead-of-time forecast claim.

\section{Chronological protocol}
The eligible rows are split once, without shuffling: 20\% train (""" + str(r["train"]) + r"""), 10\% calibration (""" + str(r["calibration"]) + r"""), 10\% fixed reference (""" + str(r["reference"]) + r""") and 60\% monitoring (""" + str(r["monitoring"]) + r"""). The 500-row monitor batches span about 500 calendar days each. The final partial cohort of """ + str(r["monitoring_remainder_dropped"]) + r""" rows is not analyzed. Preprocessing, classifier, PCA, clustering and calibration are frozen after the historical partitions.

At each monitor clock, features are scored and the online CSV is persisted before the evaluator supplies labels to the broker. Delay zero means same batch-clock availability in this replay; it is not a measured operational reporting delay. No delay-5/20 inference is made for NOAA. Released outcomes are kept in evaluator-only files, separate from online scores.

\section{Results}
Table~\ref{tab:estimators} reports equal-weight descriptive summaries over the 21 chronological monitoring batches. Batches are serially dependent; no inferential confidence interval treats them as independent replications.
\begin{table}[ht]
\centering\small\begin{tabular}{lrr}\toprule
Accuracy estimator & Mean batch MAE & Signed bias \\\midrule
""" + table + r"""\bottomrule\end{tabular}
\caption{Frozen-model accuracy estimators. Bias is estimate minus observed batch accuracy; negative values indicate underestimation.}\label{tab:estimators}
\end{table}

Mean observed accuracy was """ + f"{m['mean_batch_accuracy']:.3f}" + r"""; mean macro-F1 was """ + f"{m['mean_batch_macro_f1']:.3f}" + r""" and mean balanced accuracy was """ + f"{m['mean_batch_balanced_accuracy']:.3f}" + r""". The calibrated plug-in interval covered """ + f"{m['plug_in_interval_coverage']:.3f}" + r""" of batch accuracies, which is weak empirical coverage in this one ordered stream. The historical-constant point estimate slightly outperformed isotonic Average Confidence on mean batch MAE (see Table~\ref{tab:estimators}); calibration is not a universal improvement. The KS/PCA input alarm fired in every monitoring batch, so the result also exposes a high alarm burden under long-run temporal difference from the fixed reference.

\begin{figure}[ht]
\centering\includegraphics[width=\linewidth]{figures/weather_holdout_timeline.pdf}
\caption{Verified accuracy and frozen estimates by chronological batch. The lower panel shows rain prevalence and input alarms. Dashed/filled elements are descriptive; there is no known change-point truth for this real stream.}
\end{figure}

\section{Interpretation and limits}
This run verifies end-to-end processing on a third natural stream and fixes date/source ambiguity for the weather task. It cannot establish broad domain transfer: all batches come from one station and one chronological split, not independent domains. The 500-day cohorts span long intervals and hide within-batch dynamics. The use of same-day summaries means this is a classification benchmark, not operational early warning. The observed alarm fraction is not false-alarm rate because the natural stream has no known onset labels. Calibration and interval estimates remain conditional on a single historical partition; poor coverage here is an empirical warning, not a population guarantee. No causal, future-risk or analyst-benefit conclusion is supported.

\section{Reproduction}
From the project root with the configured Anaconda environment, run the checked PowerShell reproduction script:
\begin{verbatim}
powershell -ExecutionPolicy Bypass -File scripts/reproduce_noaa_weather.ps1
\end{verbatim}
The script refuses to overwrite its versioned output directories, preprocesses only from the preserved NOAA source files, reruns the frozen chronological evaluation, rebuilds the report and compiles the PDF. Each NOAA source file hash is listed in the manifest. Numerical comparisons should use the saved full-precision CSVs.

\section*{References}
\begin{enumerate}
\item NOAA National Centers for Environmental Information. Global Surface Summary of the Day (GSOD), dataset identifier \texttt{gov.noaa.ncdc:C00516}. \url{https://www.ncei.noaa.gov/metadata/geoportal/rest/metadata/item/gov.noaa.ncdc:C00516/html}.
\item V. Losing, B. Hammer and H. Wersing. \emph{driftDatasets}, Weather description and public benchmark stream. \url{https://github.com/vlosing/driftDatasets}.
\item scikit-multiflow contributors. \emph{streaming-datasets/weather.csv}, legacy benchmark mirror compared in the admission audit. \url{https://github.com/scikit-multiflow/streaming-datasets/blob/master/weather.csv}.
\end{enumerate}
\end{document}
"""
    tex_path = output / "REPORT.tex"
    tex_path.write_text(tex, encoding="utf-8")
    receipt = {"run_dir": str(run_dir), "run_lock_sha256": sha256_file(run_dir / "protocol_lock.json"),
               "run_summary_sha256": sha256_file(run_dir / "run_summary.json"),
               "weather_manifest_sha256": sha256_file(manifest_path),
               "processed_weather_sha256": sha256_file(data_path),
               "report_tex_sha256": sha256_file(tex_path),
               "figures": {p.name: sha256_file(p) for p in sorted(figs.iterdir())},
               "report_status": "generated from complete checksum-verified NOAA run"}
    atomic_write_json(output / "report_build_receipt.json", receipt)
    (output / "README.md").write_text(
        "# NOAA Weather natural-stream addendum\n\n"
        "Editable report: `REPORT.tex`; compile with Tectonic to generate `REPORT.pdf`. "
        "Figures and checksum receipt are included. The run is a single-station chronological "
        "same-day classification holdout, not a rain forecast or independent-domain replication.\n",
        encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "results/phase3/noaa_weather_1949_1999_v1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/weather_noaa_1949_1999_v2/weather_ordered.csv")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/processed/weather_noaa_1949_1999_v2/weather_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase3/weather_noaa_1949_1999_v2")
    args = parser.parse_args()
    build(args.run.resolve(), args.data.resolve(), args.manifest.resolve(), args.output.resolve())
    print(f"Built editable TeX and figures in {args.output.resolve()}")


if __name__ == "__main__":
    main()
