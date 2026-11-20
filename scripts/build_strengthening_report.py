"""Build the editable TeX research supplement from a completed v3 run."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.contracts import sha256_file  # noqa: E402


def tex(text):
    value = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
                 ("^", r"\textasciicircum{}")):
        value = value.replace(a, b)
    return value


def bootstrap(values, seed=73116601, replicates=2000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    if len(values) == 1:
        return float(values.mean()), np.nan, np.nan
    rng = np.random.default_rng(seed)
    index = rng.integers(0, len(values), size=(replicates, len(values)))
    draw = values[index].mean(axis=1)
    return float(values.mean()), float(np.quantile(draw, .025)), float(np.quantile(draw, .975))


def _estimator_table(df):
    d = df[(df.feature_dimension == 8) &
           df.phase.isin(["early_impact_25_40", "recurrent_65_80"])]
    seed_values = d.groupby(["seed", "estimator"], as_index=False).agg(
        mae=("mae", "mean"), bias=("bias", "mean"), coverage=("interval_coverage", "mean"),
        brier=("brier_correctness", "mean"))
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Estimator & MAE [95\% seed CI] & Bias & Cal. interval coverage \\ \midrule"]
    for i, method in enumerate(("cal", "raw", "atc", "constant")):
        g = seed_values[seed_values.estimator == method]
        mean, lo, hi = bootstrap(g.mae, seed=73116610 + i)
        lines.append(f"{method.upper()} & {mean:.4f} [{lo:.4f}, {hi:.4f}] & "
                     f"{g.bias.mean():+.4f} & {g.coverage.mean():.3f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines), seed_values


def _paired_summary_table(df):
    primary = df[(df.feature_dimension == 8) & df.phase.isin(
        ["stationary_1_24", "early_impact_25_40", "post_ramp_41_64", "recurrent_65_80", "late_81_96"])]
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Comparator & Mean paired $\Delta$MAE & Holm $q<.05$ strata & Strata \\ \midrule"]
    for comp in ("raw", "atc", "constant"):
        g = primary[primary.contrast == f"cal_minus_{comp}_mae"]
        significant = int((g.p_holm < .05).sum())
        lines.append(f"CAL $-$ {comp.upper()} & {g.mean_paired_difference.mean():+.4f} & "
                     f"{significant} & {len(g)} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _detector_table(df):
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Regime & Detector & False alarm rate & Detection rate & Mean lag (batches) \\ \midrule"]
    for mechanism in ("M0_stationary", "M1_abrupt_nuisance", "M2_gradual_predictive",
                      "M3_incremental_repeated_small", "M4_abrupt_conditional_reversal",
                      "M5_recurrent_conditional_reversal", "M6_dependence_marginals_preserved",
                      "M7_out_of_support"):
        for det in ("original_plus_pca_ks", "pairwise_correlation"):
            g = df[(df.mechanism == mechanism) & (df.feature_dimension == 8) & (df.detector == det)]
            if not len(g):
                continue
            fpr = g.stationary_false_alarm_rate.mean()
            detection = g.post_intervention_detection_rate.mean()
            lag = g.mean_detection_delay_batches.mean()
            label = "KS/PCA" if det == "original_plus_pca_ks" else "Pairwise correlation"
            regime = mechanism.replace("_", " ")
            lines.append(f"{tex(regime)} & {label} & {fpr:.3f} & {detection:.3f} & {lag:.2f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _delay_table(df):
    g = df[(df.feature_dimension == 8) & (df.phase == "early_impact_25_40")].groupby("delay").agg(
        age=("mean_evidence_age_batches", "mean"), accuracy=("mean_accuracy_at_release", "mean"),
        harm=("verified_harm_rate", "mean"), batches=("verified_batches", "sum"), seeds=("seed_count", "sum"))
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Delay & Age at release & Accuracy after release & Verified harm fraction & Verified batches \\ \midrule"]
    for delay, row in g.sort_index().iterrows():
        lines.append(f"{int(delay)} & {row.age:.0f} & {row.accuracy:.3f} & {row.harm:.3f} & {int(row.batches):,} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _released_error_table(df):
    d = df[df.feature_dimension == 8]
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"Delay & Stationary false alarm & Drift detected by input end & Drift detected after drain & Mean lag (source batches) \\ \midrule"]
    for delay, all_rows in d.groupby("delay", sort=True):
        stationary = all_rows[all_rows.mechanism == "M0_stationary"]
        shifted = all_rows[all_rows.mechanism != "M0_stationary"]
        fpr = stationary.stationary_false_alarm.mean()
        during = shifted.post_change_detected_during_input.mean()
        drained = shifted.post_change_detected_after_drain.mean()
        lag = shifted.detection_lag_source_batches.mean()
        lines.append(f"{int(delay)} & {fpr:.3f} & {during:.3f} & {drained:.3f} & {lag:.2f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def _dimension_sensitivity(analysis_dir, output_dir, primary_analysis_dir):
    """Build an optional appendix from a separate, complete sensitivity run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    manifest = json.loads((analysis_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
    run_dir = Path(manifest["run_dir"]).resolve()
    lock = json.loads((run_dir / "protocol_lock.json").read_text(encoding="utf-8"))
    if lock.get("state") != "complete":
        raise SystemExit("dimension sensitivity run is incomplete")
    for key, item in lock["completed_cells"].items():
        if sha256_file(run_dir / item["artifact"]) != item["sha256"]:
            raise SystemExit(f"sensitivity cell checksum mismatch: {key}")
    d = pd.read_csv(analysis_dir / "estimation_per_seed.csv")
    primary = pd.read_csv(primary_analysis_dir / "estimation_per_seed.csv")
    d = pd.concat([primary, d], ignore_index=True)
    d = d[d.phase.isin(["early_impact_25_40", "recurrent_65_80"])]
    seed = d.groupby(["feature_dimension", "seed", "estimator"], as_index=False).mae.mean()
    dim_means = seed.groupby(["feature_dimension", "estimator"]).mae.mean()
    dimensions = sorted(seed.feature_dimension.unique())
    sensitivity_cell_count = len(lock["expected_cells"])
    cal_values = ", ".join(f"d={int(dim)}: {dim_means.loc[(dim, 'cal')]:.4f}" for dim in dimensions)
    raw_values = ", ".join(f"d={int(dim)}: {dim_means.loc[(dim, 'raw')]:.4f}" for dim in dimensions)
    dimension_interpretation = (f"Descriptively, calibrated-estimator MAE was {cal_values}; raw Average Confidence was "
        f"{raw_values}. The synthetic feature space and signal-to-noise ratio change with dimension, so this is a "
        "sensitivity of the declared generator family, not an isolated causal test of dimensionality or evidence of a universal trend.")
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Features & Estimator & Mean seed MAE & 95\% seed-bootstrap interval \\ \midrule"]
    palette = {"cal": "#137C8B", "raw": "#E76F51", "atc": "#6C63A8", "constant": "#8993A4"}
    fig, ax = plt.subplots(figsize=(8.8, 4.8), constrained_layout=True)
    dims = sorted(seed.feature_dimension.unique())
    for est in ("cal", "raw", "atc", "constant"):
        means, lows, highs = [], [], []
        for dim in dims:
            vals = seed[(seed.feature_dimension == dim) & (seed.estimator == est)].mae.to_numpy()
            mean, lo, hi = bootstrap(vals, seed=73116700 + int(dim) + len(est))
            lines.append(f"{int(dim)} & {est.upper()} & {mean:.4f} & [{lo:.4f}, {hi:.4f}] " + r"\\")
            means.append(mean); lows.append(mean - lo); highs.append(hi - mean)
        ax.errorbar(dims, means, yerr=[lows, highs], marker="o", lw=2, capsize=4,
                    color=palette[est], label=est.upper())
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(dims, [str(int(v)) for v in dims])
    ax.set_xlabel("Number of input features"); ax.set_ylabel("Mean absolute error")
    ax.grid(axis="y", alpha=.25); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=4)
    ax.set_title("Estimator sensitivity to feature dimension", loc="left", weight="bold")
    fig.savefig(output_dir / "figures/fig05_dimension_sensitivity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)
    table = "\n".join(lines)
    return (r"\newcommand{\DimensionSensitivity}{" +
            r"\subsection{Feature-dimensionality sensitivity}" +
            f"This separate locked extension contains {sensitivity_cell_count:,} task--mechanism--seed--dimension cells and repeats the estimator comparison at 32 and 128 features; the primary eight-feature result is included for reference. Seed-bootstrap intervals resample complete seed blocks and describe Monte Carlo variability conditional on these synthetic laws.\\par{{}}" +
            dimension_interpretation + r"\par{}" +
            r"\begin{table}[htbp]\centering\small" + table +
            r"\caption{Accuracy-estimation error by feature dimension in the sensitivity run.}\end{table}" +
            r"\begin{figure}[htbp]\centering\includegraphics[width=.88\linewidth]{figures/fig05_dimension_sensitivity.png}" +
            r"\caption{Seed-level mean absolute estimation error by feature dimension. Error bars are 95\% seed-bootstrap intervals.}\end{figure}}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--sensitivity-analysis", type=Path,
                        help="analysis directory from a completed 32/128-feature sensitivity run")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase3/strengthening_v3_evaluation")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    analysis = (args.analysis or run_dir / "analysis").resolve()
    output = args.output.resolve()
    lock = json.loads((run_dir / "protocol_lock.json").read_text(encoding="utf-8"))
    if lock.get("state") != "complete":
        raise SystemExit("refusing to write a research report from an incomplete run")
    for key, item in lock["completed_cells"].items():
        if sha256_file(run_dir / item["artifact"]) != item["sha256"]:
            raise SystemExit(f"cell checksum mismatch: {key}")
    required = ["estimation_per_seed.csv", "estimation_summary.csv", "paired_estimator_tests.csv", "detection_summary.csv",
                "delayed_verification_summary.csv", "released_error_detection_per_seed.csv",
                "analysis_manifest.json"]
    if any(not (analysis / path).is_file() for path in required):
        raise SystemExit("run analysis is incomplete; run scripts/analyze_strengthening_v3.py first")
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    for name in ("fig01_estimator_mae.png", "fig02_detection_and_accuracy_timeline.png",
                 "fig03_delay_verification.png", "fig04_feature_localization.png"):
        shutil.copy2(analysis / name, output / "figures" / name)

    estimation = pd.read_csv(analysis / "estimation_per_seed.csv")
    detection = pd.read_csv(analysis / "detection_summary.csv")
    delays = pd.read_csv(analysis / "delayed_verification_summary.csv")
    released_errors = pd.read_csv(analysis / "released_error_detection_per_seed.csv")
    estimator_tex, estimator_seeds = _estimator_table(estimation)
    paired_tests = pd.read_csv(analysis / "paired_estimator_tests.csv")
    paired_tex = _paired_summary_table(paired_tests)
    paired_means = paired_tests[paired_tests.feature_dimension == 8].groupby("contrast").mean_paired_difference.mean()
    paired_readout = ("Across the primary strata, mean paired calibrated-minus-raw MAE was "
                      f"{paired_means['cal_minus_raw_mae']:+.4f}; against ATC-style it was "
                      f"{paired_means['cal_minus_atc_mae']:+.4f}; against the historical constant it was "
                      f"{paired_means['cal_minus_constant_mae']:+.4f}. This means the isotonic estimator "
                      "did not uniformly dominate the baselines, so the evidence supports a comparative "
                      "trade-off rather than a universal calibration advantage.")
    detector_tex = _detector_table(detection)
    delay_tex = _delay_table(delays)
    released_error_tex = _released_error_table(released_errors)
    dimension_tex = ( _dimension_sensitivity(args.sensitivity_analysis.resolve(), output, analysis)
                     if args.sensitivity_analysis else r"\newcommand{\DimensionSensitivity}{}")
    (output / "tables.tex").write_text(
        "\\newcommand{\\EstimatorResults}{\n" + estimator_tex + "\n}\n" +
        "\\newcommand{\\DetectorResults}{\n" + detector_tex + "\n}\n" +
        "\\newcommand{\\DelayResults}{\n" + delay_tex + "\n}\n" +
        "\\newcommand{\\ReleasedErrorResults}{\n" + released_error_tex + "\n}\n" +
        "\\newcommand{\\PairedResults}{\n" + paired_tex + "\n}\n" +
        dimension_tex + "\n", encoding="utf-8")
    d8 = estimation[estimation.feature_dimension == 8]
    paired = d8[d8.phase.isin(["early_impact_25_40", "recurrent_65_80"])].groupby(
        ["seed", "estimator"], as_index=False).mae.mean().pivot(index="seed", columns="estimator", values="mae")
    headline = []
    for name in ("cal", "raw", "atc", "constant"):
        avg = estimator_seeds.loc[estimator_seeds.estimator == name, "mae"].mean()
        headline.append(f"{name.upper()} {avg:.4f}")
    eval_seed_count = int(estimator_seeds.seed.nunique())
    cal_brier = estimator_seeds.loc[estimator_seeds.estimator == "cal", "brier"]
    brier_mean, brier_low, brier_high = bootstrap(cal_brier, seed=73116620)
    matrix = len(lock["expected_cells"])
    report_date = datetime.now(ZoneInfo("Asia/Karachi")).strftime("%d %B %Y")
    p2_hash = lock["source_sha256"].get("docs/23i-2604_23i-2581DAV-phase2.docx", "not-recorded")
    p2_hash_tex = r"\texttt{" + r"\allowbreak{}".join(p2_hash[i:i+8] for i in range(0, len(p2_hash), 8)) + "}"
    (output / "REPORT.tex").write_text(r'''\documentclass[11pt,a4paper]{article}
\usepackage[margin=23mm,headheight=15pt]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,booktabs,tabularx,longtable,graphicx,xcolor,hyperref,fancyhdr,enumitem,caption,titlesec,tcolorbox}
\tcbuselibrary{breakable,skins}
\definecolor{ink}{HTML}{172B4D}
\definecolor{teal}{HTML}{137C8B}
\definecolor{coral}{HTML}{E76F51}
\definecolor{mist}{HTML}{F1F6F8}
\hypersetup{colorlinks=true,linkcolor=teal,urlcolor=teal,citecolor=teal}
\setlength{\parindent}{0pt}\setlength{\parskip}{7pt}\setlength{\emergencystretch}{3em}
\pagestyle{fancy}\fancyhf{}\lhead{\textcolor{teal}{IDM-Viz}}\rhead{Phase 3 -- Research supplement}\cfoot{\thepage}
\titleformat{\section}{\Large\bfseries\color{ink}}{\thesection}{.6em}{}
\titleformat{\subsection}{\large\bfseries\color{teal}}{\thesubsection}{.6em}{}
\captionsetup{font=small,labelfont={bf,color=ink}}
\input{tables.tex}
\begin{document}
\begin{titlepage}\color{ink}
\vspace*{12mm}{\large\textcolor{teal}{DATA ANALYSIS AND VISUALIZATION -- PHASE 3}}\\[18mm]
{\Huge\bfseries IDM-Viz}\\[4mm]
{\LARGE Impact-aware drift monitoring under delayed feedback}\\[8mm]
{\Large A controlled robustness and evidence-quality evaluation}\\[20mm]
\begin{tcolorbox}[colback=mist,colframe=teal,boxrule=0pt,leftrule=2.2mm,arc=1.5mm]
\textbf{Study design}\\
3 synthetic task families; 8 paired mechanisms; 30 held-out seeds; 500-row batches; 96 input batches; four label-delay schedules.
\end{tcolorbox}
\vfill
\textbf{Ahmed Bilal (23I-2581) -- Umar Behram (23I-2604)}\\
FAST-NUCES Islamabad\\[4mm]
Supplementary report -- '''+report_date+r'''\\
Editable source: \texttt{REPORT.tex}
\end{titlepage}
\tableofcontents\clearpage
\begin{abstract}
This supplementary evaluation strengthens the Phase 3 implementation with paired, controlled drift mechanisms and an explicit input-before-label release runner. It compares a frozen isotonic Average-Confidence estimator with raw confidence, an ATC-style estimator and historical accuracy; measures original-feature/PCA and pairwise-correlation input evidence; and verifies cached predictions after delays of 0, 5, 20 and 100 batches. The locked held-out matrix contains \textbf{'''+str(matrix)+r''' complete task--mechanism--seed cells}; the primary comparison uses eight features and \textbf{'''+str(eval_seed_count)+r''' independent seed blocks}. Results are conditional on the stated synthetic laws. They do not establish general transfer, causal attribution, certified harm probability, future-risk performance, or improved analyst decisions.
\end{abstract}

\begin{tcolorbox}[breakable,colback=mist,colframe=ink,boxrule=.4pt,title=Interpretation at a glance]
All reported numbers are generated from checksummed cell artifacts. Point estimates and 95\% intervals resample complete seeds, never batches or delay copies. The estimator aggregates over the two predeclared post-intervention phases. The exact evidence, locked source snapshot and run state are available in the accompanying result directory. The coordinated view is described as a prototype; no analyst-benefit claim is made.
\end{tcolorbox}

\section{Phase 2 basis and changes}
The team's submitted Phase 2 DOCX remains the authoritative methodological basis and is byte-for-byte preserved (SHA-256: '''+p2_hash_tex+r'''). No instructor feedback was received. This supplement adds controlled tests to the integrated IDM-Viz objective; it does not replace the submitted research question or earlier reports.

The implementation uses a frozen logistic classifier; historical-only scaler, PCA, clustering and correctness calibration; 4,000 training, 2,000 calibration and 2,000 fixed-reference observations; and paired generators. PCA replaces the proposed autoencoder. Original-feature and correlation statistics are descriptive, not causal. Weather was not admitted because its primary source, ordering, label construction and checksum remain unverified. Causal discovery, broad adaptation and deployment validation are outside this run. The coordinated view is reported as a prototype, with no claim of improved analyst performance.

\section{Experimental setup}
The primary held-out design crosses the linear probabilistic, XOR nonlinear, and five-region generating families with: stationary control; abrupt nuisance-feature shift; gradual predictive-feature shift; repeated small shifts; abrupt conditional reversal; recurrent conditional reversal; dependence change with preserved marginal scales; and a large covariate displacement. Each stream has 96 batches of 500 rows. Batches 1--24 are a shared stationary prefix across mechanisms within a task/seed. Interventions begin at batch 25; the gradual change reaches its specified endpoint at batch 40; the recurrent conditional reversal returns during batches 65--80. The five-region ID and the true label probability are evaluator-only. The locked M7 identifier says ``out of support'', but the Gaussian source has unbounded support: M7 is interpreted only as a finite-sample low-density extrapolation stress, not a literal support violation.

Seeds are paired across mechanisms and disjoint from development/validation banks. A frozen logistic model is fitted once on historical training rows; the isotonic correctness map uses only its separate calibration rows. KS/PCA and correlation detectors receive only current features. The correlation threshold is calibrated from pseudo-cohorts in the historical calibration partition against the fixed reference features. The probability interval is an exact plug-in Poisson-binomial distribution that treats correctness probabilities as fixed. It is not a confidence interval accounting for calibration or reference uncertainty. A small-window, released-error ADWIN-style detector is also scored when labels arrive; it uses a transparent exact split scan and is not the compressed River ADWIN implementation.

At each input clock, scores and predictions are computed and persisted before any labels scheduled for that clock are released. The release replay validates prediction/label row identifiers and drains the queue after the final input batch without fabricating new input. Delay changes when verification is available; this frozen-model comparison does not allow delayed labels to change earlier predictions.

\section{Accuracy-estimation results}
Table~\ref{tab:estimators} averages each estimator's batch-level absolute error and bias over batches 25--40 and 65--80, then averages matched conditions within each seed. The interval is a 2,000-replicate percentile bootstrap over seed blocks. Coverage is the empirical fraction of batch accuracies inside the plug-in interval; width is descriptive.
\begin{table}[htbp]\centering\small\EstimatorResults\caption{Held-out accuracy estimation, eight-feature primary setting.}\label{tab:estimators}\end{table}

The comparison is internal and paired. Raw Average Confidence and ATC-style thresholding are within-study comparators, not fresh replications of published benchmarks. The calibrated correctness-probability Brier score is '''+f"{brier_mean:.4f} [{brier_low:.4f}, {brier_high:.4f}]"+r''' (95\% seed-bootstrap interval). The calibration map is learned under a historical source law and can fail when the conditional relationship between confidence and correctness changes. High confidence must not be presented as verified accuracy. The interval-coverage column evaluates the calibrated plug-in interval and is therefore repeated for point-estimator comparators.

Table~\ref{tab:paired} summarizes paired seed-level sign-flip tests. The family-wide Holm correction covers the complete predeclared primary set of task, mechanism, phase, and comparator contrasts. Negative mean differences favor the calibrated estimator; significance counts are descriptive across strata, not independent replications. '''+paired_readout+r'''
\begin{table}[htbp]\centering\small\PairedResults\caption{Paired primary-setting estimator contrasts; $\Delta$MAE is calibrated minus comparator.}\label{tab:paired}\end{table}

\DimensionSensitivity
\section{Input detection and descriptive localization}
Table~\ref{tab:detection} reports seed-averaged false-alarm rates over the stationary prefix and any-alarm detection after the programmed intervention. Detection delay is measured from batch 25, so it is not a valid onset-latency summary for mechanisms whose effect is gradual, recurrent, or conditionally localized. The correlation detector is a project-specific maximum absolute pairwise correlation-change statistic, not ADWIN or a reproduction of a named detector.
\begin{table}[htbp]\centering\scriptsize\resizebox{\linewidth}{!}{\DetectorResults}\caption{Input-evidence trade-offs by programmed mechanism.}\label{tab:detection}\end{table}

\begin{figure}[p]\centering\includegraphics[width=\linewidth,height=.86\textheight,keepaspectratio]{figures/fig02_detection_and_accuracy_timeline.png}\caption{Batch-level accuracy, input alarms and plug-in interval across all mechanisms. The vertical marker indicates the end of the shared stationary prefix. Alarm rates and outcome truth are aggregated across tasks and seeds.}\end{figure}

Feature ranks are descriptive. Precision@1 and recall@3 are defined only where the generator specifies changed marginal features. They are undefined for pure dependence and conditional-only changes. The five-region task keeps region truth out of the online scorer; no cluster-to-region accuracy is claimed.
\begin{figure}[htbp]\centering\includegraphics[width=.96\linewidth]{figures/fig04_feature_localization.png}\caption{Changed-feature retrieval in mechanisms with known marginal targets. This is not causal attribution.}\end{figure}

\section{Delayed-label verification}
Table~\ref{tab:delay} shows released evidence age and verified outcomes for the early post-intervention phase. All 96 cohorts in every cell are eventually verified after queue drainage. Delay 100 therefore means no labels arrive during the 96-batch input period; results after drainage are retrospective verification, not online evidence.
\begin{table}[htbp]\centering\small\DelayResults\caption{Verification at the scheduled release clock.}\label{tab:delay}\end{table}
The released-error comparator sees a verified cohort only after its scheduled availability time. For each release it scans eligible historical window cuts with a Hoeffding difference bound, trims the older segment after an alarm, and retains the newer segment. Table~\ref{tab:released-error} separates detection before the 96-batch input horizon from detection after the 100-batch queue drains. This is an explicitly project-defined ADWIN-style comparator, not a reproduction of River ADWIN.
\begin{table}[htbp]\centering\small\resizebox{\linewidth}{!}{\ReleasedErrorResults}\caption{Released-error detector outcomes by scheduled delay.}\label{tab:released-error}\end{table}
\begin{figure}[htbp]\centering\includegraphics[width=.96\linewidth]{figures/fig03_delay_verification.png}\caption{Release age and retrospective verified harm across the four delay settings.}\end{figure}

\section{Statistical analysis and reproducibility}
Complete seeds are the resampling unit. Point estimates use equal weight per seed after averaging the predeclared stream cells; a batch is not an independent replicate. Estimator contrasts are paired within seeds and Holm adjusted across the declared contrast family. Detector operating rates are reported by mechanism to avoid conflating high sensitivity with general quality. A full analysis manifest provides source hashes, exact cell count, bootstrap configuration, CSV checksums and figure checksums.

Reproduce from the project root with the Anaconda environment used for the measured run:
\begin{verbatim}
python scripts/run_strengthening_v3.py --profile evaluation `
  --output results/phase3/reproduction_evaluation_d8
python scripts/analyze_strengthening_v3.py `
  results/phase3/reproduction_evaluation_d8
python scripts/build_strengthening_report.py `
  results/phase3/reproduction_evaluation_d8 \\
  --output reports/phase3/reproduction_evaluation_d8
\end{verbatim}
Do not reuse an existing output directory for a new run. Resume only an interrupted run with the same lock, source snapshots and expected cells by adding \texttt{--resume}.

The optional dimensionality appendix uses a separate, 1,440-cell run:
\begin{verbatim}
python scripts/run_strengthening_v3.py --profile evaluation `
  --dimension 32 --dimension 128 `
  --output results/phase3/strengthening_v3_evaluation_sensitivity_d32_d128
python scripts/analyze_strengthening_v3.py `
  results/phase3/strengthening_v3_evaluation_sensitivity_d32_d128
python scripts/build_strengthening_report.py `
  results/phase3/strengthening_v3_evaluation_d8 `
  --sensitivity-analysis `
  results/phase3/strengthening_v3_evaluation_sensitivity_d32_d128/analysis `
  --output reports/phase3/strengthening_v3_evaluation_with_sensitivity_reproduced
\end{verbatim}

\section{Comparison with research and limits}
The classical accuracy-estimation literature emphasizes that unlabeled target performance requires assumptions about the source/target relationship; see Garg et al. (2022). Work on uncertainty under dataset shift similarly cautions that source calibration does not guarantee target reliability (Ovadia et al., 2019). This experiment contributes a controlled, label-safe comparison within its own generator laws; it does not transfer numerical rankings from those studies because their datasets, models, splits and estimands differ.

The five-region and conditional mechanisms test heterogeneous and label-law changes synthetically, not natural environments. The concern score uses plug-in probabilities and a selected harm margin; it is not calibrated risk or a population guarantee. The correlation alarm may increase stationary alert burden. No result here identifies causality, proves transfer to the pending weather stream, validates all named literature baselines, or measures decision quality. The prior v5 consolidated report and external-data supplements remain separate and unchanged; this report adds a versioned extension rather than replacing their results.

\begin{thebibliography}{9}
\bibitem{garg2022}S. Garg et al., ``Leveraging Unlabeled Data to Predict Out-of-Distribution Performance,'' 2022. \url{https://arxiv.org/abs/2201.04234}
\bibitem{ovadia2019}Y. Ovadia et al., ``Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift,'' NeurIPS, 2019. \url{https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html}
\end{thebibliography}
\end{document}
''', encoding="utf-8")
    receipt = {"run_dir": str(run_dir), "analysis_dir": str(analysis),
               "run_lock_sha256": sha256_file(run_dir / "protocol_lock.json"),
               "report_tex_sha256": sha256_file(output / "REPORT.tex"),
               "table_tex_sha256": sha256_file(output / "tables.tex"),
               "copied_figures": {p.name: sha256_file(p) for p in sorted((output / "figures").glob("*.png"))},
               "report_status": "editable TeX generated from completed, checksum-verified results"}
    (output / "report_build_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
