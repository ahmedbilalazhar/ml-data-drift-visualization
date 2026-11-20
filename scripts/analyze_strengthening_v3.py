"""Reproducible seed-level analysis and publication figures for a v3 run."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.contracts import atomic_write_json, sha256_file  # noqa: E402
from drift_monitoring.phase3.core import holm  # noqa: E402
from drift_monitoring.phase3.released_error import ReleasedErrorADWINStyle  # noqa: E402

PHASES = {
    "stationary_1_24": (1, 24),
    "early_impact_25_40": (25, 40),
    "post_ramp_41_64": (41, 64),
    "recurrent_65_80": (65, 80),
    "late_81_96": (81, 96),
}
ESTIMATORS = ("cal", "raw", "atc", "constant")
COLORS = {"cal": "#137C8B", "raw": "#E76F51", "atc": "#6C63A8", "constant": "#8993A4"}


def _read_cell(path):
    with zipfile.ZipFile(path) as zf:
        online = pd.read_csv(io.BytesIO(zf.read("online/batch_scores.csv")))
        delays = pd.read_csv(io.BytesIO(zf.read("run/delayed_verification.csv")))
        events = pd.read_csv(io.BytesIO(zf.read("run/label_release_events.csv")))
        with np.load(io.BytesIO(zf.read("online/prediction_cache.npz"))) as p:
            ids = p["ids"].copy()
            prediction = p["prediction"].copy()
            correctness_probability = p["correctness_probability"].copy()
        with np.load(io.BytesIO(zf.read("evaluator/truth.npz"))) as t:
            labels = t["labels"].copy()
            expected = t["expected_label_probability"].copy()
            region = t["region_id"].copy()
        metadata = json.loads(zf.read("evaluator/truth_metadata.json"))
    if not (len(online) == 96 and len(labels) == len(prediction) == len(expected) == 48000):
        raise ValueError(f"invalid v3 cell dimensions in {path}")
    if not np.isin(labels, (0, 1)).all() or not np.isin(prediction, (0, 1)).all():
        raise ValueError(f"invalid binary labels or predictions in {path}")
    return online, delays, events, ids, prediction, correctness_probability, labels, expected, region, metadata


def _bootstrap(values, rng, reps=2000):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan, 0
    if len(values) == 1:
        return values[0], np.nan, 1
    draws = rng.integers(0, len(values), size=(reps, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975)), len(values)


def _load_run(run_dir):
    lock_path = run_dir / "protocol_lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("state") != "complete":
        raise ValueError(f"run is not complete (state={lock.get('state')})")
    expected = set(lock["expected_cells"])
    completed = lock["completed_cells"]
    if set(completed) != expected:
        raise ValueError("completed cell registry differs from locked expected cells")
    rows, temporal, delayed, localization, detection, released_detection, released_trace = [], [], [], [], [], [], []
    cache_checks = {}
    for cell in sorted(expected):
        item = completed[cell]
        path = run_dir / item["artifact"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"artifact checksum mismatch: {cell}")
        online, delays, events, ids, pred, q, y, expected_p, region, metadata = _read_cell(path)
        task = str(online.task.iloc[0])
        mechanism = str(online.condition.iloc[0])
        seed = int(online.seed.iloc[0])
        dim = int(online.feature_dimension.iloc[0])
        if (online.batch.tolist() != list(range(1, 97)) or len(ids) != 48000
                or len(np.unique(ids)) != 48000 or not np.isfinite(q).all()):
            raise ValueError(f"online score/cache schema failed for {cell}")
        if set(delays.delay.astype(int)) != {0, 5, 20, 100}:
            raise ValueError(f"delay set mismatch in {cell}")
        for delay, dg in delays.groupby("delay"):
            dg = dg.sort_values("batch")
            if (len(dg) != 96 or not np.array_equal(dg.batch.to_numpy(), np.arange(1, 97))
                    or not np.array_equal(dg.available_at.to_numpy(), dg.batch.to_numpy() + int(delay))
                    or not np.all(dg.evidence_age_batches.to_numpy() == int(delay))):
                raise ValueError(f"delayed release clock/age invariant failed: {cell}, delay={delay}")
            de = events[(events.delay == delay) & (events.event == "labels_verified")]
            if (len(de) != 96 or not np.array_equal(de.batch.to_numpy(), np.arange(1, 97))
                    or not np.array_equal(de.clock.to_numpy(), de.batch.to_numpy() + int(delay))):
                raise ValueError(f"release event order/count invariant failed: {cell}, delay={delay}")
            persisted = events[(events.delay == delay) & (events.event == "prediction_persisted")].sort_values("clock")
            if (len(persisted) != 96 or not np.array_equal(persisted.batch.to_numpy(), np.arange(1, 97))):
                raise ValueError(f"predictions were not persisted once per batch: {cell}, delay={delay}")
            detector = ReleasedErrorADWINStyle(delta=.002, min_window_batches=2)
            alarms = []
            for v in dg.sort_values(["available_at", "batch"]).itertuples(index=False):
                out = detector.update(int(v.n - v.correct), int(v.n))
                row = {"task": task, "mechanism": mechanism, "seed": seed,
                       "feature_dimension": dim, "delay": int(delay),
                       "source_batch": int(v.batch), "available_at": int(v.available_at),
                       "error_rate": float(1.0 - v.accuracy), **out}
                released_trace.append(row)
                if out["alarm"]:
                    alarms.append(row)
            pre_alarms = [a for a in alarms if a["source_batch"] <= 24]
            changed_alarms = [a for a in alarms if a["source_batch"] >= 25]
            online_changed = [a for a in changed_alarms if a["available_at"] <= 96]
            first = changed_alarms[0] if changed_alarms else None
            released_detection.append({
                "task": task, "mechanism": mechanism, "seed": seed,
                "feature_dimension": dim, "delay": int(delay),
                "stationary_false_alarm": bool(pre_alarms),
                "stationary_alarm_count": len(pre_alarms),
                "post_change_detected_after_drain": bool(changed_alarms),
                "post_change_detected_during_input": bool(online_changed),
                "first_changed_alarm_source_batch": np.nan if first is None else first["source_batch"],
                "first_changed_alarm_release_clock": np.nan if first is None else first["available_at"],
                "detection_lag_source_batches": np.nan if first is None else first["source_batch"] - 25,
                "detector_name": "ADWIN-style exact scan; not a River/official ADWIN reproduction"})
        if mechanism in {"M0_stationary", "M4_abrupt_conditional_reversal",
                         "M5_recurrent_conditional_reversal"}:
            cache_checks[(task, seed, dim, mechanism)] = (online.copy(), pred.copy(), q.copy())
        if "correlation_alarm" not in online:
            online["correlation_alarm"] = False
        pre = online[online.batch <= 24]
        post = online[online.batch >= 25]
        for detector, column in (("original_plus_pca_ks", "input_alarm"),
                                 ("pairwise_correlation", "correlation_alarm")):
            alarm_batches = online.loc[online.batch >= 25].loc[
                online.loc[online.batch >= 25, column].astype(bool), "batch"]
            detection.append({"task": task, "mechanism": mechanism, "seed": seed,
                              "feature_dimension": dim, "detector": detector,
                              "stationary_false_alarm_rate": float(pre[column].mean()),
                              "post_intervention_detection": bool(len(alarm_batches)),
                              "first_alarm_batch": int(alarm_batches.min()) if len(alarm_batches) else np.nan,
                              "detection_delay_batches": int(alarm_batches.min() - 25) if len(alarm_batches) else np.nan,
                              "alarm_rate_post_intervention": float(post[column].mean()),
                              "is_stationary_control": mechanism == "M0_stationary"})
        accuracy = np.empty(96, dtype=float)
        confusion = np.empty((96, 4), dtype=np.int64)
        brier = np.empty(96, dtype=float)
        for b in range(96):
            sl = slice(b * 500, (b + 1) * 500)
            correct = pred[sl] == y[sl]
            accuracy[b] = correct.mean()
            confusion[b] = [np.sum((y[sl] == 0) & (pred[sl] == 0)),
                            np.sum((y[sl] == 0) & (pred[sl] == 1)),
                            np.sum((y[sl] == 1) & (pred[sl] == 0)),
                            np.sum((y[sl] == 1) & (pred[sl] == 1))]
            brier[b] = np.mean((q[sl] - correct.astype(float)) ** 2)
        online["accuracy_truth"] = accuracy
        online["brier_correctness"] = brier
        verified_zero = delays[delays.delay == 0].sort_values("batch")
        if len(verified_zero) != 96 or not np.array_equal(verified_zero.batch.to_numpy(), np.arange(1, 97)):
            raise ValueError(f"delay-zero verifier is incomplete in {path}")
        online["macro_f1_truth"] = verified_zero.macro_f1.to_numpy()
        online["verified_harm"] = verified_zero.harm.to_numpy()
        for phase, (lo, hi) in PHASES.items():
            part = online[(online.batch >= lo) & (online.batch <= hi)]
            actual = part.accuracy_truth.to_numpy()
            for estimator in ESTIMATORS:
                estimate = part[f"{estimator}_estimate"].to_numpy()
                rows.append({"task": task, "mechanism": mechanism, "seed": seed,
                             "feature_dimension": dim, "phase": phase, "estimator": estimator,
                             "mae": float(np.mean(np.abs(estimate - actual))),
                             "bias": float(np.mean(estimate - actual)),
                             "brier_correctness": float(part.brier_correctness.mean()),
                             "mean_accuracy": float(actual.mean()),
                             "concern_rate": float(part.cal_concern.mean()),
                             "harm_rate": float(part.verified_harm.mean()),
                             "interval_coverage": float(np.mean((part.cal_lower <= actual) & (actual <= part.cal_upper))),
                             "interval_width": float(np.mean(part.cal_upper - part.cal_lower))})
        for r in online.itertuples(index=False):
            temporal.append({"task": task, "mechanism": mechanism, "seed": seed,
                             "feature_dimension": dim, "batch": int(r.batch),
                             "accuracy": float(r.accuracy_truth), "input_alarm": bool(r.input_alarm),
                             "correlation_alarm": bool(r.correlation_alarm),
                             "original_alarm": bool(r.original_alarm), "cal_estimate": float(r.cal_estimate),
                             "cal_concern": float(r.cal_concern), "cal_lower": float(r.cal_lower),
                             "cal_upper": float(r.cal_upper), "brier_correctness": float(r.brier_correctness)})
        for (delay, phase), dg in delays.assign(phase=delays.batch.map(
                lambda b: next((name for name, (lo, hi) in PHASES.items() if lo <= b <= hi), "other"))).groupby(
                    ["delay", "phase"], sort=True):
            delayed.append({"task": task, "mechanism": mechanism, "seed": seed,
                            "feature_dimension": dim, "delay": int(delay), "phase": phase,
                            "mean_accuracy_at_release": float(dg.accuracy.mean()),
                            "mean_evidence_age_batches": float(dg.evidence_age_batches.mean()),
                            "verified_harm_rate": float(dg.harm.mean()), "verified_batches": len(dg)})
        changed = list(map(int, metadata.get("changed_marginal_features", [])))
        if changed:
            for r in online.itertuples(index=False):
                ranks = np.array([getattr(r, f"x{j}_rank") for j in range(dim)])
                localization.append({"task": task, "mechanism": mechanism, "seed": seed,
                                     "feature_dimension": dim, "batch": int(r.batch),
                                     "top1_changed": int(np.argmin(ranks) in changed),
                                     "recall_at_3": len(set(np.argsort(ranks)[:3]) & set(changed)) / len(changed),
                                     "changed_features": ";".join(map(str, changed))})
    # Conditional-only mechanisms share input rows with the stationary control.
    # Their cached prediction scores must therefore be identical, regardless of labels.
    for (task, seed, dim, mechanism), (online, pred, q) in cache_checks.items():
        if mechanism == "M0_stationary":
            continue
        control = cache_checks.get((task, seed, dim, "M0_stationary"))
        if control is None:
            raise ValueError(f"missing paired stationary control for {(task, seed, dim, mechanism)}")
        control_online, control_pred, control_q = control
        ks_cols = [c for c in online if c.startswith("x") and c.endswith("_ks")]
        if (not np.array_equal(pred, control_pred) or not np.array_equal(q, control_q)
                or not np.array_equal(online[ks_cols].to_numpy(), control_online[ks_cols].to_numpy())):
            raise ValueError(f"label-independent prediction invariant failed: {(task, seed, dim, mechanism)}")
    return (pd.DataFrame(rows), pd.DataFrame(temporal), pd.DataFrame(delayed),
            pd.DataFrame(localization), pd.DataFrame(detection),
            pd.DataFrame(released_detection), pd.DataFrame(released_trace), lock)


def _summary(per_seed):
    rng = np.random.default_rng(73116290)
    keys = ["task", "mechanism", "feature_dimension", "phase", "estimator"]
    rows = []
    for key, group in per_seed.groupby(keys, dropna=False, sort=True):
        row = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        for metric in ("mae", "bias", "brier_correctness", "mean_accuracy", "concern_rate",
                       "harm_rate", "interval_coverage", "interval_width"):
            vals = group[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            low, high, n = _bootstrap(vals, rng)
            row.update({f"{metric}_mean": float(np.mean(vals)) if len(vals) else np.nan,
                        f"{metric}_ci_low": low, f"{metric}_ci_high": high,
                        f"{metric}_defined_seeds": n})
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_tests(per_seed):
    rows = []
    group_cols = ["task", "mechanism", "feature_dimension", "phase"]
    for key, group in per_seed.groupby(group_cols, sort=True):
        wide = group.pivot(index="seed", columns="estimator", values="mae")
        for comp in ("raw", "atc", "constant"):
            diff = (wide["cal"] - wide[comp]).dropna().to_numpy()
            if len(diff) < 2:
                statistic, p = np.nan, np.nan
            else:
                # Paired, seed-level sign-flip randomization test; no batches are treated as units.
                rng = np.random.default_rng(73116300 + len(rows))
                observed = abs(float(np.mean(diff)))
                signs = rng.choice((-1.0, 1.0), size=(10000, len(diff)))
                statistic = float(np.mean(diff))
                p = float((1 + np.sum(np.abs((signs * diff).mean(axis=1)) >= observed)) / 10001)
            rows.append(dict(zip(group_cols, key), contrast=f"cal_minus_{comp}_mae",
                              mean_paired_difference=statistic, p_raw=p, seeds=len(diff)))
    frame = pd.DataFrame(rows)
    if len(frame):
        valid = frame.p_raw.notna()
        if valid.any():
            frame.loc[valid, "p_holm"] = holm(frame.loc[valid, "p_raw"].to_numpy())
    return frame


def _figures(out, per_seed, summary, temporal, delayed, localization):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "figure.facecolor": "#F4F7FA", "axes.facecolor": "white",
                         "savefig.facecolor": "#F4F7FA", "axes.edgecolor": "#D6DEE8",
                         "axes.labelcolor": "#344054", "text.color": "#172B4D",
                         "xtick.color": "#475467", "ytick.color": "#475467",
                         "grid.color": "#E7ECF2", "grid.linewidth": .7})
    dim = int(temporal.feature_dimension.max())
    plot_sum = summary[summary.feature_dimension == dim]
    tasks = sorted(plot_sum.task.unique())
    mechanisms = sorted(temporal.mechanism.unique())

    fig, axes = plt.subplots(1, len(tasks), figsize=(14.2, 5.2), constrained_layout=True, sharey=True)
    axes_arr = np.atleast_1d(axes)
    for ax, task in zip(axes_arr, tasks):
        d = per_seed[(per_seed.task == task) & (per_seed.feature_dimension == dim) &
                     (per_seed.phase.isin(["early_impact_25_40", "recurrent_65_80"]))]
        d = d.groupby(["mechanism", "seed", "estimator"], as_index=False).mae.mean()
        x = np.arange(len(mechanisms)); width = .19
        for j, estimator in enumerate(ESTIMATORS):
            means, lows, highs = [], [], []
            for mech in mechanisms:
                values = d[(d.mechanism == mech) & (d.estimator == estimator)].mae.to_numpy()
                if len(values):
                    rng = np.random.default_rng(73116500 + j + mechanisms.index(mech))
                    lo, hi, _ = _bootstrap(values, rng)
                    means.append(float(values.mean())); lows.append(lo); highs.append(hi)
                else:
                    means.append(np.nan); lows.append(np.nan); highs.append(np.nan)
            means, lows, highs = map(np.asarray, (means, lows, highs))
            errors = np.vstack([np.maximum(0, means - lows), np.maximum(0, highs - means)])
            ax.bar(x + (j - 1.5) * width, means, width, yerr=errors,
                   error_kw={"elinewidth": .7, "capsize": 1.5},
                   color=COLORS[estimator], label=estimator.upper(), alpha=.94)
        ax.set_title(task.replace("_", " ").title(), loc="left", weight="bold", pad=10)
        ax.set_xticks(x, [m.split("_")[0] for m in mechanisms], rotation=0)
        ax.grid(axis="y", zorder=0); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes_arr[0].set_ylabel("Seed-mean absolute error in accuracy estimate")
    axes_arr[0].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("Accuracy estimation under controlled drift", x=.02, ha="left", fontsize=15, weight="bold")
    fig.savefig(out / "fig01_estimator_mae.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(4, 2, figsize=(13.5, 12.5), sharex=True, constrained_layout=False)
    fig.subplots_adjust(top=.89, bottom=.055, left=.075, right=.99, hspace=.40, wspace=.12)
    for ax, mech in zip(axes.flat, mechanisms):
        d = temporal[(temporal.mechanism == mech) & (temporal.feature_dimension == dim)]
        g = d.groupby("batch").agg(acc=("accuracy", "mean"), alarm=("input_alarm", "mean"),
                                    corr_alarm=("correlation_alarm", "mean"),
                                    low=("cal_lower", "mean"), high=("cal_upper", "mean"))
        ax.plot(g.index, g.acc, color="#137C8B", lw=1.8, label="Observed accuracy")
        ax.plot(g.index, g.alarm, color="#E76F51", lw=1.3, label="KS/PCA alarm rate")
        ax.plot(g.index, g.corr_alarm, color="#6C63A8", lw=1.2, ls=":", label="Correlation alarm rate")
        ax.fill_between(g.index, g.low, g.high, color="#137C8B", alpha=.12, label="Mean plug-in interval")
        ax.axvline(24.5, color="#475467", lw=1, ls="--")
        ax.set_title(mech.replace("_", " "), loc="left", fontsize=9, weight="bold")
        ax.set_ylim(-.04, 1.04); ax.grid(axis="y"); ax.spines[["top", "right"]].set_visible(False)
    axes[-1, 0].set_xlabel("Completed batch"); axes[-1, 1].set_xlabel("Completed batch")
    axes[0, 0].set_ylabel("Rate / accuracy"); axes[1, 0].set_ylabel("Rate / accuracy")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.56, .955))
    fig.suptitle("Online evidence and realized outcomes", x=.02, y=.99,
                 ha="left", fontsize=15, weight="bold")
    fig.savefig(out / "fig02_detection_and_accuracy_timeline.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), constrained_layout=True)
    delay_plot = delayed[(delayed.feature_dimension == dim) & (delayed.phase == "early_impact_25_40")]
    for ax, metric, ylabel in [(axes[0], "mean_evidence_age_batches", "Evidence age at release (batches)"),
                               (axes[1], "verified_harm_rate", "Verified harm rate")]:
        for mech in mechanisms:
            g = delay_plot[delay_plot.mechanism == mech].groupby("delay")[metric].mean().sort_index()
            ax.plot(g.index, g.values, marker="o", lw=1.6, label=mech.split("_")[0])
        ax.set_xlabel("Scheduled label delay (batches)"); ax.set_ylabel(ylabel)
        ax.grid(axis="y"); ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(title="Mechanism", frameon=False, ncol=2)
    fig.suptitle("Delayed-label verification and staleness", x=.02, ha="left", fontsize=15, weight="bold")
    fig.savefig(out / "fig03_delay_verification.png", dpi=240, bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.8, 5.4), constrained_layout=True)
    if len(localization):
        d = localization[(localization.feature_dimension == dim) & localization.batch.between(25, 96)]
        g = d.groupby(["task", "mechanism"])[["top1_changed", "recall_at_3"]].mean().reset_index()
        labels = [f"{t.split('_')[0]} · {m.split('_')[0]}" for t, m in zip(g.task, g.mechanism)]
        xx = np.arange(len(g)); w = .36
        ax.bar(xx - w/2, g.top1_changed, w, color="#137C8B", label="Changed feature at rank 1")
        ax.bar(xx + w/2, g.recall_at_3, w, color="#E9A23B", label="Changed-feature recall@3")
        ax.set_xticks(xx, labels, rotation=35, ha="right"); ax.set_ylim(0, 1)
        ax.legend(frameon=False, ncol=2, loc="upper right")
    else:
        ax.text(.5, .5, "No marginal-feature localization targets in this run.", ha="center", va="center")
        ax.set_axis_off()
    ax.set_ylabel("Localization score"); ax.grid(axis="y"); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Descriptive feature localization", x=.02, ha="left", fontsize=15, weight="bold")
    fig.savefig(out / "fig04_feature_localization.png", dpi=240, bbox_inches="tight"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    out = (args.output or run_dir / "analysis").resolve()
    out.mkdir(parents=True, exist_ok=True)
    (per_seed, temporal, delayed, localization, detection,
     released_detection, released_trace, lock) = _load_run(run_dir)
    summary = _summary(per_seed)
    paired = _paired_tests(per_seed)
    delay_summary = delayed.groupby(["task", "mechanism", "feature_dimension", "delay", "phase"],
                                    as_index=False).agg(
        mean_accuracy_at_release=("mean_accuracy_at_release", "mean"),
        mean_evidence_age_batches=("mean_evidence_age_batches", "mean"),
        verified_harm_rate=("verified_harm_rate", "mean"),
        verified_batches=("verified_batches", "sum"), seed_count=("seed", "nunique"))
    detection_summary = detection.groupby(["task", "mechanism", "feature_dimension", "detector"],
                                         as_index=False).agg(
        stationary_false_alarm_rate=("stationary_false_alarm_rate", "mean"),
        post_intervention_detection_rate=("post_intervention_detection", "mean"),
        mean_detection_delay_batches=("detection_delay_batches", "mean"),
        post_intervention_alarm_rate=("alarm_rate_post_intervention", "mean"),
        seed_count=("seed", "nunique"))
    released_summary = released_detection.groupby(
        ["task", "mechanism", "feature_dimension", "delay"], as_index=False).agg(
        stationary_false_alarm_rate=("stationary_false_alarm", "mean"),
        mean_stationary_alarm_count=("stationary_alarm_count", "mean"),
        detected_after_drain_rate=("post_change_detected_after_drain", "mean"),
        detected_during_input_rate=("post_change_detected_during_input", "mean"),
        mean_detection_lag_source_batches=("detection_lag_source_batches", "mean"),
        seed_count=("seed", "nunique"))
    per_seed.to_csv(out / "estimation_per_seed.csv", index=False)
    summary.to_csv(out / "estimation_summary.csv", index=False)
    paired.to_csv(out / "paired_estimator_tests.csv", index=False)
    temporal.to_csv(out / "temporal_per_seed.csv", index=False)
    delayed.to_csv(out / "delayed_verification_per_seed.csv", index=False)
    delay_summary.to_csv(out / "delayed_verification_summary.csv", index=False)
    localization.to_csv(out / "localization_per_seed_batch.csv", index=False)
    detection.to_csv(out / "detection_per_seed.csv", index=False)
    detection_summary.to_csv(out / "detection_summary.csv", index=False)
    released_detection.to_csv(out / "released_error_detection_per_seed.csv", index=False)
    released_summary.to_csv(out / "released_error_detection_summary.csv", index=False)
    released_trace.to_csv(out / "released_error_detector_trace.csv", index=False)
    _figures(out, per_seed, summary, temporal, delay_summary, localization)
    manifest = {"run_dir": str(run_dir), "protocol_lock_sha256": sha256_file(run_dir / "protocol_lock.json"),
                "analyzer_sha256": sha256_file(Path(__file__)),
                "released_error_detector_sha256": sha256_file(
                    ROOT / "src/drift_monitoring/phase3/released_error.py"),
                "run_state": lock["state"], "completed_cells": len(lock["completed_cells"]),
                "bootstrap_replicates": 2000, "bootstrap_unit": "independent seed",
                "paired_tests": "two-sided seed-level sign-flip randomization; Holm across reported contrasts",
                "released_error_detector": "ADWIN-style exact split scan of released cohort errors; not a River/official ADWIN implementation",
                "decision_effect_evaluation": "not measured; the evidence does not support analyst decision-benefit claims",
                "files": {p.name: sha256_file(p) for p in sorted(out.iterdir()) if p.is_file()}}
    atomic_write_json(out / "analysis_manifest.json", manifest)
    print(json.dumps({"analysis": str(out), "cells": len(lock["completed_cells"]),
                      "seed_rows": len(per_seed), "figures": 4}, indent=2))


if __name__ == "__main__":
    main()
