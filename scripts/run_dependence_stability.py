"""Run a locked, feature-only pairwise-dependence stability experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.core import holm
from drift_monitoring.phase3.dependence import calibrate_threshold, max_correlation_change

PROTOCOL = ROOT / "docs/phase3/DEPENDENCE_STABILITY_PROTOCOL_20261116_v1.md"
DETECTOR = ROOT / "src/drift_monitoring/phase3/dependence.py"
PAIRS = ((0, 1), (2, 5), (6, 7))
RHOS = (0.30, 0.60, 0.90)
SIZES = (250, 500, 1000)
BATCHES = 24
DEVELOPMENT_SEEDS = tuple(range(931000, 931030))
EVALUATION_SEEDS = tuple(range(931100, 931150))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rng_for(seed: int, *namespace: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), *map(int, namespace)]))


def correlated(z: np.ndarray, pair: tuple[int, int] | None, rho: float) -> np.ndarray:
    """Create a unit-marginal Gaussian pair intervention without changing other coordinates."""
    x = np.asarray(z, dtype=float).copy()
    if x.ndim != 2 or x.shape[1] != 8 or not np.isfinite(x).all():
        raise ValueError("z must be a finite n-by-8 matrix")
    if pair is None:
        if rho != 0:
            raise ValueError("rho must be zero for the stationary control")
        return x
    i, j = pair
    if not (0 <= i < j < 8) or not (0 <= rho < 1):
        raise ValueError("pair must be ordered feature indices and rho in [0,1)")
    x[:, j] = rho * z[:, i] + np.sqrt(1-rho*rho) * z[:, j]
    return x


def ks_alarm(ref_z: np.ndarray, ref_pca: np.ndarray, current_z: np.ndarray,
              pca: PCA) -> bool:
    cur_pca = pca.transform(current_z)
    pvals = [ks_2samp(ref_z[:, j], current_z[:, j], method="auto").pvalue
             for j in range(ref_z.shape[1])]
    pvals.extend(ks_2samp(ref_pca[:, j], cur_pca[:, j], method="auto").pvalue
                 for j in range(cur_pca.shape[1]))
    return bool(np.any(holm(pvals) <= .05))


def scenario_specs():
    specs = [("C0", None, 0.0)]
    specs.extend((f"rho{rho:.2f}_pair{i}{j}", (i, j), rho)
                 for rho in RHOS for i, j in PAIRS)
    return specs


def run(mode: str, out: Path, threshold_file: Path | None = None) -> None:
    out = out.resolve()
    if out.exists():
        raise FileExistsError(out)
    if mode == "development":
        seeds, specs = DEVELOPMENT_SEEDS, [("C0", None, 0.0)]
        thresholds = None
    elif mode == "evaluation":
        if threshold_file is None:
            raise ValueError("evaluation mode requires the development threshold lock")
        lock = json.loads(threshold_file.read_text(encoding="utf-8"))
        if (lock.get("mode") != "development"
                or lock.get("protocol_sha256") != digest(PROTOCOL)
                or lock.get("source_sha256") != digest(Path(__file__))
                or lock.get("detector_sha256") != digest(DETECTOR)):
            raise ValueError("threshold lock does not match current protocol, runner, and detector")
        thresholds = {int(k): float(v) for k, v in lock["threshold_by_batch_size"].items()}
        seeds, specs = EVALUATION_SEEDS, scenario_specs()
    else:
        raise ValueError(mode)

    out.mkdir(parents=True)
    rows = []
    with threadpool_limits(limits=1):
        for seed in seeds:
            train = rng_for(seed, 11).normal(size=(2000, 8))
            reference = rng_for(seed, 12).normal(size=(2000, 8))
            scaler = StandardScaler().fit(train)
            train_z = scaler.transform(train)
            reference_z = scaler.transform(reference)
            pca = PCA(n_components=3, svd_solver="full").fit(train_z)
            reference_pca = pca.transform(reference_z)
            for n in SIZES:
                for scenario, pair, rho in specs:
                    for batch in range(1, BATCHES + 1):
                        # Shared innovations across the paired intervention grid.
                        z = rng_for(seed, 21, n, batch).normal(size=(n, 8))
                        active = scenario != "C0" and batch >= 9
                        x = correlated(z, pair if active else None, rho if active else 0.0)
                        score, found_pair = max_correlation_change(reference, x)
                        current_z = scaler.transform(x)
                        alarm_ks = ks_alarm(reference_z, reference_pca, current_z, pca)
                        threshold = thresholds.get(n) if thresholds is not None else np.nan
                        rows.append(dict(seed=seed, scenario=scenario, rho=rho,
                            target_pair=(f"{pair[0]}-{pair[1]}" if pair else "none"),
                            batch_size=n, batch=batch, intervention_active=active,
                            dependence_score=score, found_i=found_pair[0], found_j=found_pair[1],
                            dependence_threshold=threshold,
                            dependence_alarm=(score >= threshold) if thresholds is not None else False,
                            ks_pca_alarm=alarm_ks))

    batches = pd.DataFrame(rows)
    if mode == "development":
        threshold_map = {}
        dev_burden = {}
        for n in SIZES:
            scores = batches.loc[batches.batch_size.eq(n), "dependence_score"].to_numpy()
            threshold = calibrate_threshold(scores, .95)
            threshold_map[str(n)] = threshold
            alarms = scores >= threshold
            dev_burden[str(n)] = dict(stationary_batches=len(scores), alarms=int(alarms.sum()),
                                      alarms_per_24_batches=float(alarms.reshape(-1, 24).sum(axis=1).mean()))
        batches["dependence_alarm"] = batches.apply(
            lambda r: r.dependence_score >= threshold_map[str(int(r.batch_size))], axis=1)
        batches["dependence_threshold"] = batches.batch_size.map(
            {n: threshold_map[str(n)] for n in SIZES})
        lock = dict(protocol="IDMVIZ_DEPENDENCE_STABILITY_20261116_V1", mode="development",
            development_seeds=[DEVELOPMENT_SEEDS[0], DEVELOPMENT_SEEDS[-1]],
            batch_sizes=list(SIZES), quantile=.95, quantile_rule="higher",
            threshold_by_batch_size=threshold_map, development_burden=dev_burden,
            protocol_sha256=digest(PROTOCOL), source_sha256=digest(Path(__file__)),
            detector_sha256=digest(DETECTOR))
        (out / "threshold.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
    else:
        (out / "threshold_used.json").write_text(json.dumps(dict(
            protocol="IDMVIZ_DEPENDENCE_STABILITY_20261116_V1", mode="evaluation",
            development_threshold_sha256=digest(threshold_file),
            protocol_sha256=digest(PROTOCOL), source_sha256=digest(Path(__file__)),
            detector_sha256=digest(DETECTOR), thresholds=thresholds), indent=2), encoding="utf-8")

    batches.to_csv(out / "online_dependence_stability.csv", index=False)
    env = dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
               protocol_sha256=digest(PROTOCOL), source_sha256=digest(Path(__file__)),
               detector_sha256=digest(DETECTOR), mode=mode)
    (out / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(json.dumps(dict(mode=mode, out=str(out), rows=len(batches), seeds=len(seeds)), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("development", "evaluation"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold-file", type=Path)
    args = parser.parse_args()
    run(args.mode, args.out, args.threshold_file)
