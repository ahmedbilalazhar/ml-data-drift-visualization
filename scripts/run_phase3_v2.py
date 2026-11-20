"""Run the locked IDM-Viz v2 extension without touching v1 outputs."""
from __future__ import annotations

import os
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.v2 import (BATCHES, BATCH_SIZE, DELAYS, MODELS, TASKS,
    V1_CONDITIONS, XOR_CONDITIONS, FeatureBatch, fit_models, historical_partition,
    monitoring_partition, score_model)
from drift_monitoring.phase3.replay import LabelBroker, replay


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    paths = [ROOT / "src/drift_monitoring/phase3/v2.py",
             ROOT / "src/drift_monitoring/phase3/core.py",
             ROOT / "src/drift_monitoring/phase3/replay.py",
             ROOT / "scripts/run_phase3_v2.py",
             ROOT / "scripts/evaluate_phase3_v2.py",
             ROOT / "tests/phase3/test_v2_extension_contract.py",
             ROOT / "docs/phase3/V2_EXTENSION_MANIFEST.md"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths}


def memory_record():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return dict(rss_bytes=info.rss, peak_working_set_bytes=getattr(info, "peak_wset", None))
    except ImportError:
        return dict(rss_bytes=None, peak_working_set_bytes=None)


def environment_record():
    packages = ("numpy", "scipy", "scikit-learn", "pandas", "matplotlib", "pytest", "threadpoolctl")
    return dict(executable=sys.executable, python=sys.version, platform=platform.platform(),
                packages={name: importlib.metadata.version(name) for name in packages},
                thread_limit=1, memory_start=memory_record())


def run(mode: str, out: Path):
    started = time.perf_counter()
    manifest_path = ROOT / "docs/phase3/V2_EXTENSION_MANIFEST.md"
    if out.exists():
        raise FileExistsError(f"refusing to overwrite v2 run: {out}")
    out.mkdir(parents=True)
    for sub in ("predictions", "evaluator_truth", "historical_ids"):
        (out / sub).mkdir()
    seeds_by_task = ({"v1_synthetic": [190001, 190002, 190003, 190004, 190005],
                      "xor_synthetic": [190001, 190002, 190003, 190004, 190005]} if mode == "development" else
                     {"v1_synthetic": list(range(100, 130)), "xor_synthetic": list(range(200, 230))})
    protocol = dict(protocol="IDM-Viz Phase 3 v2 outcome-informed extension", mode=mode,
                    status="running", started_at_utc=datetime.now(timezone.utc).isoformat(),
                    seeds_by_task=seeds_by_task, conditions_by_task={
                        "v1_synthetic": list(V1_CONDITIONS), "xor_synthetic": list(XOR_CONDITIONS)},
                    models=list(MODELS), delays=list(DELAYS), batch_size=BATCH_SIZE,
                    batches_per_stream=BATCHES, manifest_sha256=sha(manifest_path),
                    source_sha256=source_hashes(), thread_limit=1,
                    original_documents_sha256={
                        "docs/23i-2604_23i-2581DAV-phase2.docx": sha(ROOT / "docs/23i-2604_23i-2581DAV-phase2.docx"),
                        "docs/Phase 03 - Implementation.pdf": sha(ROOT / "docs/Phase 03 - Implementation.pdf")},
                    v1_archives_sha256={
                        "IDM_Viz_Phase3_v1_submission.zip": sha(ROOT / "output/phase3/IDM_Viz_Phase3_v1_submission.zip"),
                        "IDM_Viz_Phase3_v1_full_reproducibility.zip": sha(ROOT / "output/phase3/IDM_Viz_Phase3_v1_full_reproducibility.zip")})
    (out / "protocol_lock.json").write_text(json.dumps(protocol, indent=2))
    (out / "environment.json").write_text(json.dumps(environment_record(), indent=2))
    online, events, verifications, fits = [], [], [], []
    (out / "evaluator_truth").mkdir(exist_ok=True)
    try:
        with threadpool_limits(limits=1):
            for task in TASKS:
                conditions = protocol["conditions_by_task"][task]
                for seed in seeds_by_task[task]:
                    seed_started = time.perf_counter()
                    train = historical_partition(task, seed, 0, 4000)
                    calibration = historical_partition(task, seed, 1, 2000)
                    reference = historical_partition(task, seed, 2, 2000)
                    artifacts = fit_models(seed, train, calibration, reference)
                    np.savez_compressed(out / "historical_ids" / f"{task}_{seed}.npz",
                                        **artifacts["logistic"].fit_ids)
                    for model_name, artifact in artifacts.items():
                        fit_row = dict(task=task, seed=seed, model=model_name,
                                       fit_seconds=artifact.fit_seconds,
                                       reference_correct=artifact.reference_correct,
                                       reference_n=artifact.reference_n,
                                       reference_accuracy=artifact.reference_correct / artifact.reference_n,
                                       atc_threshold=artifact.atc)
                        if model_name == "logistic":
                            fit_row["logistic_iterations"] = int(artifact.model.n_iter_.max())
                        else:
                            fit_row["tree_depth"] = int(artifact.model.get_depth())
                            fit_row["tree_leaves"] = int(artifact.model.get_n_leaves())
                        fits.append(fit_row)
                    for condition in conditions:
                        part = monitoring_partition(task, seed, condition)
                        truth_path = out / "evaluator_truth" / f"{task}_{seed}_{condition}.npz"
                        np.savez_compressed(truth_path, ids=part.ids, labels=part.y)
                        for model_name, artifact in artifacts.items():
                            records, caches = [], []
                            for batch in range(1, BATCHES + 1):
                                sl = slice((batch - 1) * BATCH_SIZE, batch * BATCH_SIZE)
                                record, cache = score_model(artifact, FeatureBatch(
                                    part.x[sl], part.ids[sl], seed, condition, batch), task, model_name)
                                records.append(record)
                                caches.append(cache)
                            cache_path = out / "predictions" / f"{task}_{seed}_{condition}_{model_name}.npz"
                            np.savez_compressed(cache_path, **{
                                key: np.stack([item[key] for item in caches]) for key in caches[0]})
                            pd.DataFrame(records).to_csv(
                                out / "predictions" / f"{task}_{seed}_{condition}_{model_name}_online.csv", index=False)
                            for delay in DELAYS:
                                broker = LabelBroker(part.ids, part.y, BATCH_SIZE, delay)
                                replay_events, verified = replay(records, caches, broker)
                                events.extend(dict(task=task, model=model_name, **item) for item in replay_events)
                                verifications.extend(dict(task=task, model=model_name, **item) for item in verified)
                            online.extend(records)
                    print(f"{task} seed {seed}: {time.perf_counter()-seed_started:.1f}s", flush=True)
        pd.DataFrame(online).to_csv(out / "online_batches.csv", index=False)
        pd.DataFrame(events).to_csv(out / "clock_events.csv", index=False)
        pd.DataFrame(verifications).to_csv(out / "released_verification.csv", index=False)
        pd.DataFrame(fits).to_csv(out / "fit_records.csv", index=False)
        protocol.update(status="complete", ended_at_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter() - started, memory_end=memory_record(),
                        streams=sum(len(seeds_by_task[t]) * len(protocol["conditions_by_task"][t]) for t in TASKS),
                        model_streams=len(online) // BATCHES, online_batches=len(online),
                        monitoring_rows=sum(len(seeds_by_task[t]) * len(protocol["conditions_by_task"][t])
                                            for t in TASKS) * BATCH_SIZE * BATCHES,
                        verification_rows=len(verifications))
    except BaseException as error:
        protocol.update(status="failed", error=repr(error), elapsed_seconds=time.perf_counter() - started)
        raise
    finally:
        (out / "protocol_lock.json").write_text(json.dumps(protocol, indent=2))
    print(json.dumps({key: protocol[key] for key in
                      ("status", "elapsed_seconds", "streams", "model_streams", "online_batches")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("development", "final"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.mode, args.out.resolve())
