"""Execute the locked Phase 3 protocol; never reads legacy result files."""
from __future__ import annotations
import os
# Bound numerical libraries before importing numpy/sklearn on a four-core laptop.
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
import warnings
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from drift_monitoring.phase3.core import (historical_partition, monitoring_partition,
    fit_artifacts, score_batch, FeatureBatch)
from drift_monitoring.phase3.replay import LabelBroker, replay


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(m):
    """Fail on unsupported changes rather than silently ignoring manifest edits."""
    expected = {
        'batch_size': 500, 'batches_per_stream': 24, 'features': 8,
        'baseline_correlation': .3, 'baseline_label_coefficients': [1.2, -.9, .6, 0, 0, 0, 0, 0],
        'input_reference_rows': 1000, 'abrupt_onset_one_based': 4001,
        'label_delays_batches': [0, 5, 20], 'proxy_concern_threshold': .95,
        'partitions_per_seed': {'train': 4000, 'correctness_calibration': 2000,
                               'historical_risk_reference': 2000, 'monitor_per_condition': 12000},
        'main_model': {'name': 'LogisticRegression', 'C': 1., 'max_iter': 500, 'class_threshold': .5},
    }
    for key, value in expected.items():
        if m[key] != value:
            raise ValueError(f'unsupported manifest amendment: {key}')
    assert m['correctness_calibration']['name'] == 'IsotonicRegression'
    assert m['correctness_calibration']['out_of_bounds'] == 'clip'
    assert m['representations']['pca_components'] == 3
    assert m['representations']['clusters'] == 3
    assert m['harm_target']['margin_numerator'] == 1
    assert m['harm_target']['margin_denominator'] == 20
    assert m['input_test'] == dict(name='two_sided_KS', method='auto', family_correction='Holm',
                                   alpha=.05, scope='per_batch', main_family_size=11)
    assert [c['id'] for c in m['conditions']] == [f'C{i}' for i in range(7)]
    assert m['conditions'][1]['mean_shift'] == 1.5 and m['conditions'][1]['feature'] == 7
    assert m['conditions'][2]['mean_shift'] == 1.5 and m['conditions'][2]['feature'] == 0
    for c in (m['conditions'][2], m['conditions'][4]):
        assert c['transition_start_one_based'] == 4001 and c['transition_end_one_based'] == 6000
    assert m['conditions'][5]['return_one_based'] == 8001
    assert m['conditions'][6]['post_correlation'] == .9
    assert m['impact_interval']['quantiles'] == [.025, .975]


def source_hashes():
    files = (list((ROOT/'src/drift_monitoring/phase3').glob('*.py')) +
             list((ROOT/'scripts').glob('*phase3*.py')) +
             list((ROOT/'tests/phase3').glob('*.py')))
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in sorted(files)}


def memory_record():
    try:
        import psutil
        info = psutil.Process().memory_info()
        return dict(rss_bytes=info.rss, peak_working_set_bytes=getattr(info, 'peak_wset', None))
    except ImportError:
        return dict(rss_bytes=None, peak_working_set_bytes=None)


def run(mode, out, seeds_override=None):
    started = time.perf_counter()
    manifest_path = ROOT/'docs/phase3/experiment_manifest.json'
    m = json.loads(manifest_path.read_text())
    validate_manifest(m)
    seeds = m['final_seeds'] if mode == 'final' else m['development_seeds']
    if seeds_override:
        if mode == 'final':
            raise ValueError('final seed selection is locked; use development mode')
        seeds = seeds_override
        if not set(seeds).issubset(m['development_seeds']):
            raise ValueError('development mode must use declared development seeds')
    # Never overwrite an old snapshot, successful or failed.
    out.mkdir(parents=True, exist_ok=False)
    for sub in ('predictions', 'evaluator_truth'):
        (out/sub).mkdir()
    lock = dict(protocol='IDM-Viz Phase 3 v1', mode=mode, seeds=seeds,
                started_at_utc=datetime.now(timezone.utc).isoformat(),
                original_manifest_sha256=sha(manifest_path), manifest=m,
                source_sha256=source_hashes(),
                original_documents_sha256={str(p.relative_to(ROOT)).replace('\\', '/'): sha(p)
                    for p in (ROOT/'docs/23i-2604_23i-2581DAV-phase2.docx',
                              ROOT/'docs/Phase 03 - Implementation.pdf')},
                thread_limit=1,
                label_clock='1-based batch completion; persist phase 0 then release phase 1; drain to 24+d',
                legacy_evidence_used=False, status='running')
    (out/'protocol_lock.json').write_text(json.dumps(lock, indent=2))
    versions = {p: importlib.metadata.version(p) for p in
                ('numpy', 'scipy', 'scikit-learn', 'pandas', 'matplotlib', 'pytest', 'threadpoolctl')}
    (out/'environment.json').write_text(json.dumps(dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), packages=versions, memory_start=memory_record()), indent=2))
    all_records, all_events, all_verifications, fit_records, warning_records = [], [], [], [], []
    try:
        with threadpool_limits(limits=1):
            for seed in seeds:
                seed_start = time.perf_counter()
                tr = historical_partition(seed, 0, 4000)
                ca = historical_partition(seed, 1, 2000)
                ref = historical_partition(seed, 2, 2000)
                a = fit_artifacts(seed, tr, ca, ref)
                np.savez_compressed(out/'predictions'/f'{seed}_historical_ids.npz', **a.fit_ids)
                fit_records.append(dict(seed=seed, fit_seconds=a.fit_seconds,
                                        reference_correct=a.reference_correct, reference_n=a.reference_n,
                                        atc_threshold=a.atc,
                                        logistic_iterations=int(a.model.n_iter_.max())))
                for condition in [c['id'] for c in m['conditions']]:
                    partition = monitoring_partition(seed, condition)
                    records, caches = [], []
                    for b in range(1, 25):
                        sl = slice((b-1)*500, b*500)
                        record, cache, messages = score_batch(a, FeatureBatch(
                            partition.x[sl], partition.ids[sl], seed, condition, b))
                        records.append(record)
                        caches.append(cache)
                        for message in messages:
                            warning_records.append(dict(seed=seed, condition=condition, batch=b, message=message))
                    # Persist predictions and feature/proxy evidence before releasing truth.
                    keys = caches[0].keys()
                    np.savez_compressed(out/'predictions'/f'{seed}_{condition}.npz',
                                        **{k: np.stack([c[k] for c in caches]) for k in keys})
                    pd.DataFrame(records).to_csv(out/'predictions'/f'{seed}_{condition}_online.csv', index=False)
                    # Evaluator-only storage is separate and explicitly named.
                    np.savez_compressed(out/'evaluator_truth'/f'{seed}_{condition}.npz',
                                        ids=partition.ids, labels=partition.y)
                    for delay in m['label_delays_batches']:
                        broker = LabelBroker(partition.ids, partition.y, 500, delay)
                        events, verified = replay(records, caches, broker)
                        all_events.extend(events)
                        all_verifications.extend(verified)
                    all_records.extend(records)
                print(f'seed {seed}: {time.perf_counter()-seed_start:.2f}s, 7 streams complete', flush=True)
        pd.DataFrame(all_records).to_csv(out/'online_batches.csv', index=False)
        pd.DataFrame(all_events).to_csv(out/'clock_events.csv', index=False)
        pd.DataFrame(all_verifications).to_csv(out/'released_verification.csv', index=False)
        pd.DataFrame(fit_records).to_csv(out/'fit_records.csv', index=False)
        pd.DataFrame(warning_records, columns=['seed', 'condition', 'batch', 'message']).to_csv(out/'warnings.csv', index=False)
        lock.update(status='complete', ended_at_utc=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter()-started, memory_end=memory_record(),
                    streams=len(seeds)*7, monitoring_rows=len(seeds)*7*12000,
                    online_batches=len(all_records), verification_rows=len(all_verifications),
                    ks_warning_count=len(warning_records))
    except BaseException as e:
        lock.update(status='failed', error=repr(e), elapsed_seconds=time.perf_counter()-started)
        raise
    finally:
        (out/'protocol_lock.json').write_text(json.dumps(lock, indent=2))
    print(json.dumps({k: lock[k] for k in ('status', 'elapsed_seconds', 'streams', 'online_batches')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['development', 'final'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', help='declared development seeds only')
    args = parser.parse_args()
    run(args.mode, args.out.resolve(), args.seeds)
