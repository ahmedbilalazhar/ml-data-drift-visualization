"""Invariant checks for the held-out conditional-randomization experiment."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scripts.run_future_harm_unseen_family import EVAL,ETA,monitor_c7
from drift_monitoring.phase3.core import innovations,baseline_x,label_probability
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/phase3/future_harm_unseen_family_balanced_20261116_v1'

def test_unseen_family_changes_only_conditional_label_law():
    seed=EVAL[0];x,y,ids=monitor_c7(seed,9);z,u=innovations(seed,3,12000);expected_x=baseline_x(z)
    p=label_probability(expected_x);p[4000:]=(1-ETA)*p[4000:]+ETA*.5
    assert np.array_equal(x,expected_x)
    assert np.array_equal(y,(u<p).astype(np.int8))
    assert np.array_equal(ids,seed*1_000_000+500_000+np.arange(12000))

def test_balanced_onset_protocol_and_label_safe_outputs():
    lock=json.loads((RUN/'protocol_lock.json').read_text())
    schedule=lock['onset_batches_by_seed']
    assert set(schedule.values())=={5,9,13,17}
    assert sorted(pd.Series(list(schedule.values())).value_counts().to_list())==[7,7,8,8]
    X=pd.read_csv(RUN/'predictions/online_features.csv');T=pd.read_csv(RUN/'evaluator_truth/future_targets.csv')
    assert len(X)==len(T)==30*3*19
    assert not {'accuracy','correct','harm','future_harm','future_accuracy','reference_accuracy'} & set(X.columns)
    assert T.groupby(['seed','delay']).size().eq(19).all()
    assert X[X.delay.eq(20)].latest_accuracy.isna().all()

def test_evaluator_pass_and_all_reported_metrics_are_finite():
    v=json.loads((RUN/'validation.json').read_text());assert v['status']=='passed'
    assert v['feature_outcome_separation'] and v['release_clock_strict'] and v['seed_disjoint']
    for model,metrics in v['metrics_by_model'].items():
        assert metrics['n']==1710 and np.isfinite(metrics['brier'])
    assert v['paired_seed_bootstrap'][1]['lower_95']<0<v['paired_seed_bootstrap'][1]['upper_95']
