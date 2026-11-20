"""Structural assertions for the saved, locked future-harm forecast study."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'results/phase3/future_harm_forecast_20261116_v1'

def test_future_harm_split_and_target_contract():
    lock=json.loads((RUN/'protocol_lock.json').read_text())
    assert set(lock['development_seeds']).isdisjoint(lock['evaluation_seeds'])
    target=pd.read_csv(RUN/'evaluator_truth/future_targets.csv')
    assert len(target)==60*7*3*19
    assert target.batch.min()==1 and target.batch.max()==19
    assert target.groupby(['seed','condition','delay']).size().eq(19).all()
    assert target.groupby(['seed','condition','delay']).future_accuracy.count().eq(19).all()

def test_saved_live_features_exclude_current_and_future_labels():
    live=pd.read_csv(RUN/'predictions/online_features.csv')
    assert not {'correct','accuracy','harm','future_harm','future_accuracy','reference_accuracy'} & set(live.columns)
    assert len(live)==60*7*3*19
    # With strict `release_clock < forecast_clock`, labels at delay d become
    # available only after their clock; same-clock d=0 evidence is not usable.
    g=live[(live.seed==950100)&(live.condition=='C0')]
    assert pd.isna(g[(g.delay==0)&(g.batch==1)].latest_accuracy.iloc[0])
    assert pd.notna(g[(g.delay==0)&(g.batch==2)].latest_accuracy.iloc[0])
    assert pd.isna(g[(g.delay==5)&(g.batch==6)].latest_accuracy.iloc[0])
    assert pd.notna(g[(g.delay==5)&(g.batch==7)].latest_accuracy.iloc[0])
    # Five-batch targets end at t=24, so no d=20 labels can arrive before any
    # registered forecast origin t<=19; batch 1 becomes usable only at t=22.
    assert g[g.delay==20].latest_accuracy.isna().all()
    assert 1+20 < 22 and not (1+20 < 21)

def test_forecast_probabilities_are_held_out_only_and_complete():
    lock=json.loads((RUN/'protocol_lock.json').read_text())
    p=pd.read_csv(RUN/'predictions/heldout_probabilities.csv')
    assert set(p.seed)==set(lock['evaluation_seeds'])
    assert set(p.model)=={'time_only','unlabeled_monitor','full_released'}
    assert len(p)==30*7*3*19*3
    assert p.groupby('model').probability.apply(lambda x: x.between(0,1).all()).all()
