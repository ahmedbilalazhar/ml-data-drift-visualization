"""Scientific-invariant checks for matched-budget Platt recalibration."""
import json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
PLATT=ROOT/'results/phase3/elec2_platt_recalibration_20261116_v1'
ISO=ROOT/'results/phase3/elec2_recalibration_20261116_v1'
EVAL=ROOT/'results/phase3/elec2_calibration_family_20261116_v1'

def test_cumulative_calibrators_have_matched_strictly_past_label_budgets():
    for delay in (0,5,20):
        p=pd.read_csv(PLATT/f'delay_{delay}_fit_audit.csv')
        i=pd.read_csv(ISO/f'delay_{delay}_fit_audit.csv')
        assert len(p)==len(i)==55
        assert np.array_equal(p.available_monitoring_labels,i.available_n)
        assert (p.max_available_at.fillna(0)<p.batch).all()
        assert (i.max_available_at.fillna(0)<i.batch).all()
        if delay==0:
            assert p.loc[p.batch.eq(1),'available_monitoring_labels'].iloc[0]==0
            assert p.loc[p.batch.eq(2),'available_monitoring_labels'].iloc[0]==500
        if delay==20:
            assert p.loc[p.batch.le(21),'available_monitoring_labels'].eq(0).all()
            assert p.loc[p.batch.eq(22),'available_monitoring_labels'].iloc[0]==500

def test_evaluator_confirms_same_predictions_and_three_method_comparison():
    result=json.loads((EVAL/'validation.json').read_text())
    assert result['status']=='passed'
    assert result['same_predictions'] and result['matched_label_budgets'] and result['strict_release_guard']
    per=pd.read_csv(EVAL/'per_batch_metrics.csv')
    assert len(per)==3*3*55
    assert set(per.method)=={'frozen_isotonic','cumulative_isotonic','cumulative_platt'}
    assert per.groupby(['delay','method']).size().eq(55).all()
    assert per.estimate.between(0,1).all()
