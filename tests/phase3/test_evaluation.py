import numpy as np
from drift_monitoring.phase3.evaluation import flag_metrics, bootstrap_summary, regime
import pandas as pd


def test_undefined_metrics_and_counts():
    m=flag_metrics([False,True],[False,False],[.1,.3])
    assert np.isnan(m['precision']) and m['recall']==0 and m['harmful_unflagged_fraction']==1
    assert m['tp']==0 and m['fn']==1 and m['tn']==1 and m['fp']==0
    m=flag_metrics([False,False],[False,False],[.1,.3])
    assert np.isnan(m['recall']) and np.isnan(m['harm_auroc'])


def test_bootstrap_is_seed_based_and_deterministic():
    frame=pd.DataFrame(dict(seed=[1,2,3],condition=['C0']*3,mae=[.1,.2,.3]))
    a=bootstrap_summary(frame,['condition'],['mae'])
    b=bootstrap_summary(frame,['condition'],['mae'])
    pd.testing.assert_frame_equal(a,b)
    assert a.iloc[0].defined_seeds==3
    assert np.isclose(a.iloc[0]['mean'],.2)


def test_regime_endpoints():
    assert regime('C5',16)=='changed' and regime('C5',17)=='recovered'
    assert regime('C2',12)=='transition' and regime('C2',13)=='changed'
def test_runtime_is_not_included_in_detection_uncertainty_metrics():
    from scripts.evaluate_phase3_v2 import DETECTION_SUMMARY_METRICS

    assert "input_seconds" not in DETECTION_SUMMARY_METRICS
    assert set(DETECTION_SUMMARY_METRICS) == {
        "null_alerts_per_1000", "post_alert_fraction",
        "first_alarm_delay_batches", "missed_event",
    }
