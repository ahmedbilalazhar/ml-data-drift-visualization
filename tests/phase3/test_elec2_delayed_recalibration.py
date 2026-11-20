import numpy as np

from scripts.run_elec2_delayed_recalibration import recalibrate_delay


def _inputs():
    cal_conf=np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9])
    cal_correct=np.array([0,0,0,0,1,1,1,1,1],dtype=float)
    conf={b:np.array([.2,.4,.6,.8]) for b in range(1,4)}
    pred={b:np.array([0,0,1,1]) for b in range(1,4)}
    truth={b:np.array([0,1,1,1]) for b in range(1,4)}
    return cal_conf,cal_correct,conf,pred,truth


def test_same_clock_labels_do_not_change_that_batch_calibration():
    cc,cy,conf,pred,truth=_inputs()
    a,_=recalibrate_delay(cc,cy,conf,pred,truth,0)
    changed={k:v.copy() for k,v in truth.items()};changed[3]=1-changed[3]
    b,_=recalibrate_delay(cc,cy,conf,pred,changed,0)
    for batch in (1,2,3):np.testing.assert_array_equal(a[batch],b[batch])


def test_delay20_has_no_released_monitor_labels_with_only_three_batches():
    cc,cy,conf,pred,truth=_inputs()
    q,audit=recalibrate_delay(cc,cy,conf,pred,truth,20)
    assert len(q)==3
    assert audit.available_n.tolist()==[0,0,0]
    assert audit.max_available_at.isna().all()
