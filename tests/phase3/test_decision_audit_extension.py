import numpy as np
from scripts.strengthen_phase3_decisions import audit_schedule

def test_quota_and_no_future_score_ranking():
    for budget in (2,4):
        for delay in (0,5,20):
            a=np.linspace(0,1,24);b=a.copy();b[12:]=1e6
            left=audit_schedule(a,.5,budget,delay);right=audit_schedule(b,.5,budget,delay)
            assert sum(r['selected'] for r in left)==budget
            assert left[:12]==right[:12]

def test_current_clock_labels_are_unavailable_even_at_zero_delay():
    calls=[]
    def labels(batch):calls.append(batch);return .1
    rows=audit_schedule(np.zeros(24),.05,2,0,labels,.8)
    assert [r['batch'] for r in rows if r['selected']]==[12,13]
    assert calls[0]==12
    assert rows[11]['prior_released_selected_batches']=='[]'
    assert rows[12]['prior_released_selected_batches']=='[12]'

def test_long_delay_has_no_feedback_during_sequence():
    def unavailable(batch):raise AssertionError('unavailable label accessed')
    rows=audit_schedule(np.zeros(24),.05,4,20,unavailable,.8)
    assert sum(r['selected'] for r in rows)==4
