"""Evaluate released-label ELEC2 recalibration after its prediction caches exist."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from drift_monitoring.phase3.core import impact
from evaluate_elec2_temporal_validation import ece10

def main(run:Path,recal:Path):
 lockp=recal/'protocol_lock.json';lock=json.loads(lockp.read_text(encoding='utf-8'))
 if lock.get('status')!='recalibration_cache_complete_evaluation_pending':raise ValueError('recalibration cache is not pending evaluation')
 truth=np.load(run/'evaluator_truth.npz');ymap=dict(zip(truth['row_id'].tolist(),truth['y'].tolist()))
 monitor=pd.read_csv(run/'online_batches.csv');ref_correct=ref_n=None
 # Reconstruct the fixed reference target from the exact source split and frozen classifier.
 from run_elec2_temporal_validation import load_ele2
 from sklearn.linear_model import LogisticRegression
 from sklearn.preprocessing import StandardScaler
 x,y,*_=load_ele2();scaler=StandardScaler().fit(x[:10800]);model=LogisticRegression(C=1,max_iter=1000,random_state=2242026).fit(scaler.transform(x[:10800]),y[:10800])
 rp=model.predict(scaler.transform(x[14462:18124]));ref_correct=int(np.sum(rp==y[14462:18124]));ref_n=18124-14462
 rows=[]
 for delay in (0,5,20):
  audit=pd.read_csv(recal/f'delay_{delay}_fit_audit.csv')
  if np.any(audit.max_available_at.fillna(0).to_numpy()>=audit.batch.to_numpy()):raise ValueError('fit used current or future labels')
  for b in range(1,int(lock['batches'])+1):
   f=np.load(run/f'batch_{b:03}.npz');u=np.load(recal/f'delay_{delay}_batch_{b:03}.npz')
   if not np.array_equal(f['row_id'],u['row_id']) or not np.array_equal(f['prediction'],u['prediction']):raise ValueError('recalibration changed classifier predictions or IDs')
   yb=np.asarray([ymap[int(i)] for i in f['row_id']],dtype=np.int8);pred=f['prediction'];correct=(pred==yb).astype(np.int8)
   rec=audit[audit.batch.eq(b)].iloc[0]
   for method,q in [('frozen',f['calibrated_q']),('released_label_isotonic',u['q_recalibrated'])]:
    q=np.asarray(q,dtype=float);pm,_=impact(q,ref_correct,ref_n);acc=float(accuracy_score(yb,pred))
    rows.append({'delay':delay,'method':method,'batch':b,'n':len(yb),'accuracy':acc,
      'estimate':float(q.mean()),'signed_error':float(q.mean()-acc),'abs_error':abs(float(q.mean()-acc)),
      'brier_correctness':float(np.mean((q-correct)**2)),'ece10':ece10(q,correct),
      'interval_lower':pm['lower'],'interval_upper':pm['upper'],'interval_width':pm['upper']-pm['lower'],
      'interval_covered':bool(pm['lower']<=acc<=pm['upper']),'concern_score':pm['concern'],
      'available_monitoring_labels':int(rec.available_n),'max_available_at':None if pd.isna(rec.max_available_at) else int(rec.max_available_at),
      'input_alarm':bool(monitor.loc[monitor.batch.eq(b),'input_alarm'].iloc[0])})
 per=pd.DataFrame(rows);out=recal/'evaluation';out.mkdir(exist_ok=False)
 per.to_csv(out/'per_batch_metrics.csv',index=False)
 agg=per.groupby(['delay','method']).agg(batches=('batch','nunique'),
  equal_batch_mae=('abs_error','mean'),equal_batch_bias=('signed_error','mean'),
  row_weighted_brier=('brier_correctness','mean'),equal_batch_ece10=('ece10','mean'),
  interval_coverage=('interval_covered','sum'),mean_interval_width=('interval_width','mean')).reset_index()
 agg.to_csv(out/'summary.csv',index=False)
 full={ 'status':'passed','rows':len(truth['y']),'batches':int(lock['batches']),
  'label_time_guard_passed':True,'same_classifier_predictions_across_methods':True,
  'delay_summary':agg.to_dict('records'),
  'scope':'one stream sequence; exploratory recalibration tradeoff; no independent-sequence interval'}
 (out/'validation.json').write_text(json.dumps(full,indent=2),encoding='utf-8')
 lock['status']='complete';lock['evaluator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 lockp.write_text(json.dumps(lock,indent=2),encoding='utf-8')
 print(json.dumps(full,indent=2));return full

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--recalibration',type=Path,required=True);a=p.parse_args();main(a.run,a.recalibration)
