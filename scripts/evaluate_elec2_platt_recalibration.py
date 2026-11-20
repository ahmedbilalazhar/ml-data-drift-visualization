"""Evaluate Platt-style calibration beside frozen and cumulative isotonic caches."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from drift_monitoring.phase3.core import impact
from evaluate_elec2_temporal_validation import ece10
from run_elec2_temporal_validation import load_ele2
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def main(run:Path,platt:Path,isotonic:Path,out:Path):
 lock=json.loads((platt/'protocol_lock.json').read_text());
 if lock.get('status')!='complete':raise ValueError('Platt cache incomplete')
 iso_lock=json.loads((isotonic/'protocol_lock.json').read_text());
 if iso_lock.get('status')!='complete' or iso_lock['source_run_lock_sha256']!=__import__('hashlib').sha256((run/'protocol_lock.json').read_bytes()).hexdigest():raise ValueError('isotonic comparator is incomplete or from another source run')
 x,y,*_=load_ele2();scaler=StandardScaler().fit(x[:10800]);clf=LogisticRegression(C=1,max_iter=1000,random_state=2242026).fit(scaler.transform(x[:10800]),y[:10800])
 ref=clf.predict(scaler.transform(x[14462:18124]));ref_correct=int(np.sum(ref==y[14462:18124]));ref_n=18124-14462
 yfile=np.load(run/'evaluator_truth.npz');ymap=dict(zip(yfile['row_id'].astype(int).tolist(),yfile['y'].astype(int).tolist()))
 online=pd.read_csv(run/'online_batches.csv');rows=[]
 for d in (0,5,20):
  ia=pd.read_csv(isotonic/f'delay_{d}_fit_audit.csv');pa=pd.read_csv(platt/f'delay_{d}_fit_audit.csv')
  if (ia.max_available_at.fillna(0)>=ia.batch).any() or (pa.max_available_at.fillna(0)>=pa.batch).any():raise ValueError('same-clock or future label in calibrator fit')
  if not np.array_equal(ia.available_n.to_numpy(),pa.available_monitoring_labels.to_numpy()):raise ValueError('isotonic and Platt methods do not use the same released-label budget')
  for b in range(1,len(online)+1):
   src=np.load(run/f'batch_{b:03}.npz');ic=np.load(isotonic/f'delay_{d}_batch_{b:03}.npz');pc=np.load(platt/f'delay_{d}_batch_{b:03}.npz')
   if not (np.array_equal(src['row_id'],ic['row_id']) and np.array_equal(src['row_id'],pc['row_id']) and np.array_equal(src['prediction'],ic['prediction']) and np.array_equal(src['prediction'],pc['prediction'])):raise ValueError('fixed predictions/IDs changed between methods')
   yy=np.asarray([ymap[int(i)] for i in src['row_id']],dtype=np.int8);correct=(src['prediction']==yy).astype(int)
   for method,q in [('frozen_isotonic',src['calibrated_q']),('cumulative_isotonic',ic['q_recalibrated']),('cumulative_platt',pc['q_platt'])]:
    q=np.clip(np.asarray(q,dtype=float),0,1);inter,_=impact(q,ref_correct,ref_n);acc=float(np.mean(correct))
    rows.append(dict(delay=d,batch=b,method=method,n=len(yy),accuracy=acc,estimate=float(q.mean()),signed_error=float(q.mean()-acc),abs_error=float(abs(q.mean()-acc)),
      brier_correctness=float(np.mean((q-correct)**2)),ece10=ece10(q,correct),interval_lower=inter['lower'],interval_upper=inter['upper'],interval_width=inter['upper']-inter['lower'],interval_covered=bool(inter['lower']<=acc<=inter['upper']),
      available_monitoring_labels=int(pa.loc[pa.batch.eq(b),'available_monitoring_labels'].iloc[0])))
 per=pd.DataFrame(rows);out.mkdir(parents=True,exist_ok=False);per.to_csv(out/'per_batch_metrics.csv',index=False)
 agg=per.groupby(['delay','method']).agg(batches=('batch','nunique'),equal_batch_mae=('abs_error','mean'),equal_batch_bias=('signed_error','mean'),mean_correctness_brier=('brier_correctness','mean'),equal_batch_ece10=('ece10','mean'),interval_coverage=('interval_covered','sum'),mean_interval_width=('interval_width','mean')).reset_index()
 agg.to_csv(out/'summary.csv',index=False)
 result=dict(status='passed',batch_count=int(per.batch.max()),rows=len(per),same_predictions=True,matched_label_budgets=True,strict_release_guard=True,summary=agg.to_dict('records'),scope='one ELEC2 sequence; outcome-informed calibration-family sensitivity; no independent-stream inference')
 (out/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--platt',type=Path,required=True);p.add_argument('--isotonic',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.run,a.platt,a.isotonic,a.out)
