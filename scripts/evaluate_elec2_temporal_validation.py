"""Join ELEC2 evaluator truth only after frozen online batches are persisted."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[1]

def ece10(q,correct):
 edges=np.linspace(0,1,11);total=len(q);v=0.
 for i in range(10):
  m=(q>=edges[i]) & ((q<edges[i+1]) if i<9 else (q<=edges[i+1]))
  if m.any():v+=m.sum()/total*abs(float(q[m].mean())-float(correct[m].mean()))
 return v

def main(run:Path):
 lockp=run/'protocol_lock.json';lock=json.loads(lockp.read_text(encoding='utf-8'))
 if lock.get('status')!='online_complete_evaluator_pending':raise ValueError('runner lock is not pending evaluation')
 online=pd.read_csv(run/'online_batches.csv')
 if set(online)&{'accuracy','correct','harm','y','label','macro_f1'}:raise ValueError('online file contains truth')
 truth=np.load(run/'evaluator_truth.npz');truth_ids=truth['row_id'];truth_y=truth['y']
 if len(np.unique(truth_ids))!=len(truth_ids):raise ValueError('duplicate truth IDs')
 ymap=dict(zip(truth_ids.tolist(),truth_y.tolist()));rows=[];class_rows=[]
 for b,g in online.groupby('batch',sort=True):
  a=np.load(run/f'batch_{int(b):03}.npz');ids=a['row_id'];pred=a['prediction'];q=a['calibrated_q']
  if len(g)!=1 or int(g.n.iloc[0])!=len(ids) or g.batch.nunique()!=1:raise ValueError('online batch metadata mismatch')
  if not set(ids.tolist()).issubset(ymap):raise ValueError('prediction row lacks evaluator label')
  y=np.asarray([ymap[int(i)] for i in ids],dtype=np.int8);correct=(pred==y).astype(np.int8)
  acc=float(accuracy_score(y,pred));pr,rc,f1,support=precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
  rows.append({'batch':int(b),'n':len(y),'accuracy':acc,'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
   'macro_f1':float(f1_score(y,pred,labels=[0,1],average='macro',zero_division=0)),
   'estimate':float(q.mean()),'estimate_signed_error':float(q.mean()-acc),'estimate_abs_error':abs(float(q.mean()-acc)),
   'brier_correctness':float(np.mean((q-correct)**2)),'ece10_correctness':ece10(q,correct),
   'interval_lower':float(g.interval_lower.iloc[0]),'interval_upper':float(g.interval_upper.iloc[0]),
   'interval_width':float(g.interval_width.iloc[0]),'interval_covered':bool(g.interval_lower.iloc[0]<=acc<=g.interval_upper.iloc[0]),
   'input_alarm':bool(g.input_alarm.iloc[0]),'alarm_count':int(g.alarm_count.iloc[0]),
   'class_0_support':int(np.sum(y==0)),'class_1_support':int(np.sum(y==1))})
  for j,label in enumerate([0,1]):class_rows.append({'batch':int(b),'class':int(label),'support':int(support[j]),
     'precision':float(pr[j]),'recall':float(rc[j]),'f1':float(f1[j])})
 if sum(r['n'] for r in rows)!=len(truth_y):raise ValueError('not all truth rows evaluated')
 batch=pd.DataFrame(rows);out=run/'evaluation';out.mkdir(exist_ok=False)
 batch.to_csv(out/'evaluator_batches.csv',index=False);pd.DataFrame(class_rows).to_csv(out/'per_class_metrics.csv',index=False)
 release=[]; n_batches=len(batch)
 for delay in (0,5,20):
  for b in batch.batch:
   release.append({'batch':int(b),'delay':delay,'available_at':int(b+delay),
       'released_after_prediction':True,'available_at_input_end':bool(b+delay<=n_batches),
       'n':int(batch.loc[batch.batch.eq(b),'n'].iloc[0])})
 rel=pd.DataFrame(release);rel.to_csv(out/'delay_release.csv',index=False)
 delay_summary=rel.groupby('delay').agg(total_batch_labels=('n','sum'),batches=('batch','count'),
   verified_by_input_end=('available_at_input_end','sum'),first_release=('available_at','min'),
   final_release=('available_at','max')).reset_index()
 delay_summary['unresolved_at_input_end']=n_batches-delay_summary.verified_by_input_end
 delay_summary['labels_after_drain']=delay_summary.total_batch_labels
 delay_summary.to_csv(out/'delay_summary.csv',index=False)
 overall={'status':'passed','dataset_rows':len(truth_y),'monitor_batches':n_batches,'final_partial_batch_n':int(batch.n.iloc[-1]),
  'row_weighted_accuracy':float(np.average(batch.accuracy,weights=batch.n)),
  'row_weighted_balanced_accuracy':float(np.average(batch.balanced_accuracy,weights=batch.n)),
  'row_weighted_macro_f1':float(np.average(batch.macro_f1,weights=batch.n)),
  'equal_batch_estimate_mae':float(batch.estimate_abs_error.mean()),'equal_batch_estimate_bias':float(batch.estimate_signed_error.mean()),
  'row_weighted_correctness_brier':float(np.average(batch.brier_correctness,weights=batch.n)),
  'equal_batch_ece10':float(batch.ece10_correctness.mean()),'interval_coverage_batches':int(batch.interval_covered.sum()),
  'input_alarm_batches':int(batch.input_alarm.sum()),'no_ground_truth_drift_onset':True,
  'delay_release_counts':delay_summary.to_dict('records'),
  'online_truth_columns_absent':True,
  'input_hashes':{'online_batches.csv':hashlib.sha256((run/'online_batches.csv').read_bytes()).hexdigest(),
                  'evaluator_truth.npz':hashlib.sha256((run/'evaluator_truth.npz').read_bytes()).hexdigest(),
                  'protocol_lock.json':hashlib.sha256(lockp.read_bytes()).hexdigest()}}
 (out/'summary.json').write_text(json.dumps(overall,indent=2),encoding='utf-8')
 lock['status']='complete';lock['evaluator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 lockp.write_text(json.dumps(lock,indent=2),encoding='utf-8')
 print(json.dumps(overall,indent=2))
 return overall

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
