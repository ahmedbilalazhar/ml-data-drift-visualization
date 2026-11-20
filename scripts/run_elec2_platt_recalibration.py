"""Matched-budget Platt calibration replay on the fixed ELEC2 predictions."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.special import logit
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from run_elec2_temporal_validation import load_ele2

def run(source_run:Path,out:Path):
 if out.exists():raise FileExistsError(f'use fresh directory: {out}')
 lock=json.loads((source_run/'protocol_lock.json').read_text(encoding='utf-8'))
 if lock.get('dataset')!='OpenML electricity 151 v1 / ELEC2' or lock.get('status')!='complete':raise ValueError('incomplete or wrong source run')
 x,y,*_=load_ele2()
 if hashlib.sha256(Path(ROOT/'data/external/openml_151/electricity.arff').read_bytes()).hexdigest()!=lock['source_sha256']:raise ValueError('source hash mismatch')
 train_end,cal_end=10800,14462
 from sklearn.preprocessing import StandardScaler
 scale=StandardScaler().fit(x[:train_end]);model=LogisticRegression(C=1,max_iter=1000,random_state=2242026).fit(scale.transform(x[:train_end]),y[:train_end])
 cp=model.predict_proba(scale.transform(x[train_end:cal_end]));pred=model.classes_[cp.argmax(1)]
 cal_conf=np.maximum(cp[:,0],cp[:,1]);cal_correct=(pred==y[train_end:cal_end]).astype(int)
 truth=np.load(source_run/'evaluator_truth.npz');truth_map=dict(zip(truth['row_id'].astype(int).tolist(),truth['y'].astype(int).tolist()))
 batches=int(lock['monitor_batches']); conf={};preds={};actual={}
 for b in range(1,batches+1):
  z=np.load(source_run/f'batch_{b:03}.npz');conf[b]=z['confidence'];preds[b]=z['prediction'];actual[b]=np.asarray([truth_map[int(i)] for i in z['row_id']],dtype=np.int8)
 out.mkdir(parents=True)
 for delay in (0,5,20):
  released_conf=[];released_corr=[];release_at=[];aud=[]
  for t in range(1,batches+1):
   cx=np.concatenate([cal_conf,*released_conf]);cy=np.concatenate([cal_correct,*released_corr])
   if len(np.unique(cy))!=2:raise ValueError('correctness calibration requires both outcomes')
   model_cal=LogisticRegression(C=1.0,max_iter=1000,solver='lbfgs')
   model_cal.fit(logit(np.clip(cx,.500001,.999999)).reshape(-1,1),cy)
   q=model_cal.predict_proba(logit(np.clip(conf[t],.500001,.999999)).reshape(-1,1))[:,1]
   z=np.load(source_run/f'batch_{t:03}.npz')
   np.savez_compressed(out/f'delay_{delay}_batch_{t:03}.npz',row_id=z['row_id'],prediction=preds[t],confidence=conf[t],q_platt=q)
   aud.append(dict(delay=delay,batch=t,calibration_rows=len(cx),available_monitoring_labels=int(sum(map(len,released_corr))),max_available_at=max(release_at) if release_at else None,
     same_clock_released_after_forecast=True,calibrator='logistic_regression_on_logit_confidence',C=1.0))
   due=t-delay
   if 1<=due<=batches:
    released_conf.append(conf[due].copy());released_corr.append((preds[due]==actual[due]).astype(int));release_at.append(due+delay)
  pd.DataFrame(aud).to_csv(out/f'delay_{delay}_fit_audit.csv',index=False)
 if any((pd.read_csv(out/f'delay_{d}_fit_audit.csv').max_available_at.fillna(0)>=pd.read_csv(out/f'delay_{d}_fit_audit.csv').batch).any() for d in (0,5,20)):raise AssertionError('future or same-clock labels used')
 meta=dict(status='complete',source_run=str(source_run),delays=[0,5,20],method='cumulative Platt-style logistic calibration of correctness on logit predicted-class confidence',
  paired_training_budget='same initial correctness-calibration set plus exactly the released monitoring batches used by the cumulative isotonic comparator',
  availability_rule='monitor batch b usable at forecast t only when b+delay < t; same-clock release is processed after current forecast',
  classifier_frozen=True,source_predictions_unchanged=True,source_sha256=lock['source_sha256'],runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (out/'protocol_lock.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');print(json.dumps(meta,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.run,a.out)
