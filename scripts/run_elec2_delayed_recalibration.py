"""Replay isotonic correctness recalibration using only labels due before a forecast."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'scripts'))
from run_elec2_temporal_validation import load_ele2

def recalibrate_delay(cal_conf,cal_correct,conf_by_batch,pred_by_batch,truth_by_batch,delay):
 """Return pre-release q caches; release same-clock outcomes only after each score."""
 if delay not in (0,5,20):raise ValueError('delay not in locked protocol')
 released_conf=[];released_correct=[];released_from=[];outputs={};audit=[]
 n_batches=len(conf_by_batch)
 for batch in range(1,n_batches+1):
  train_conf=np.concatenate([np.asarray(cal_conf),*released_conf])
  train_correct=np.concatenate([np.asarray(cal_correct),*released_correct])
  calibrator=IsotonicRegression(out_of_bounds='clip').fit(train_conf,train_correct)
  conf=np.asarray(conf_by_batch[batch],dtype=float);pred=np.asarray(pred_by_batch[batch]);
  outputs[batch]=np.clip(calibrator.predict(conf),0,1)
  audit.append({'batch':batch,'delay':delay,'available_n':int(sum(len(v) for v in released_conf)),
    'calibration_n':len(train_conf),'max_available_at':max(released_from) if released_from else None})
  due=batch-delay
  if 1<=due<=n_batches:
   # The due labels arrive after this batch's prediction has been written.
   y=np.asarray(truth_by_batch[due]);source_conf=np.asarray(conf_by_batch[due]);source_pred=np.asarray(pred_by_batch[due])
   if len(y)!=len(source_conf) or len(y)!=len(source_pred):raise ValueError('release cohort misalignment')
   released_conf.append(source_conf.copy());released_correct.append((source_pred==y).astype(float))
   released_from.append(due+delay)
 return outputs,pd.DataFrame(audit)

def run(source_run:Path,out:Path):
 if out.exists():raise FileExistsError(f'choose a fresh output directory: {out}')
 lock=json.loads((source_run/'protocol_lock.json').read_text(encoding='utf-8'))
 if lock.get('dataset')!='OpenML electricity 151 v1 / ELEC2' or lock.get('status')!='complete':raise ValueError('source run is not complete ELEC2 evidence')
 x,y,date,period,ids,digest=load_ele2()
 if digest!=lock['source_sha256']:raise ValueError('source hash mismatch')
 train_end,cal_end,ref_end=10800,14462,18124
 scaler=StandardScaler().fit(x[:train_end]);model=LogisticRegression(C=1.0,max_iter=1000,random_state=2242026).fit(scaler.transform(x[:train_end]),y[:train_end])
 calp=model.predict_proba(scaler.transform(x[train_end:cal_end]));calpred=model.classes_[calp.argmax(axis=1)]
 calconf=calp.max(axis=1);calcorrect=(calpred==y[train_end:cal_end]).astype(float)
 n_batches=int(lock['monitor_batches']);conf_by={};pred_by={};truth_by={}
 # Offline evaluator truth is loaded here but used only after the matching release clock.
 truth_cache=np.load(source_run/'evaluator_truth.npz');truth_map=dict(zip(truth_cache['row_id'].tolist(),truth_cache['y'].tolist()))
 for b in range(1,n_batches+1):
  a=np.load(source_run/f'batch_{b:03}.npz');conf_by[b]=a['confidence'];pred_by[b]=a['prediction']
  truth_by[b]=np.asarray([truth_map[int(i)] for i in a['row_id']],dtype=np.int8)
 out.mkdir(parents=True)
 all_rows=[]
 for delay in (0,5,20):
  q_by,audit=recalibrate_delay(calconf,calcorrect,conf_by,pred_by,truth_by,delay)
  audit.insert(0,'method','released_label_isotonic');audit.to_csv(out/f'delay_{delay}_fit_audit.csv',index=False)
  for b in range(1,n_batches+1):
   source=np.load(source_run/f'batch_{b:03}.npz');q=q_by[b]
   np.savez_compressed(out/f'delay_{delay}_batch_{b:03}.npz',row_id=source['row_id'],prediction=source['prediction'],
                       confidence=source['confidence'],q_frozen=source['calibrated_q'],q_recalibrated=q)
   rec=audit[audit.batch.eq(b)].iloc[0]
   all_rows.append({'delay':delay,'batch':b,'n':len(q),'available_monitoring_labels':int(rec.available_n),
     'calibration_rows':int(rec.calibration_n),'max_available_at':None if pd.isna(rec.max_available_at) else int(rec.max_available_at),
     'forecast_persisted_before_same_clock_release':True,'frozen_q_mean':float(source['calibrated_q'].mean()),
     'recalibrated_q_mean':float(q.mean())})
 online=pd.DataFrame(all_rows)
 if np.any(online.max_available_at.fillna(0).to_numpy()>=online.batch.to_numpy()):raise AssertionError('current/future label entered fit')
 online.to_csv(out/'online_recalibration.csv',index=False)
 meta={'status':'recalibration_cache_complete_evaluation_pending','source_run':str(source_run),
  'source_run_lock_sha256':hashlib.sha256((source_run/'protocol_lock.json').read_bytes()).hexdigest(),
  'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'delays':[0,5,20],
  'batches':n_batches,'prediction_order':'score and persist each batch before adding labels with source_batch+delay == current clock',
  'feature_model_frozen':True,'only_correctness_calibrator_updated':True,
  'strict_label_time_assertion_passed':True,'online_metrics_contain_no_truth':True}
 (out/'protocol_lock.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 print(json.dumps(meta,indent=2));return meta

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();run(a.run,a.out)
