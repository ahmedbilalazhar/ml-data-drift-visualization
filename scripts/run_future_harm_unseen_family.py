"""Evaluate the existing future-harm predictor on a held-out synthetic law C7."""
from __future__ import annotations
import os
for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[n]='1'
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from drift_monitoring.phase3.core import historical_partition,fit_artifacts,score_batch,FeatureBatch,innovations,baseline_x,label_probability

DEV=list(range(950000,950030));EVAL=list(range(950100,950130));DELAYS=(0,5,20);ETA=.70;TRAIN_RUN=ROOT/'results/phase3/future_harm_forecast_20261116_v1'
FEATURES={'time_only':['time_norm','delay_norm'],
 'unlabeled_monitor':['time_norm','delay_norm','estimate','alarm_count','min_adjusted_p','input_alarm','concern'],
 'full_released':['time_norm','delay_norm','estimate','alarm_count','min_adjusted_p','input_alarm','concern','latest_accuracy','rolling_accuracy','released_harm_fraction','released_n','latest_age']}

def monitor_c7(seed,onset_batch=9):
 z,u=innovations(seed,3,12000);x=baseline_x(z);p=label_probability(x);onset_row=(int(onset_batch)-1)*500;p[onset_row:]=(1-ETA)*p[onset_row:]+ETA*.5
 y=(u<p).astype(np.int8);ids=seed*1_000_000+500_000+np.arange(len(y))
 return x,y,ids

def main(out:Path,onset_policy='fixed9'):
 if out.exists():raise FileExistsError(f'use a fresh output folder: {out}')
 lock=json.loads((TRAIN_RUN/'protocol_lock.json').read_text(encoding='utf-8'))
 if lock.get('protocol')!='future_harm_forecast_20261116_v1':raise ValueError('training forecast source lock mismatch')
 train_x=pd.read_csv(TRAIN_RUN/'predictions/online_features.csv');train_y=pd.read_csv(TRAIN_RUN/'evaluator_truth/future_targets.csv')
 train=train_x[train_x.seed.isin(DEV)].merge(train_y,on=['seed','condition','delay','batch'],validate='one_to_one')
 prevalence=float(train.future_harm.mean());models={}
 for name,cols in FEATURES.items():
  pipe=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=2026))
  xx=train[cols].replace([np.inf,-np.inf],np.nan).fillna(-1);pipe.fit(xx,train.future_harm);models[name]=pipe
 out.mkdir(parents=True);(out/'predictions').mkdir();(out/'evaluator_truth').mkdir();online=[];truth=[];probs=[];started=time.time()
 with threadpool_limits(limits=1):
  for si,seed in enumerate(EVAL,1):
   tr=historical_partition(seed,0,4000);ca=historical_partition(seed,1,2000);ref=historical_partition(seed,2,2000);art=fit_artifacts(seed,tr,ca,ref)
   onset_batch=9 if onset_policy=='fixed9' else (5,9,13,17)[(seed-EVAL[0])%4]
   x,y,ids=monitor_c7(seed,onset_batch);rec_batch=[];truth_batch=[]
   for b in range(1,25):
    sl=slice((b-1)*500,b*500);fb=FeatureBatch(x[sl],ids[sl],seed,'C7',b);rec,cache,_=score_batch(art,fb)
    correct=int((cache['prediction']==y[sl]).sum());row=dict(seed=seed,condition='C7',batch=b,ref_acc=art.reference_correct/art.reference_n,
      estimate=rec['cal_estimate'],concern=rec['cal_concern'],alarm_count=sum(bool(rec[f'x{i}_flag']) for i in range(8))+sum(bool(rec[f'pc{i}_adjusted_p']<=.05) for i in range(3)),
      min_adjusted_p=min([rec[f'x{i}_adjusted_p'] for i in range(8)]+[rec[f'pc{i}_adjusted_p'] for i in range(3)]),input_alarm=int(rec['input_alarm']),
      accuracy=correct/500,harm=int(correct/500<art.reference_correct/art.reference_n-.05))
    row['correct']=correct;rec_batch.append(row);truth_batch.append(dict(seed=seed,condition='C7',batch=b,correct=correct,accuracy=correct/500,harm=row['harm'],reference_accuracy=art.reference_correct/art.reference_n))
   g=pd.DataFrame(rec_batch)
   for delay in DELAYS:
    for t in range(1,20):
     released=g[g.batch+delay<t];future=g[(g.batch>t)&(g.batch<=t+5)]
     if len(future)!=5:raise AssertionError('incomplete future target window')
     row=dict(seed=seed,condition='C7',delay=delay,batch=t,time_norm=t/24,delay_norm=delay/20,
      estimate=g.loc[t-1,'estimate'],concern=g.loc[t-1,'concern'],alarm_count=g.loc[t-1,'alarm_count'],
      min_adjusted_p=g.loc[t-1,'min_adjusted_p'],input_alarm=g.loc[t-1,'input_alarm'])
     if len(released):row.update(latest_accuracy=float(released.iloc[-1].accuracy),rolling_accuracy=float(released.tail(5).accuracy.mean()),
       released_harm_fraction=float(released.tail(5).harm.mean()),released_n=int(len(released)*500),latest_age=int(t-released.iloc[-1].batch))
     else:row.update(latest_accuracy=np.nan,rolling_accuracy=np.nan,released_harm_fraction=np.nan,released_n=0,latest_age=25)
     online.append(row);future_acc=float(future.accuracy.mean());ref=float(g.ref_acc.iloc[0]);truth.append(dict(seed=seed,condition='C7',delay=delay,batch=t,
       future_accuracy=future_acc,reference_accuracy=ref,future_harm=int(future_acc<ref-.05),latest_released_harm=(int(released.iloc[-1].harm) if len(released) else np.nan)))
   if si%5==0:print(f'{si}/{len(EVAL)} unseen-family seeds; {time.time()-started:.0f}s',flush=True)
 X=pd.DataFrame(online);T=pd.DataFrame(truth);outputs=[]
 for name,cols in FEATURES.items():
  q=X[cols].replace([np.inf,-np.inf],np.nan).fillna(-1);outputs.append(X[['seed','condition','delay','batch']].assign(model=name,probability=models[name].predict_proba(q)[:,1]))
 base=[]
 teval=T.merge(X,on=['seed','condition','delay','batch'],validate='one_to_one')
 for name,p in [('development_prevalence',np.full(len(teval),prevalence)),('last_released_harm',teval.latest_released_harm.fillna(prevalence).to_numpy()),('current_concern',teval.concern.clip(0,1).to_numpy())]:
  base.append(teval[['seed','condition','delay','batch']].assign(model=name,probability=p))
 (out/'predictions/online_features.csv').write_text(X.to_csv(index=False),encoding='utf-8');T.to_csv(out/'evaluator_truth/future_targets.csv',index=False)
 pd.concat(outputs+base,ignore_index=True).to_csv(out/'predictions/heldout_probabilities.csv',index=False)
 feature_digest=hashlib.sha256((TRAIN_RUN/'predictions/online_features.csv').read_bytes()).hexdigest();target_digest=hashlib.sha256((TRAIN_RUN/'evaluator_truth/future_targets.csv').read_bytes()).hexdigest()
 protocol=dict(protocol='future_harm_unseen_family_20261116_v1',status='generated_evaluator_pending',source_protocol='future_harm_forecast_20261116_v1',training_seeds=DEV,evaluation_seeds=EVAL,
   training_conditions=[f'C{i}' for i in range(7)],evaluation_condition='C7_conditional_randomization_eta_0.70',delays=list(DELAYS),future_window='t+1..t+5',forecast_origins='t=1..19',
   onset_policy=onset_policy,onset_batches_by_seed={str(seed):(9 if onset_policy=='fixed9' else (5,9,13,17)[(seed-EVAL[0])%4]) for seed in EVAL},eta=ETA,marginal_invariant='X generated identically before and after onset; only P(y|x) flattened toward 0.5',
   predictors=FEATURES,availability='monitoring label b may enter only if b+delay < forecast t; current-clock outcomes appended after prediction',
   training_online_sha256=feature_digest,training_target_sha256=target_digest,online_output_has_no_outcomes=True,training_only_on_development_seeds=True,
   generator='core innovations/baseline_x/label_probability with post-onset p=(1-eta)*p_base+eta*0.5',runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 (out/'protocol_lock.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');print(json.dumps(protocol,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--onset-policy',choices=['fixed9','balanced_onsets'],default='fixed9');a=p.parse_args();main(a.out,a.onset_policy)
