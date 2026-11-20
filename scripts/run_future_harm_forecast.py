"""Locked, seed-held-out forecast of five-batch future harm."""
from __future__ import annotations
import os
for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[n]='1'
from pathlib import Path
import sys,json,time
import argparse
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from drift_monitoring.phase3.core import historical_partition, monitoring_partition, fit_artifacts, score_batch, FeatureBatch

DEV=list(range(950000,950030)); EVAL=list(range(950100,950130)); CONDITIONS=[f'C{i}' for i in range(7)]; DELAYS=[0,5,20]
OUT=ROOT/'results/phase3/future_harm_forecast_20261116_v1'
def stream(seed,condition):
    tr=historical_partition(seed,0,4000); ca=historical_partition(seed,1,2000); ref=historical_partition(seed,2,2000)
    art=fit_artifacts(seed,tr,ca,ref); part=monitoring_partition(seed,condition,12000)
    online=[]; truth=[]
    for b in range(1,25):
        sl=slice((b-1)*500,b*500); fb=FeatureBatch(part.x[sl],part.ids[sl],seed,condition,b)
        rec,cache,_=score_batch(art,fb); correct=int((cache['prediction']==part.y[sl]).sum())
        online.append(dict(seed=seed,condition=condition,batch=b,ref_acc=art.reference_correct/art.reference_n,
          estimate=rec['cal_estimate'],concern=rec['cal_concern'],alarm_count=sum(bool(rec[f'x{i}_flag']) for i in range(8))+sum(bool(rec[f'pc{i}_adjusted_p']<=.05) for i in range(3)),
          min_adjusted_p=min([rec[f'x{i}_adjusted_p'] for i in range(8)]+[rec[f'pc{i}_adjusted_p'] for i in range(3)]),
          input_alarm=int(any(rec[f'x{i}_flag'] for i in range(8)) or any(rec[f'pc{i}_adjusted_p']<=.05 for i in range(3))),
          correct=correct,accuracy=correct/500,harm=int(correct/500 < art.reference_correct/art.reference_n-.05)))
    return online

def main(out=OUT):
    out=Path(out)
    out.mkdir(parents=True,exist_ok=False); (out/'predictions').mkdir(); (out/'evaluator_truth').mkdir()
    started=time.time(); allrows=[]
    with threadpool_limits(limits=1):
      for si,seed in enumerate(DEV+EVAL,1):
        for condition in CONDITIONS: allrows.extend(stream(seed,condition))
        if si%5==0: print(f'{si}/{len(DEV)+len(EVAL)} seeds complete; {time.time()-started:.0f}s',flush=True)
    df=pd.DataFrame(allrows); online=[]; truth=[]
    # Keep the saved live-side artifact label-free; outcomes stay in evaluator truth.
    safe_online=df.drop(columns=['correct','accuracy','harm'])
    safe_online.to_csv(out/'predictions/stream_features.csv',index=False)
    for seed,condition in df[['seed','condition']].drop_duplicates().itertuples(index=False):
      g=df[(df.seed==seed)&(df.condition==condition)].sort_values('batch').reset_index(drop=True)
      for delay in DELAYS:
       for t in range(1,20):
        released=g[g.batch+delay < t] # same-clock events occur after forecast; strict availability
        future=g[(g.batch>t)&(g.batch<=t+5)]
        if len(future)!=5: raise AssertionError('future target window incomplete')
        row=dict(seed=seed,condition=condition,delay=delay,batch=t,time_norm=t/24,delay_norm=delay/20,
          estimate=g.loc[t-1,'estimate'],alarm_count=g.loc[t-1,'alarm_count'],min_adjusted_p=g.loc[t-1,'min_adjusted_p'],
          input_alarm=g.loc[t-1,'input_alarm'],concern=g.loc[t-1,'concern'])
        if len(released):
          row.update(latest_accuracy=released.iloc[-1].accuracy,rolling_accuracy=released.tail(5).accuracy.mean(),
             released_harm_fraction=released.tail(5).harm.mean(),released_n=len(released)*500,
             latest_age=t-int(released.iloc[-1].batch))
        else: row.update(latest_accuracy=np.nan,rolling_accuracy=np.nan,released_harm_fraction=np.nan,released_n=0,latest_age=25)
        online.append(row); truth.append(dict(seed=seed,condition=condition,delay=delay,batch=t,
          future_accuracy=float(future.accuracy.mean()),reference_accuracy=float(g.ref_acc.iloc[0]),
          future_harm=int(future.accuracy.mean() < g.ref_acc.iloc[0]-.05),
          latest_released_harm=(int(released.iloc[-1].harm) if len(released) else np.nan)))
    pred=pd.DataFrame(online); tgt=pd.DataFrame(truth)
    train=pred.seed.isin(DEV); features={
      'time_only':['time_norm','delay_norm'],
      'unlabeled_monitor':['time_norm','delay_norm','estimate','alarm_count','min_adjusted_p','input_alarm','concern'],
      'full_released':['time_norm','delay_norm','estimate','alarm_count','min_adjusted_p','input_alarm','concern','latest_accuracy','rolling_accuracy','released_harm_fraction','released_n','latest_age']}
    outputs=[]
    for name,cols in features.items():
      X=pred[cols].replace([np.inf,-np.inf],np.nan).fillna(-1); y=tgt.future_harm
      model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=2026))
      model.fit(X.loc[train],y.loc[train]); ix=~train
      outputs.append(pred.loc[ix,['seed','condition','delay','batch']].assign(model=name,probability=model.predict_proba(X.loc[ix])[:,1]))
    pred.to_csv(out/'predictions/online_features.csv',index=False); tgt.to_csv(out/'evaluator_truth/future_targets.csv',index=False)
    pd.concat(outputs,ignore_index=True).to_csv(out/'predictions/heldout_probabilities.csv',index=False)
    (out/'protocol_lock.json').write_text(json.dumps(dict(protocol='future_harm_forecast_20261116_v1',development_seeds=DEV,evaluation_seeds=EVAL,conditions=CONDITIONS,delays=DELAYS,feature_sets=features,future_window='t+1..t+5',availability='batch+delay < forecast batch; same-clock release occurs after forecast',outcome='future five-batch mean accuracy < historical reference accuracy - 0.05',seed_count=len(DEV)+len(EVAL)),indent=2))
    print(f'Completed in {time.time()-started:.1f}s: {out}')
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=OUT); main(parser.parse_args().out)
