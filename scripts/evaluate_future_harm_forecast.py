"""Evaluation-only metrics for the locked future-harm forecast extension."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
ROOT=Path(__file__).resolve().parents[1]
def scores(y,p):
 y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,dtype=float),1e-8,1-1e-8)
 return dict(n=len(y),prevalence=float(y.mean()),roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
  average_precision=float(average_precision_score(y,p)) if int(y.sum())>0 else np.nan,
  brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])))
def main(path):
 path=Path(path); lock=json.loads((path/'protocol_lock.json').read_text())
 t=pd.read_csv(path/'evaluator_truth/future_targets.csv'); p=pd.read_csv(path/'predictions/heldout_probabilities.csv')
 d=p.merge(t,on=['seed','condition','delay','batch'],validate='many_to_one')
 if len(d)!=30*7*3*19*3: raise ValueError(f'unexpected prediction row count {len(d)}')
 # Registered heuristic comparators. Release history is evaluator-side here,
 # and only the latest label that was strictly available is used.
 f=pd.read_csv(path/'predictions/online_features.csv')
 f=f[f.seed.isin(lock['evaluation_seeds'])].merge(t,on=['seed','condition','delay','batch'],validate='one_to_one')
 prevalence=float(t[t.seed.isin(lock['development_seeds'])].future_harm.mean())
 f['probability_prevalence']=prevalence
 f['probability_last_released_harm']=f.latest_released_harm.fillna(prevalence)
 f['probability_current_concern']=f.concern.clip(0,1)
 baseline_rows=[]
 for name,col in [('development_prevalence','probability_prevalence'),('last_released_harm','probability_last_released_harm'),('current_concern','probability_current_concern')]:
  baseline_rows.append(f.assign(model=name,probability=f[col])[['seed','condition','delay','batch','model','probability','future_harm']])
 baselines=pd.concat(baseline_rows,ignore_index=True); baselines.to_csv(path/'predictions/heuristic_baseline_probabilities.csv',index=False)
 d=pd.concat([d,baselines],ignore_index=True)
 rows=[]
 for (model,delay),g in d.groupby(['model','delay']): rows.append(dict(model=model,delay=int(delay),**scores(g.future_harm,g.probability)))
 for (model,condition),g in d.groupby(['model','condition']): rows.append(dict(model=model,condition=condition,**scores(g.future_harm,g.probability)))
 summ=pd.DataFrame(rows); summ.to_csv(path/'metrics_by_delay_condition.csv',index=False)
 # Seed is the resampling unit; score differences aggregate all paired rows per seed.
 w=d.pivot_table(index='seed',columns='model',values='probability',aggfunc='size')
 brier=d.assign(loss=(d.probability-d.future_harm)**2).groupby(['seed','model']).loss.mean().unstack()
 rng=np.random.default_rng(817260); boot=[]
 for a,b in [('time_only','unlabeled_monitor'),('unlabeled_monitor','full_released'),('time_only','full_released')]:
  delta=(brier[a]-brier[b]).to_numpy(); means=np.mean(delta[rng.integers(0,len(delta),(2000,len(delta)))],axis=1)
  boot.append(dict(contrast=f'{a} minus {b} Brier',seed_count=len(delta),mean=float(delta.mean()),lower_95=float(np.quantile(means,.025)),upper_95=float(np.quantile(means,.975))))
 bootdf=pd.DataFrame(boot); bootdf.to_csv(path/'paired_seed_bootstrap.csv',index=False)
 # 0.5 threshold metrics are descriptive only; no threshold selection.
 d['predicted_harm_05']=d.probability>=.5
 th=[]
 for model,g in d.groupby('model'):
  tp=int(((g.predicted_harm_05)&(g.future_harm==1)).sum())
  th.append(dict(model=model,accuracy=float((g.predicted_harm_05==g.future_harm).mean()),
    precision=float(tp/max(1,int(g.predicted_harm_05.sum()))),recall=float(tp/max(1,int((g.future_harm==1).sum())))))
 threshold=pd.DataFrame(th)
 threshold.to_csv(path/'descriptive_threshold_metrics.csv',index=False)
 report=dict(protocol=lock['protocol'],heldout_seeds=lock['evaluation_seeds'],overall={m:scores(g.future_harm,g.probability) for m,g in d.groupby('model')},
  prevalence_constant=float(t[t.seed.isin(lock['development_seeds'])].future_harm.mean()),bootstrap=bootdf.to_dict('records'),by_delay=[r for r in rows if 'delay' in r])
 (path/'evaluation_summary.json').write_text(json.dumps(report,indent=2,allow_nan=True)); print(json.dumps(report,indent=2,allow_nan=True))
if __name__=='__main__': main(sys.argv[1] if len(sys.argv)>1 else str(ROOT/'results/phase3/future_harm_forecast_20261116_v1'))
