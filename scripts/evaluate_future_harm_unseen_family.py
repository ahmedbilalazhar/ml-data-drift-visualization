"""Evaluate unseen-family future-harm forecasts without row-level inference."""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss
ROOT=Path(__file__).resolve().parents[1]
def score(y,p):
 y=np.asarray(y,dtype=int);p=np.clip(np.asarray(p,dtype=float),1e-8,1-1e-8)
 return dict(n=len(y),prevalence=float(y.mean()),roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
  average_precision=float(average_precision_score(y,p)) if y.sum()>0 else None,brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])))
def main(run:Path):
 lock=json.loads((run/'protocol_lock.json').read_text(encoding='utf-8'))
 if lock.get('status')!='generated_evaluator_pending':raise ValueError('unseen-family run is not pending evaluation')
 X=pd.read_csv(run/'predictions/online_features.csv');T=pd.read_csv(run/'evaluator_truth/future_targets.csv');P=pd.read_csv(run/'predictions/heldout_probabilities.csv')
 forbidden={'correct','accuracy','harm','future_harm','future_accuracy','reference_accuracy'}
 if forbidden & set(X.columns):raise ValueError('online feature artifact leaks current or future outcomes')
 if set(X.seed)&set(lock['training_seeds']):raise ValueError('evaluation rows overlap development seeds')
 if len(X)!=30*3*19 or len(T)!=len(X) or len(P)!=len(X)*6:raise ValueError('unexpected row counts')
 if T.batch.min()!=1 or T.batch.max()!=19 or not T.groupby(['seed','delay']).size().eq(19).all():raise ValueError('target origins/windows are not complete')
 if not np.array_equal(T.future_harm.to_numpy(),(T.future_accuracy<T.reference_accuracy-.05).astype(int).to_numpy()):raise ValueError('future-harm target definition mismatch')
 # Explicit boundary audit, including the no-monitor-label case at delay 20.
 for d in (0,5,20):
  g=X[X.delay.eq(d)]
  if d==0:
   if not g[g.batch.eq(1)].latest_accuracy.isna().all() or not g[g.batch.gt(1)].latest_accuracy.notna().all():raise ValueError('delay-0 strict release clock violated')
  elif d==5:
   if not g[g.batch.le(6)].latest_accuracy.isna().all() or not g[g.batch.ge(7)].latest_accuracy.notna().all():raise ValueError('delay-5 strict release clock violated')
  else:
   if not g.latest_accuracy.isna().all():raise ValueError('delay-20 label unexpectedly available by forecast horizon')
 d=P.merge(T,on=['seed','condition','delay','batch'],validate='many_to_one')
 if not d.probability.between(0,1).all():raise ValueError('probabilities outside [0,1]')
 rows=[]
 for model,g in d.groupby('model'):rows.append(dict(model=model,**score(g.future_harm,g.probability)))
 for (model,delay),g in d.groupby(['model','delay']):rows.append(dict(model=model,delay=int(delay),**score(g.future_harm,g.probability)))
 summ=pd.DataFrame(rows);summ.to_csv(run/'metrics_overall_by_delay.csv',index=False)
 br=d.assign(loss=(d.probability-d.future_harm)**2).groupby(['seed','model']).loss.mean().unstack();rng=np.random.default_rng(981744);boot=[]
 for a,b in [('time_only','unlabeled_monitor'),('unlabeled_monitor','full_released'),('time_only','full_released'),('full_released','development_prevalence')]:
  v=(br[a]-br[b]).to_numpy();draw=v[rng.integers(0,len(v),(2000,len(v)))].mean(axis=1)
  boot.append(dict(contrast=f'{a} minus {b} Brier',seed_n=len(v),mean=float(v.mean()),lower_95=float(np.quantile(draw,.025)),upper_95=float(np.quantile(draw,.975))))
 bdf=pd.DataFrame(boot);bdf.to_csv(run/'paired_seed_bootstrap.csv',index=False)
 result=dict(status='passed',seed_n=int(T.seed.nunique()),target_prevalence=float(T.future_harm.mean()),rows_per_model=int(len(T)),condition=lock['evaluation_condition'],
  metrics_by_model={m:score(g.future_harm,g.probability) for m,g in d.groupby('model')},paired_seed_bootstrap=boot,
  feature_outcome_separation=True,release_clock_strict=True,target_window_verified=True,seed_disjoint=True,
  scope='one manually designed held-out synthetic conditional-randomization law; not natural-domain or unseen-law generalization')
 lock['status']='evaluated';(run/'protocol_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
 (run/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run)
