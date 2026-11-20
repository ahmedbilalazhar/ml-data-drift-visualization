"""Score cached UCI predictions against evaluator-only labels."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_uci224_day2 import load_cohorts

LABELS=np.arange(1,7)
KEY=['history_window','split_seed','model','batch','row_index']

def main(run:Path):
 lockp=run/'protocol_lock.json'; lock=json.loads(lockp.read_text(encoding='utf-8'))
 if lock.get('status')!='online_complete_evaluator_pending': raise ValueError('runner lock is not in pending evaluator state')
 archive=ROOT/'data/external/uci224_source/uci224_curl.zip'
 if hashlib.sha256(archive.read_bytes()).hexdigest()!=lock['source_sha256']: raise ValueError('source hash mismatch')
 online=pd.read_csv(run/'online_predictions.csv.gz'); cohorts=load_cohorts()
 if set(online.columns)&{'correct','accuracy','harm','label','y'}: raise ValueError('online prediction cache leaks truth')
 if online.duplicated(KEY).any(): raise ValueError('duplicate online rows')
 batch_truth={b:y for b,(x,y) in cohorts.items()}
 if not online.groupby(['history_window','split_seed','model']).size().eq(
      online.groupby(['history_window','split_seed','model']).size()).all(): raise AssertionError
 predictions=[]; class_rows=[]; fit=pd.read_csv(run/'fit_records.csv')
 ref_lookup={(r.history_window,int(r.split_seed),r.model):float(r.reference_accuracy) for r in fit.itertuples()}
 expected=0
 for (window,seed,model),g in online.groupby(['history_window','split_seed','model'],sort=True):
  batches=range(2,11) if window=='batch1' else range(3,11)
  expected+=len(batches)
  if set(g.batch.unique())!=set(batches): raise ValueError(f'missing batch for {window}/{seed}/{model}')
  for batch,h in g.groupby('batch',sort=True):
   h=h.sort_values('row_index'); y=batch_truth[int(batch)]
   if len(h)!=len(y) or not np.array_equal(h.row_index.to_numpy(),np.arange(len(y))): raise ValueError('row alignment mismatch')
   pred=h.prediction.to_numpy(); conf=h.confidence.to_numpy(); q=h.calibrated_q.to_numpy(); correct=(pred==y).astype(int)
   acc=float(accuracy_score(y,pred)); raw=float(conf.mean()); cal=float(q.mean())
   ps,rs,fs,support=precision_recall_fscore_support(y,pred,labels=LABELS,zero_division=0)
   ref_accuracy=ref_lookup[(window,int(seed),model)]
   predictions.append({'history_window':window,'split_seed':seed,'model':model,'batch':int(batch),
    'n':len(y),'accuracy':acc,'macro_f1_fixed_six':float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0)),
    'raw_mean_confidence':raw,'calibrated_accuracy_estimate':cal,'reference_accuracy_estimate':ref_accuracy,
    'raw_estimate_abs_error':abs(raw-acc),'calibrated_estimate_abs_error':abs(cal-acc),
    'reference_estimate_abs_error':abs(ref_accuracy-acc),
    'calibrated_estimate_signed_error':cal-acc,'calibrated_correctness_brier':float(np.mean((q-correct)**2))})
   for i,label in enumerate(LABELS): class_rows.append({'history_window':window,'split_seed':seed,'model':model,'batch':int(batch),
      'class':int(label),'support':int(support[i]),'precision':float(ps[i]),'recall':float(rs[i]),'f1':float(fs[i])})
 if len(predictions)!=expected or expected!=204: # 6 seeds x 2 models x (9+8) cohorts
  raise AssertionError(f'unexpected evaluation cohort count {len(predictions)}; expected {expected}')
 out=run/'evaluation'; out.mkdir(exist_ok=False)
 per=pd.DataFrame(predictions); per.to_csv(out/'per_cohort_metrics.csv',index=False)
 pd.DataFrame(class_rows).to_csv(out/'per_class_metrics.csv',index=False)
 agg=(per.groupby(['history_window','model','batch']).agg(
    split_count=('split_seed','nunique'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
    macro_f1_mean=('macro_f1_fixed_six','mean'),macro_f1_sd=('macro_f1_fixed_six','std'),
    raw_confidence_mae=('raw_estimate_abs_error','mean'),calibrated_mae=('calibrated_estimate_abs_error','mean'),
    reference_mae=('reference_estimate_abs_error','mean'),calibrated_brier=('calibrated_correctness_brier','mean'),
    calibrated_bias=('calibrated_estimate_signed_error','mean')).reset_index())
 agg.to_csv(out/'per_batch_descriptive_all.csv',index=False)
 common=per[per.batch.between(3,10)]
 per_split=(common.groupby(['history_window','split_seed','model']).agg(
    cohorts=('batch','nunique'),accuracy=('accuracy','mean'),macro_f1=('macro_f1_fixed_six','mean'),
    raw_confidence_mae=('raw_estimate_abs_error','mean'),calibrated_mae=('calibrated_estimate_abs_error','mean'),
    reference_mae=('reference_estimate_abs_error','mean'),calibrated_brier=('calibrated_correctness_brier','mean'),
    calibrated_bias=('calibrated_estimate_signed_error','mean')).reset_index())
 per_split.to_csv(out/'common_cohorts_per_split.csv',index=False)
 split_summary=(per_split.groupby(['history_window','model']).agg(
    split_count=('split_seed','nunique'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
    macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),
    raw_confidence_mae_mean=('raw_confidence_mae','mean'),raw_confidence_mae_sd=('raw_confidence_mae','std'),
    calibrated_mae_mean=('calibrated_mae','mean'),calibrated_mae_sd=('calibrated_mae','std'),
    reference_mae_mean=('reference_mae','mean'),reference_mae_sd=('reference_mae','std'),
    calibrated_brier_mean=('calibrated_brier','mean'),calibrated_brier_sd=('calibrated_brier','std'),
    calibrated_bias_mean=('calibrated_bias','mean'),calibrated_bias_sd=('calibrated_bias','std')).reset_index())
 split_summary.to_csv(out/'common_cohort_split_sensitivity.csv',index=False)
 paired=common.pivot(index=['history_window','split_seed','batch'],columns='model',
   values=['accuracy','macro_f1_fixed_six','calibrated_estimate_abs_error','raw_estimate_abs_error'])
 diffs=pd.DataFrame(index=paired.index).reset_index()
 for metric in ['accuracy','macro_f1_fixed_six','calibrated_estimate_abs_error','raw_estimate_abs_error']:
  diffs[f'tree_minus_logistic_{metric}']=paired[(metric,'tree_depth6')].to_numpy()-paired[(metric,'logistic')].to_numpy()
 diffs.to_csv(out/'paired_model_differences_by_seed_cohort.csv',index=False)
 summary={'status':'passed','cohort_records':len(per),'per_class_records':len(class_rows),
  'split_seeds_are_sensitivity_repeats_not_independent_domains':True,
  'all_online_cache_rows_scored':len(online),'comparison_batches':[3,4,5,6,7,8,9,10],
  'batch1_only_includes_batch2_outside_common_comparison':True,
  'common_cohort_mae_batches_3_to_10':common.groupby(['history_window','model'])[['raw_estimate_abs_error','calibrated_estimate_abs_error','reference_estimate_abs_error']].mean().reset_index().to_dict('records'),
  'batch2_logistic_history1_only':per.query("history_window=='batch1' and batch==2").groupby('model')[['raw_estimate_abs_error','calibrated_estimate_abs_error','reference_estimate_abs_error']].mean().reset_index().to_dict('records'),
  'split_seed_sd_is_descriptive_only':True}
 (out/'validation.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 lock['status']='complete';lock['evaluator_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 (lockp).write_text(json.dumps(lock,indent=2),encoding='utf-8')
 print(json.dumps(summary,indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run)
