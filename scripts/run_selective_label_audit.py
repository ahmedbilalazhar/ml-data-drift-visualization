"""Probability-sampled selective-label audit with feature-only acquisition."""
from __future__ import annotations
import argparse,hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,sklearn
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from drift_monitoring.phase3.core import Partition,fit_artifacts  # noqa: E402
from drift_monitoring.phase3.strengthening_generators import generate_historical,generate_monitoring  # noqa: E402

def select_from_confidence(confidence:np.ndarray, rng:np.random.Generator, *,
                           policy:str, target_fraction:float=.10, audit_probability:float=.10)->tuple[np.ndarray,np.ndarray]:
    """Return selected indices and known first-order inclusion probabilities.

    This acquisition function receives confidence only, never labels or correctness.
    """
    n=len(confidence)
    if confidence.ndim!=1 or n==0 or not np.isfinite(confidence).all() or np.any((confidence<.5)|(confidence>1)):
        raise ValueError("confidence must be a nonempty vector in [.5,1]")
    if not 0<target_fraction<1 or not 0<audit_probability<1:
        raise ValueError("target fraction and audit probability must lie in (0,1)")
    if policy=="targeted_10pct":
        k=max(1,int(round(n*target_fraction)))
        chosen=np.argsort(confidence,kind="stable")[:k]
        pi=np.ones(k)
    elif policy=="stratified_random_audit":
        k=max(1,int(round(n*target_fraction)))
        target=np.argsort(confidence,kind="stable")[:k]
        is_target=np.zeros(n,dtype=bool);is_target[target]=True
        random=(rng.random(n)<audit_probability)&~is_target
        chosen=np.flatnonzero(is_target|random)
        pi=np.where(is_target[chosen],1.0,audit_probability)
    elif policy=="simple_random_budget_matched":
        p=target_fraction+(1-target_fraction)*audit_probability
        chosen=np.flatnonzero(rng.random(n)<p)
        pi=np.full(len(chosen),p)
    else: raise ValueError(f"unknown selective-label policy: {policy}")
    return chosen.astype(int),pi.astype(float)

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--config",type=Path,default=ROOT/"configs/phase3_strengthening_v3.json")
 p.add_argument("--output",type=Path,default=ROOT/"results/phase3/selective_label_audit_20261117_v2")
 p.add_argument("--target-fraction",type=float,default=.10)
 p.add_argument("--audit-probability",type=float,default=.10)
 p.add_argument("--seed-limit",type=int,default=None,help="development smoke only; default uses the full locked evaluation bank")
 a=p.parse_args();a.config=a.config.resolve();a.output=a.output.resolve()
 if a.output.exists():raise FileExistsError(f"refusing to overwrite {a.output}")
 cfg=json.loads(a.config.read_text(encoding="utf-8"));seeds=list(map(int,cfg["seed_banks"]["evaluation"]))
 if a.seed_limit is not None:
  if a.seed_limit<1:raise ValueError("seed limit must be positive")
  seeds=seeds[:a.seed_limit]
 rows=[];selected_rows=[];tasks=("linear_probabilistic","five_region");mechanisms=("M0_stationary","M4_abrupt_conditional_reversal")
 task_code={t:i for i,t in enumerate(tasks)};mech_code={m:i for i,m in enumerate(mechanisms)}
 ntrain=int(cfg["history"]["train_rows"]);ncal=int(cfg["history"]["calibration_rows"]);nref=int(cfg["history"]["reference_rows"])
 for task in tasks:
  for seed in seeds:
   train=generate_historical(task,seed,1,ntrain,8);cal=generate_historical(task,seed,2,ncal,8);ref=generate_historical(task,seed,3,nref,8)
   fitted=fit_artifacts(seed,train,cal,ref,n_pca_components=3,n_clusters=3,model_max_iter=2000)
   for mechanism in mechanisms:
    stream=generate_monitoring(task,mechanism,seed,feature_count=8)
    mx=stream.monitor.x;my=stream.evaluator.labels;mids=stream.monitor.ids
    prob=fitted.model.predict_proba(fitted.scaler.transform(mx))[:,1]
    pred=(prob>=.5).astype(np.int8);confidence=np.maximum(prob,1-prob)
    for batch in range(96):
     lo=batch*500;hi=lo+500;conf=confidence[lo:hi];ids=mids[lo:hi]
     # Finish all policy decisions from X/prediction confidence before opening
     # the current cohort's evaluator outcomes.
     selections=[]
     for policy,pcode in (("targeted_10pct",0),("stratified_random_audit",1),("simple_random_budget_matched",2)):
      rng=np.random.default_rng(np.random.SeedSequence([seed,task_code[task],mech_code[mechanism],batch+1,pcode,761]))
      idx,pi=select_from_confidence(conf,rng,policy=policy,target_fraction=a.target_fraction,audit_probability=a.audit_probability)
      selections.append((policy,idx,pi))
     # Evaluator-only label join occurs after all feature-only selections.
     truth=(pred[lo:hi]==my[lo:hi]).astype(np.int8);n=len(truth)
     for policy,idx,pi in selections:
      selected=truth[idx].astype(float);n=len(truth)
      naive=float(selected.mean()) if len(idx) else float("nan")
      if policy=="targeted_10pct":
       ht=var=lo95=hi95=float("nan")
      elif policy=="stratified_random_audit":
       target=pi==1.0;rand=~target
       total=float(selected[target].sum()+np.sum(selected[rand]/pi[rand]))
       ht=total/n
       variance=float(np.sum(selected[rand]*(1-a.audit_probability)/(a.audit_probability**2)))/n**2
       se=np.sqrt(max(variance,0));lo95=ht-1.96*se;hi95=ht+1.96*se
      else:
       psel=a.target_fraction+(1-a.target_fraction)*a.audit_probability
       ht=float(np.sum(selected/pi)/n)
       variance=float(np.sum(selected*(1-psel)/(psel**2)))/n**2
       se=np.sqrt(max(variance,0));lo95=ht-1.96*se;hi95=ht+1.96*se
      ess=float((np.sum(1/pi)**2)/np.sum(1/pi**2)) if len(pi) else 0.0
      rows.append({"task":task,"seed":seed,"mechanism":mechanism,"batch":batch+1,"policy":policy,
       "target_rows":int(round(n*a.target_fraction)) if policy!="simple_random_budget_matched" else 0,
       "selected_rows":len(idx),"full_stream_accuracy":float(truth.mean()),"selected_naive_accuracy":naive,
       "horvitz_thompson_accuracy":ht,"ht_variance_estimate":var if policy=="targeted_10pct" else variance,
       "ht_ci_lower":lo95,"ht_ci_upper":hi95,"ht_ci_covers":bool(lo95<=truth.mean()<=hi95) if np.isfinite(lo95) else None,
       "effective_sample_size":ess,"known_inclusion_probabilities":True,
       "population_estimable":policy!="targeted_10pct"})
      for row_i,prob_i in zip(idx,pi,strict=True):
       selected_rows.append({"task":task,"seed":seed,"mechanism":mechanism,"batch":batch+1,"policy":policy,
        "row_id":int(ids[row_i]),"inclusion_probability":float(prob_i),"confidence":float(conf[row_i]),
        "prediction":int(pred[lo+row_i]),"correctness":int(truth[row_i]),
        "selection_rule":"confidence-only target plus independent Bernoulli audit" if policy=="stratified_random_audit" else policy})
 a.output.mkdir(parents=True);batch_frame=pd.DataFrame(rows);selected_frame=pd.DataFrame(selected_rows)
 batch_frame.to_csv(a.output/"batch_estimates.csv",index=False);selected_frame.to_csv(a.output/"selected_label_log.csv",index=False)
 # Average within seed first; seed is the independent simulation unit.
 metrics=["selected_rows","selected_naive_accuracy","horvitz_thompson_accuracy","ht_ci_covers","effective_sample_size"]
 seeds_frame=batch_frame.groupby(["task","mechanism","policy","seed"],as_index=False)[metrics].mean()
 rng=np.random.default_rng(20261117);summaries=[]
 for (task,mechanism,policy),g in seeds_frame.groupby(["task","mechanism","policy"],sort=True):
  for metric in metrics:
   v=g[metric].dropna().to_numpy(float)
   if not len(v):continue
   draws=v[rng.integers(0,len(v),(5000,len(v)))].mean(axis=1)
   summaries.append({"task":task,"mechanism":mechanism,"policy":policy,"metric":metric,"seeds":len(v),
    "mean":float(v.mean()),"seed_sd":float(v.std(ddof=1)),"lower":float(np.quantile(draws,.025)),"upper":float(np.quantile(draws,.975))})
 summary=pd.DataFrame(summaries);summary.to_csv(a.output/"seed_bootstrap_summary.csv",index=False)
 sources=[Path(__file__),a.config,ROOT/"src/drift_monitoring/phase3/strengthening_generators.py",ROOT/"src/drift_monitoring/phase3/core.py"]
 outputs=[a.output/"batch_estimates.csv",a.output/"selected_label_log.csv",a.output/"seed_bootstrap_summary.csv"]
 sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
 receipt={"status":"complete","protocol":"feature-only-selective-label-known-probability-audit-v1","seed_bank":seeds,
  "tasks":list(tasks),"mechanisms":list(mechanisms),"batches_per_stream":96,"batch_size":500,
  "target_fraction":a.target_fraction,"audit_probability_outside_target":a.audit_probability,
  "equal_expected_budget_control_probability":a.target_fraction+(1-a.target_fraction)*a.audit_probability,
  "selection_feature_only":True,"selected_label_inclusion_probability_logged":True,
  "targeted_only_population_estimability":False,"stratified_and_random_estimator":"Horvitz-Thompson with Bernoulli design variance; untruncated, so estimates/intervals can exceed [0,1]",
  "monitoring_truth_role":"used after feature-only selection for selected outcomes and evaluator-only full-population scoring",
  "independent_unit":"seed; batches averaged within seed before bootstrap","bootstrap_replicates":5000,
  "source_sha256":{q.relative_to(ROOT).as_posix():sha(q) for q in sources},
  "output_sha256":{q.relative_to(ROOT).as_posix():sha(q) for q in outputs},
  "environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__}}
 (a.output/"audit_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
 print(summary.to_string(index=False));print(f"Selected {len(selected_frame):,} labels across {len(batch_frame):,} batch-policy cells at {a.output}")

if __name__=="__main__":main()
