"""Retrospective probability-sampling replay on the frozen NOAA monitoring stream.

Acquisition policies receive confidence only; evaluator correctness is joined only
after all policy samples for a draw have been fixed. This is not a prospective
label-arrival experiment and does not alter the original NOAA transfer run.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT/"scripts"))
from drift_monitoring.phase3.core import Partition,fit_artifacts  # noqa: E402
from run_selective_label_audit import select_from_confidence  # noqa: E402
from run_noaa_weather_transfer import FEATURES,split_counts  # noqa: E402

POLICIES=("targeted_10pct","stratified_random_audit","simple_random_budget_matched")
P_CODES={name:i for i,name in enumerate(POLICIES)}
BATCH_SIZE=500

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"data/processed/weather_noaa_1949_1999_v2/weather_ordered.csv")
    p.add_argument("--manifest",type=Path,default=ROOT/"data/processed/weather_noaa_1949_1999_v2/weather_manifest.json")
    p.add_argument("--reference-run",type=Path,default=ROOT/"results/phase3/noaa_weather_1949_1999_v1")
    p.add_argument("--output",type=Path,default=ROOT/"results/phase3/noaa_selective_label_replay_20261117_v1")
    p.add_argument("--replicates",type=int,default=200)
    p.add_argument("--seed",type=int,default=261117)
    a=p.parse_args();a.data=a.data.resolve();a.manifest=a.manifest.resolve();a.reference_run=a.reference_run.resolve();a.output=a.output.resolve()
    if a.output.exists():raise FileExistsError(a.output)
    if a.replicates<20:raise ValueError("at least 20 design replicates are required")
    df=pd.read_csv(a.data);manifest=json.loads(a.manifest.read_text(encoding="utf-8"))
    if len(df)!=int(manifest["processed"]["row_count"]) or not np.array_equal(df.row_id.to_numpy(),np.arange(len(df))):
        raise ValueError("NOAA row count or stable row IDs do not match manifest")
    if hashlib.sha256(a.data.read_bytes()).hexdigest()!=manifest["processed"]["sha256"]:
        raise ValueError("NOAA processed data checksum does not match manifest")
    if df.timestamp.duplicated().any() or not df.timestamp.is_monotonic_increasing:
        raise ValueError("NOAA chronology is invalid")
    counts=split_counts(len(df));ntrain=counts["train"];ncal=counts["calibration"];nref=counts["reference"]
    cut1=ntrain;cut2=cut1+ncal;cut3=cut2+nref
    train=Partition(df.loc[:cut1-1,FEATURES].to_numpy(float),df.label_rain.iloc[:cut1].to_numpy(np.int8),df.row_id.iloc[:cut1].to_numpy(np.int64))
    calibration=Partition(df.loc[cut1:cut2-1,FEATURES].to_numpy(float),df.label_rain.iloc[cut1:cut2].to_numpy(np.int8),df.row_id.iloc[cut1:cut2].to_numpy(np.int64))
    reference=Partition(df.loc[cut2:cut3-1,FEATURES].to_numpy(float),df.label_rain.iloc[cut2:cut3].to_numpy(np.int8),df.row_id.iloc[cut2:cut3].to_numpy(np.int64))
    fitted=fit_artifacts(seed=0,train=train,calibration=calibration,reference=reference,n_pca_components=3,n_clusters=3,model_max_iter=2000)
    monitor=df.iloc[cut3:].reset_index(drop=True);n_batches=len(monitor)//BATCH_SIZE;used=n_batches*BATCH_SIZE
    monitor=monitor.iloc[:used].copy();x=monitor.loc[:,FEATURES].to_numpy(float);y=monitor.label_rain.to_numpy(np.int8);ids=monitor.row_id.to_numpy(np.int64)
    prob=fitted.model.predict_proba(fitted.scaler.transform(x))[:,1];pred=(prob>=.5).astype(np.int8);confidence=np.maximum(prob,1-prob)
    correctness=(pred==y).astype(np.int8);truth=float(correctness.mean())
    # Confirm the reconstructed frozen classifier exactly reproduces the locked NOAA batch outcomes.
    ref_metrics=pd.read_csv(a.reference_run/"evaluator/batch_metrics.csv")
    if len(ref_metrics)!=n_batches:raise ValueError("reference NOAA run and replay batch counts differ")
    per_batch=np.array([correctness[j*BATCH_SIZE:(j+1)*BATCH_SIZE].mean() for j in range(n_batches)])
    if not np.allclose(per_batch,ref_metrics.accuracy.to_numpy(float),atol=0,rtol=0):
        raise ValueError("reconstructed model predictions do not exactly match locked NOAA batch accuracy")
    a.output.mkdir(parents=True)
    metric_rows=[];selected_path=a.output/"selected_label_log.csv.gz"
    columns=["replicate","batch","policy","row_id","inclusion_probability","confidence","prediction","correctness"]
    with gzip.open(selected_path,"wt",encoding="utf-8",newline="",compresslevel=6) as gz:
        writer=csv.DictWriter(gz,fieldnames=columns);writer.writeheader()
        for rep in range(a.replicates):
            selections=[]
            for batch in range(n_batches):
                lo=batch*BATCH_SIZE;hi=lo+BATCH_SIZE;conf=confidence[lo:hi]
                for policy in POLICIES:
                    rng=np.random.default_rng(np.random.SeedSequence([a.seed,rep,batch+1,P_CODES[policy],991]))
                    idx,pi=select_from_confidence(conf,rng,policy=policy,target_fraction=.10,audit_probability=.10)
                    selections.append((batch,policy,lo,idx,pi))
            # No evaluator correctness is read until all policy selections for the full stream are fixed.
            totals={policy:{"weighted_correct":0.0,"variance":0.0,"n_selected":0,"weights":[]} for policy in POLICIES}
            targeted_correct=targeted_n=0
            for batch,policy,lo,idx,pi in selections:
                t=correctness[lo+idx].astype(float);s=totals[policy];s["n_selected"]+=len(idx)
                s["weights"].extend((1/pi).tolist())
                for k,i,prob_i,conf_i in zip(idx,t,pi,confidence[lo+idx],strict=True):
                    writer.writerow({"replicate":rep,"batch":batch+1,"policy":policy,"row_id":int(ids[lo+k]),
                      "inclusion_probability":float(prob_i),"confidence":float(conf_i),"prediction":int(pred[lo+k]),"correctness":int(i)})
                if policy=="targeted_10pct":
                    targeted_correct+=float(t.sum());targeted_n+=len(t)
                elif policy=="stratified_random_audit":
                    is_target=pi==1.0;rand=~is_target
                    s["weighted_correct"]+=float(t[is_target].sum()+np.sum(t[rand]/pi[rand]))
                    s["variance"]+=float(np.sum(t[rand]*(1-.10)/(.10**2)))
                else:
                    s["weighted_correct"]+=float(np.sum(t/pi))
                    s["variance"]+=float(np.sum(t*(1-.19)/(.19**2)))
            for policy,s in totals.items():
                if policy=="targeted_10pct":estimate=float(targeted_correct/targeted_n);var=float("nan");estimable=False
                else:estimate=s["weighted_correct"]/used;var=s["variance"]/(used**2);estimable=True
                se=float(np.sqrt(var)) if estimable else float("nan")
                weights=np.asarray(s["weights"],float);ess=float(weights.sum()**2/np.sum(weights**2))
                metric_rows.append({"replicate":rep,"policy":policy,"finite_stream_truth":truth,"selected_naive_accuracy":estimate,
                  "ht_accuracy":estimate if estimable else float("nan"),"ht_variance_estimate":var,"ht_ci_lower":estimate-1.96*se if estimable else float("nan"),
                  "ht_ci_upper":estimate+1.96*se if estimable else float("nan"),"ht_ci_covers":bool(estimate-1.96*se<=truth<=estimate+1.96*se) if estimable else None,
                  "selected_rows":int(s["n_selected"]),"effective_sample_size":ess,"population_estimable":estimable})
    metrics=pd.DataFrame(metric_rows);metrics.to_csv(a.output/"design_replicate_metrics.csv",index=False)
    summaries=[]
    for policy,g in metrics.groupby("policy",sort=False):
        est=g.selected_naive_accuracy.to_numpy(float);use=g[g.population_estimable]
        summaries.append({"policy":policy,"design_replicates":a.replicates,"finite_stream_truth":truth,
          "mean_selected_naive_accuracy":float(est.mean()),"mean_estimation_error":float((est-truth).mean()),
          "design_sd_selected_accuracy":float(est.std(ddof=1)),"mean_selected_rows":float(g.selected_rows.mean()),
          "mean_ess":float(g.effective_sample_size.mean()),"population_estimable":bool(g.population_estimable.all()),
          "ht_bias":float((use.ht_accuracy-truth).mean()) if len(use) else float("nan"),
          "ht_ci_coverage":float(use.ht_ci_covers.mean()) if len(use) else float("nan"),
          "mean_ht_ci_width":float((use.ht_ci_upper-use.ht_ci_lower).mean()) if len(use) else float("nan")})
    summary=pd.DataFrame(summaries);summary.to_csv(a.output/"design_summary.csv",index=False)
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    source_paths=[Path(__file__),Path(__file__).with_name("run_selective_label_audit.py"),a.data,a.manifest,
      ROOT/"src/drift_monitoring/phase3/core.py",a.reference_run/"evaluator/batch_metrics.csv"]
    outputs=[a.output/q for q in ("selected_label_log.csv.gz","design_replicate_metrics.csv","design_summary.csv")]
    receipt={"status":"complete","protocol":"noaa-retrospective-known-probability-selective-label-replay-v1",
      "interpretation":"retrospective simulated sampling on one fixed NOAA chronology; not prospective label operations or domain-level replication",
      "dataset_id":manifest["dataset_id"],"rows_scored":used,"unscored_monitoring_remainder":int(len(df)-cut3-used),
      "batches":n_batches,"batch_size":BATCH_SIZE,"finite_stream_accuracy":truth,"design_replicates":a.replicates,"design_seed":a.seed,
      "policies":{"targeted_10pct":"certainty sample of 10% lowest confidence; zero inclusion probability outside target so population accuracy is not estimable",
        "stratified_random_audit":"10% lowest-confidence certainty stratum plus Bernoulli p=0.10 on remainder; known inclusion probabilities 1 or 0.10",
        "simple_random_budget_matched":"Bernoulli p=0.19 across stream; same expected label count"},
      "sampling_unit":"row within batch; design-replicate summaries condition on the single observed finite monitoring stream",
      "selection_label_order":"all feature-only policy selections are fixed before any monitoring correctness is joined for that design draw",
      "model":"same frozen LogisticRegression and historical splits as the locked NOAA transfer runner; reconstructed batch accuracy exactly matched reference",
      "source_sha256":{str(q.relative_to(ROOT)).replace('\\','/'):sha(q) for q in source_paths},
      "output_sha256":{q.relative_to(ROOT).as_posix():sha(q) for q in outputs},
      "software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scikit_learn":sklearn.__version__}}
    (a.output/"audit_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(summary.to_string(index=False));print(f"Saved {len(metrics):,} design-policy estimates and sampled-row logs to {a.output}")

if __name__=="__main__":main()
