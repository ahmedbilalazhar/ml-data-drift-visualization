"""Training-only moving-block refit uncertainty for the fixed NOAA stream."""
from __future__ import annotations
import argparse,hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,sklearn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"));sys.path.insert(0,str(ROOT))
from drift_monitoring.phase3.core import Partition,fit_artifacts,impact  # noqa: E402
from scripts.run_noaa_weather_transfer import FEATURES,split_counts  # noqa: E402
from scripts.run_noaa_reference_calibration_uncertainty import moving_block_indices  # noqa: E402

BATCH=500

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"data/processed/weather_noaa_1949_1999_v2/weather_ordered.csv")
    p.add_argument("--manifest",type=Path,default=ROOT/"data/processed/weather_noaa_1949_1999_v2/weather_manifest.json")
    p.add_argument("--output",type=Path,default=ROOT/"results/phase3/noaa_training_uncertainty_20261117_v1")
    p.add_argument("--replicates",type=int,default=200);p.add_argument("--block-length",type=int,default=30);p.add_argument("--seed",type=int,default=261117)
    a=p.parse_args();a.data=a.data.resolve();a.manifest=a.manifest.resolve();a.output=a.output.resolve()
    if a.output.exists():raise FileExistsError(a.output)
    if a.replicates<20 or a.block_length<1:raise ValueError("need >=20 bootstrap replicates and a positive block length")
    manifest=json.loads(a.manifest.read_text(encoding="utf-8"));data_sha=hashlib.sha256(a.data.read_bytes()).hexdigest()
    if data_sha!=manifest["processed"]["sha256"]:raise ValueError("NOAA processed data checksum mismatch")
    df=pd.read_csv(a.data);counts=split_counts(len(df));ntr=counts["train"];ncal=counts["calibration"];nref=counts["reference"];c0=ntr+ncal+nref
    x=df.loc[:,FEATURES].to_numpy(float);y=df.label_rain.to_numpy(np.int8);ids=df.row_id.to_numpy(np.int64)
    part=lambda lo,hi:Partition(x[lo:hi],y[lo:hi],ids[lo:hi])
    train,cal,ref=part(0,ntr),part(ntr,ntr+ncal),part(ntr+ncal,c0)
    monitor=df.iloc[c0:].reset_index(drop=True);batches=len(monitor)//BATCH;used=batches*BATCH;monitor=monitor.iloc[:used]
    mx=monitor.loc[:,FEATURES].to_numpy(float);my=monitor.label_rain.to_numpy(np.int8)
    baseline=fit_artifacts(0,train,cal,ref,n_pca_components=3,n_clusters=3,model_max_iter=2000)
    base_prob=baseline.model.predict_proba(baseline.scaler.transform(mx))[:,1]
    base_pred=(base_prob>=.5).astype(np.int8);base_conf=np.maximum(base_prob,1-base_prob)
    base_q=baseline.calibration.predict(base_conf);monitor_correct=(base_pred==my).astype(np.int8)
    baseline_batch=[]
    for b in range(batches):
        sl=slice(b*BATCH,(b+1)*BATCH);res,_=impact(base_q[sl],baseline.reference_correct,baseline.reference_n)
        baseline_batch.append({"batch":b+1,"accuracy_estimate":res["estimate"],"concern":res["concern"],"concern_flag":res["concern_flag"],"verified_accuracy":float(monitor_correct[sl].mean())})
    rows=[];meta=[];rng=np.random.default_rng(a.seed)
    for rep in range(a.replicates):
        # Logistic refitting needs both classes. Regenerate deterministically if a rare block draw omits one.
        attempt=0
        while True:
            attempt+=1;idx=moving_block_indices(ntr,a.block_length,rng)
            if len(np.unique(y[:ntr][idx]))==2:break
            if attempt>=100:raise RuntimeError(f"bootstrap replicate {rep+1} had 100 single-class training draws")
        # Unique synthetic bootstrap IDs avoid duplicate-source-ID rejection while labels/features retain block multiplicity.
        bid=-(rep+1)*ntr-np.arange(ntr,dtype=np.int64)
        bt=Partition(x[idx],y[idx],bid)
        fitted=fit_artifacts(seed=0,train=bt,calibration=cal,reference=ref,n_pca_components=3,n_clusters=3,model_max_iter=2000)
        probability=fitted.model.predict_proba(fitted.scaler.transform(mx))[:,1]
        prediction=(probability>=.5).astype(np.int8);confidence=np.maximum(probability,1-probability)
        q=fitted.calibration.predict(confidence)
        # Outcomes are joined only after this replicate's forecast probabilities and impacts are fixed.
        for b in range(batches):
            sl=slice(b*BATCH,(b+1)*BATCH);res,_=impact(q[sl],fitted.reference_correct,fitted.reference_n)
            rows.append({"replicate":rep+1,"batch":b+1,"accuracy_estimate":res["estimate"],"concern":res["concern"],
              "concern_flag":bool(res["concern_flag"]),"plug_in_lower":res["lower"],"plug_in_upper":res["upper"],
              "reference_accuracy":fitted.reference_correct/fitted.reference_n,"replicate_verified_accuracy":float((prediction[sl]==my[sl]).mean()),
              "baseline_accuracy_estimate":baseline_batch[b]["accuracy_estimate"],"baseline_concern":baseline_batch[b]["concern"],
              "baseline_concern_flag":baseline_batch[b]["concern_flag"],"baseline_verified_accuracy":baseline_batch[b]["verified_accuracy"]})
        meta.append({"replicate":rep+1,"resampling_attempts":attempt,"bootstrap_training_rows":ntr,"class_0_rows":int(np.sum(y[idx]==0)),
          "class_1_rows":int(np.sum(y[idx]==1)),"fit_seconds":float(fitted.fit_seconds)})
    a.output.mkdir(parents=True)
    frame=pd.DataFrame(rows);mf=pd.DataFrame(meta)
    frame.to_csv(a.output/"training_bootstrap_draws.csv",index=False);mf.to_csv(a.output/"replicate_metadata.csv",index=False)
    summary=frame.groupby("batch",as_index=False).agg(draws=("replicate","nunique"),accuracy_mean=("accuracy_estimate","mean"),
      accuracy_lower=("accuracy_estimate",lambda v:float(np.quantile(v,.025))),accuracy_upper=("accuracy_estimate",lambda v:float(np.quantile(v,.975))),
      concern_mean=("concern","mean"),concern_lower=("concern",lambda v:float(np.quantile(v,.025))),concern_upper=("concern",lambda v:float(np.quantile(v,.975))),
      flag_rate=("concern_flag","mean"),reference_accuracy_sd=("reference_accuracy","std"),verified_accuracy=("baseline_verified_accuracy","first"),
      replicate_verified_accuracy_mean=("replicate_verified_accuracy","mean"),
      replicate_estimator_mae=("replicate_verified_accuracy",lambda v:float(np.mean(np.abs(v.to_numpy()-frame.loc[v.index,"accuracy_estimate"].to_numpy())))),
      baseline_estimator_mae=("baseline_accuracy_estimate",lambda v:float(abs(v.iloc[0]-frame.loc[v.index[0],"baseline_verified_accuracy"]))),
      baseline_estimate=("baseline_accuracy_estimate","first"),baseline_concern=("baseline_concern","first"),baseline_flag=("baseline_concern_flag","first"))
    summary["accuracy_interval_covers_original_estimator"]=(summary.accuracy_lower<=summary.baseline_estimate)&(summary.baseline_estimate<=summary.accuracy_upper)
    summary["baseline_bootstrap_flag_disagreement_rate"]=frame.groupby("batch").apply(lambda g:float(np.mean(g.concern_flag!=g.baseline_concern_flag)),include_groups=False).to_numpy()
    summary.to_csv(a.output/"training_uncertainty_summary.csv",index=False)
    sources=[Path(__file__),a.data,a.manifest,ROOT/"src/drift_monitoring/phase3/core.py",ROOT/"scripts/run_noaa_weather_transfer.py",ROOT/"scripts/run_noaa_reference_calibration_uncertainty.py"]
    outputs=[a.output/n for n in ("training_bootstrap_draws.csv","replicate_metadata.csv","training_uncertainty_summary.csv")]
    sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
    receipt={"status":"complete","protocol":"NOAA-training-only-moving-block-full-pipeline-refit-v2","replicates":a.replicates,"training_block_length_rows":a.block_length,
      "training_rows":ntr,"calibration_reference_and_monitor":"fixed original chronological partitions; only training partition resampled",
      "refit_scope":"scaler, logistic model, PCA anchor, clusters, isotonic calibrator, and reference correctness are refit/recomputed for every training bootstrap draw",
      "bootstrap_unit":"moving blocks of processed chronological training rows; model parameter/bootstrap spread is conditional on one station and one period",
      "monitoring_labels":"used only after each draw's scores and impacts are computed to report retrospective batch accuracy; never used in fitting or thresholds",
      "single_class_draws":"regenerated deterministically and attempt counts recorded; no fitted replicate silently dropped",
      "rows_scored":used,"batches":batches,"unscored_monitoring_remainder":int(len(df)-c0-used),"baseline_reference_match":"original unresampled model is included in every row for paired comparison",
      "source_sha256":{q.relative_to(ROOT).as_posix():sha(q) for q in sources},"output_sha256":{q.relative_to(ROOT).as_posix():sha(q) for q in outputs},
      "environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__}}
    (a.output/"audit_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(summary[["accuracy_lower","accuracy_upper","concern_lower","concern_upper","baseline_bootstrap_flag_disagreement_rate","accuracy_interval_covers_original_estimator"]].agg(["mean"]).to_string(index=False))
    print(f"Completed {len(frame):,} fixed batch×training-bootstrap evaluations; output={a.output}")

if __name__=="__main__":main()
