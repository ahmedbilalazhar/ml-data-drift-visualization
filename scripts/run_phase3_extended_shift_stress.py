"""Run a fixed-model forecast stress test on fresh paired conditional shifts."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, time
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
from threadpoolctl import threadpool_limits
from run_phase3_six_hour_forecast import ROOT, condition as old_condition, released_features
from run_future_harm_unseen_family import FEATURES, DEV, TRAIN_RUN
from drift_monitoring.phase3.core import FeatureBatch, baseline_x, fit_artifacts, historical_partition, innovations, label_probability, score_batch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PLAN=ROOT/"docs/phase3/SIX_HOUR_EXTENSION_20261120.md"
EVAL_SEEDS=range(973000,974200)
ONSETS=(5,9,13,17,21)
DELAYS=(0,5,20)
LAWS=("stationary","flatten_035","flatten_070","flatten_100","regional_neg075",
      "regional_zero","regional_pos075","sensor_sign_inversion","joint_boundary")

def sha(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def train_models():
    lock=json.loads((TRAIN_RUN/"protocol_lock.json").read_text())
    if lock.get("protocol")!="future_harm_forecast_20261116_v1":raise ValueError("Training lock mismatch")
    xp=TRAIN_RUN/"predictions/online_features.csv";yp=TRAIN_RUN/"evaluator_truth/future_targets.csv"
    x,y=pd.read_csv(xp),pd.read_csv(yp);x=x[x.seed.isin(DEV)];y=y[y.seed.isin(DEV)]
    tr=x.merge(y,on=["seed","condition","delay","batch"],validate="one_to_one")
    models={}
    for name,cols in FEATURES.items():
        model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=2026))
        model.fit(tr[cols].replace([np.inf,-np.inf],np.nan).fillna(-1),tr.future_harm);models[name]=model
    return models,float(tr.future_harm.mean()),[xp,yp,TRAIN_RUN/"protocol_lock.json"]

def make_condition(clean,u,law,onset):
    x=clean.copy();p=label_probability(clean);start=(onset-1)*500;threshold=((-0.75,0.0,0.75)[LAWS.index(law)-4] if law.startswith("regional") else 0)
    if law=="flatten_035":p[start:]=.65*p[start:]+.35*.5
    elif law=="flatten_070":p[start:]=.30*p[start:]+.70*.5
    elif law=="flatten_100":p[start:]=.5
    elif law.startswith("regional"):
        mask=(np.arange(len(x))>=start)&(clean[:,0]>=threshold);p[mask]=1-p[mask]
    elif law=="sensor_sign_inversion":x[start:,0]*=-1
    elif law=="joint_boundary":
        x[start:,:3]*=.25;p=label_probability(x)
    elif law!="stationary":raise ValueError(law)
    y=(u<p).astype(np.int8)
    return x,y

def job_run(job):
    seed,onset,deadline=job
    with threadpool_limits(limits=1):
        art=fit_artifacts(seed,*(historical_partition(seed,s,n) for s,n in enumerate((4000,2000,2000))))
        ref=art.reference_correct/art.reference_n;z,u=innovations(seed,3,12000);clean=baseline_x(z)
        ids=seed*1_000_000+500_000+np.arange(len(u));pred_cache={};rows=[];targets=[]
        for law in LAWS:
            x,y=make_condition(clean,u,law,onset)
            key=law if law in ("sensor_sign_inversion","joint_boundary") else "clean"
            if key not in pred_cache:
                pp=[]
                for b in range(1,25):
                    if time.monotonic()>deadline:raise TimeoutError("Six-hour wall clock reached")
                    sl=slice((b-1)*500,b*500);fb=FeatureBatch(x[sl],ids[sl],seed,law,b)
                    rec,cache,_=score_batch(art,fb);alarm=int(bool(rec["input_alarm"]))
                    pp.append((cache["prediction"],dict(estimate=rec["cal_estimate"],concern=rec["cal_concern"],
                        alarm_count=sum(bool(rec[f"x{i}_flag"]) for i in range(8)),min_adjusted_p=min(rec[f"x{i}_adjusted_p"] for i in range(8)),
                        input_alarm=alarm)))
                pred_cache[key]=pp
            accuracy=np.asarray([(pred==y[b*500:(b+1)*500]).mean() for b,(pred,_) in enumerate(pred_cache[key])])
            for delay in DELAYS:
                for t in range(1,20):
                    avail=max(0,t-delay-1);hist=accuracy[:avail]
                    latest=float(hist[-1]) if avail else ref
                    rolling=float(hist[-3:].mean()) if avail else ref
                    future=float(accuracy[t:t+5].mean());feat=pred_cache[key][t-1][1]
                    rows.append(dict(seed=seed,onset=onset,law=law,delay=delay,origin=t,accuracy=accuracy[t-1],
                       reference_accuracy=ref,estimate=feat["estimate"],concern=feat["concern"],alarm_count=feat["alarm_count"],
                       input_alarm=feat["input_alarm"],latest_accuracy=latest,rolling_accuracy=rolling,released_n=avail*500,
                       time_norm=t/24,delay_norm=delay/20))
                    targets.append(dict(seed=seed,onset=onset,law=law,delay=delay,origin=t,future_accuracy=future,
                        future_harm=int(future<ref-.05)))
        return rows,targets

def write(path,records):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]),lineterminator="\n");w.writeheader();w.writerows(records)

def main(out,workers,seconds,eval_count,seed_start):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic();models,prevalence,train_sources=train_models()
    sources=[*train_sources,PLAN,Path(__file__).resolve(),ROOT/"scripts/run_future_harm_unseen_family.py",ROOT/"src/drift_monitoring/phase3/core.py"]
    eval_seeds=range(seed_start,seed_start+eval_count)
    metadata=dict(protocol="phase3_extended_shift_stress_20261120_v1",status="running",
      training_seeds=list(DEV),evaluation_seeds=[*eval_seeds],onsets=list(ONSETS),laws=list(LAWS),delays=list(DELAYS),
      model_names=list(FEATURES),development_prevalence=prevalence,python=__import__("platform").python_version(),
      source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    lock=out/"protocol_lock.json";lock.write_text(json.dumps(metadata,indent=2)+"\n")
    done=[];features=[];targets=[];jobs=[(s,ONSETS[(s-seed_start)%len(ONSETS)],start+seconds) for s in eval_seeds]
    try:
      with ProcessPoolExecutor(max_workers=workers) as pool:
       for j,(x,y) in enumerate(pool.map(job_run,jobs),1):
        features.extend(x);targets.extend(y);done.append(jobs[j-1][0])
        if j%25==0:
         write(out/"online_features.partial.csv",features);write(out/"evaluator_targets.partial.csv",targets)
        if j%100==0:
         (out/"progress.json").write_text(json.dumps(dict(completed_seeds=j,total_seeds=len(jobs),elapsed_seconds=time.monotonic()-start),indent=2)+"\n")
         print(f"{j}/{len(jobs)} seeds in {time.monotonic()-start:.0f}s",flush=True)
      write(out/"online_features.csv",features);write(out/"evaluator_targets.csv",targets)
      metadata["status"]="complete"
    except BaseException as e:
      metadata["status"]="incomplete";metadata["error"]=repr(e)
      if features:write(out/"online_features.partial.csv",features);write(out/"evaluator_targets.partial.csv",targets)
      raise
    finally:
      metadata["completed_seeds"]=done;metadata["elapsed_seconds"]=time.monotonic()-start
      lock.write_text(json.dumps(metadata,indent=2)+"\n")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True);p.add_argument("--workers",type=int,choices=(1,2,3),default=3);p.add_argument("--seconds",type=int,default=16200);p.add_argument("--eval-seeds",type=int,default=1200);p.add_argument("--seed-start",type=int,default=973000);a=p.parse_args();main(a.out,a.workers,a.seconds,a.eval_seeds,a.seed_start)
