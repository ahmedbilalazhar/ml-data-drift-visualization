"""Audit and summarize the large-seed delayed-label adaptation replication."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT/"src"))
from drift_monitoring.phase3.adaptation import make_stream, TASK_CONDITION, BATCHES, BATCH_SIZE
DEV=set(range(999001,999006)); FINAL=set(range(10000,11500))


def bootstrap(frame, groups, metric, reps=2000):
    rng=np.random.default_rng(15112026); rows=[]
    for key,g in frame.groupby(groups,sort=True,dropna=False):
        if not isinstance(key,tuple):key=(key,)
        x=g[metric].to_numpy(float);x=x[np.isfinite(x)]
        if len(x):
            draws=x[rng.integers(0,len(x),size=(reps,len(x)))].mean(1)
            mean=float(x.mean());lo,hi=np.quantile(draws,[.025,.975])
        else:mean=lo=hi=np.nan
        row={**dict(zip(groups,key)),"estimate":metric,"mean":mean,"lower":float(lo),"upper":float(hi),
             "defined_seeds":len(x),"total_seeds":len(g),"replicates":reps}
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(run:Path):
    run=run.resolve();lock=json.loads((run/"protocol_lock.json").read_text(encoding="utf-8"))
    if lock["status"]!="complete":raise ValueError("incomplete run")
    seeds=FINAL if lock["mode"]=="final" else DEV
    online=pd.read_csv(run/"online_batches.csv");ev=pd.read_csv(run/"evaluator_batches.csv");rel=pd.read_csv(run/"release_log.csv")
    forbidden={"label","y","accuracy","macro_f1","correct","harm","truth","f1"}
    if forbidden.intersection(online.columns):raise AssertionError("online output contains truth")
    if not online.prediction_persisted.astype(bool).all():raise AssertionError("online persistence marker failed")
    keys=["task","seed","condition","delay","method","batch"]
    if online.duplicated(keys).any() or ev.duplicated(keys).any():raise AssertionError("duplicate batch key")
    for task in ("v1_synthetic","xor_synthetic"):
        if set(online.loc[online.task==task,"seed"].unique())!=seeds:raise AssertionError("seed set mismatch")
    if len(rel)!=len(seeds)*2*24*3:raise AssertionError("release event denominator mismatch")
    if rel.duplicated(["task","seed","condition","delay","source_batch"]).any():raise AssertionError("duplicate release event")
    # Same source batch IDs are scheduled at different clocks but never differ by delay.
    pivot=rel.pivot(index=["task","seed","condition","source_batch"],columns="delay",values="ids_sha256")
    if pivot.isna().any().any() or not (pivot[0]==pivot[5]).all() or not (pivot[0]==pivot[20]).all():
        raise AssertionError("release ID identity mismatch across delays")
    if not np.array_equal(rel.queue_draining.to_numpy(bool), (rel.release_clock>24)):
        raise AssertionError("queue drainage marker mismatch")
    online_by=online.set_index(keys)
    for row in online.itertuples(index=False):
        expected=500*min(24,max(0,int(row.batch)-int(row.delay)-1))
        if int(row.available_monitoring_labels)!=expected:raise AssertionError("preprediction budget mismatch")
        if row.method=="cumulative_refit":
            latest=-1 if pd.isna(row.max_available_at) else int(row.max_available_at)
            if latest>=row.batch:raise AssertionError("future or same-clock label in fit")
        elif row.method!="frozen":raise AssertionError("unknown method")
    # Regenerate evaluator truth and verify every persisted class prediction.
    joins=0
    for task,condition in TASK_CONDITION:
        short="original" if task=="v1_synthetic" else "xor"
        for seed in sorted(seeds):
            *_,monitor=make_stream(seed,task,condition)
            with np.load(run/"predictions"/short/f"{seed}.npz") as cache:
                np.testing.assert_array_equal(cache["ids"],monitor.ids)
                for delay in (0,5,20):
                    pred=cache[f"pred_d{delay}"]
                    prob=cache[f"prob_d{delay}"]
                    if pred.shape!=(2,BATCHES,BATCH_SIZE) or prob.shape!=pred.shape:
                        raise AssertionError("prediction cache shape mismatch")
                    np.testing.assert_array_equal(pred,prob>=.5)
                    for mi,method in enumerate(("frozen","cumulative_refit")):
                        tab=ev[(ev.task==task)&(ev.seed==seed)&(ev.condition==condition)&
                               (ev.delay==delay)&(ev.method==method)].sort_values("batch")
                        correct=(pred[mi]==monitor.y.reshape(BATCHES,BATCH_SIZE)).sum(1)
                        np.testing.assert_array_equal(tab.correct.to_numpy(),correct)
                        live=online[(online.task==task)&(online.seed==seed)&(online.condition==condition)&
                                    (online.delay==delay)&(online.method==method)].sort_values("batch")
                        np.testing.assert_allclose(live.probability_mean.to_numpy(),prob[mi].mean(1),atol=1e-14)
                        np.testing.assert_allclose(live.predicted_positive_rate.to_numpy(),pred[mi].mean(1),atol=0)
                        expected_ids=[hashlib.sha256(np.asarray(monitor.ids[b*BATCH_SIZE:(b+1)*BATCH_SIZE],dtype="<i8").tobytes()).hexdigest()
                                      for b in range(BATCHES)]
                        if list(live.prediction_id_sha256)!=expected_ids:raise AssertionError("prediction ID digest mismatch")
                        joins+=len(correct)
    # Add batch regimes and form seed-level paired differences before bootstrap.
    joined=ev.copy()
    joined["regime"]=["pre" if b<=8 else "late_post" if b>=17 else "post" for b in joined.batch]
    seed=joined.groupby(["task","seed","condition","delay","method","regime"],as_index=False).agg(
        accuracy=("accuracy","mean"),macro_f1=("macro_f1","mean"),batches=("batch","nunique"))
    wide=seed.pivot(index=["task","seed","condition","delay","regime"],columns="method",values=["accuracy","macro_f1"])
    delta=[]
    for metric in ("accuracy","macro_f1"):
        d=(wide[(metric,"cumulative_refit")]-wide[(metric,"frozen")]).rename("difference").reset_index()
        d["metric"]=metric;delta.append(d)
    delta=pd.concat(delta,ignore_index=True)
    seed.to_csv(run/"per_seed_regime.csv",index=False)
    delta.to_csv(run/"paired_seed_differences.csv",index=False)
    summaries=[]
    for metric in ("accuracy","macro_f1"):
        part=delta[delta.metric==metric]
        summaries.append(bootstrap(part,["task","condition","delay","regime","metric"],"difference"))
    pd.concat(summaries,ignore_index=True).to_csv(run/"paired_bootstrap.csv",index=False)
    budget=online[online.method=="cumulative_refit"].groupby(["task","seed","condition","delay"],as_index=False).agg(
        batch24_labels=("available_monitoring_labels","max"))
    budget.to_csv(run/"batch24_label_budget_per_seed.csv",index=False)
    validation={"status":"passed","mode":lock["mode"],"seeds_per_task":len(seeds),
        "online_rows":len(online),"evaluator_rows":len(ev),"release_rows":len(rel),
        "prediction_truth_joins":joins,"release_ids_equal_across_delays":True,
        "preprediction_budget_exact":True,"no_same_or_future_labels":True,
        "queues_drained":True,"online_truth_absent":True,"bootstrap_replicates":2000,
        "resampling_unit":"seed"}
    (run/"validation.json").write_text(json.dumps(validation,indent=2),encoding="utf-8")
    return validation


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,required=True);a=p.parse_args()
    print(json.dumps(evaluate(a.run),indent=2))
