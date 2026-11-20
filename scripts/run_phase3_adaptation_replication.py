"""Run a large fresh-seed replication of the locked label-update pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import warnings
from datetime import datetime, timezone

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from drift_monitoring.phase3.adaptation import (  # noqa: E402
    BATCHES, BATCH_SIZE, CumulativeLogisticAdapter, DELAYS, LabelReleaseBroker,
    TASK_CONDITION, digest_ids, label_budget_before, make_stream,
)

MANIFEST = ROOT / "docs/phase3/ADAPTATION_SEED_REPLICATION_PROTOCOL.md"
DEV_SEEDS = tuple(range(999001, 999006))
FINAL_SEEDS = tuple(range(10000, 11500))


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def run(mode: str, out: Path) -> None:
    out=out.resolve()
    if out.exists(): raise FileExistsError(f"refusing to overwrite {out}")
    seeds=DEV_SEEDS if mode=="development" else FINAL_SEEDS if mode=="final" else None
    if seeds is None: raise ValueError(mode)
    out.mkdir(parents=True)
    (out/"predictions").mkdir()
    online_fields=("task","seed","condition","delay","method","batch","n",
        "available_monitoring_labels","training_rows","fit_count","monitor_id_sha256",
        "max_available_at","prediction_persisted","probability_mean","predicted_positive_rate",
        "prediction_id_sha256")
    eval_fields=("task","seed","condition","delay","method","batch","n","correct",
        "accuracy","macro_f1","class0_n","class1_n")
    release_fields=("task","seed","condition","delay","source_batch","available_at",
        "release_clock","queue_draining","n","ids_sha256")
    online_path=out/"online_batches.csv"; eval_path=out/"evaluator_batches.csv"; release_path=out/"release_log.csv"
    lock={"protocol":"large fresh-seed replication of released-label adaptation pilot",
      "status":"running","mode":mode,"started_at_utc":datetime.now(timezone.utc).isoformat(),
      "seeds_by_task":{task:list(seeds) for task,_ in TASK_CONDITION},
      "conditions":{task:condition for task,condition in TASK_CONDITION},
      "delays":list(DELAYS),"batch_size":BATCH_SIZE,"batches":BATCHES,
      "manifest_sha256":sha(MANIFEST),"source_sha256":{p:sha(ROOT/p) for p in (
        "src/drift_monitoring/phase3/adaptation.py","scripts/run_phase3_adaptation_replication.py",
        "docs/phase3/ADAPTATION_SEED_REPLICATION_PROTOCOL.md")},"thread_limit":1,
      "online_contract":"no truth fields; write and flush before broker release",
      "cache_policy":"one compact prediction array per task-seed, truth kept in evaluator-only output"}
    (out/"protocol_lock.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")
    packages=("numpy","scikit-learn","pandas","threadpoolctl")
    (out/"environment.json").write_text(json.dumps({"python":sys.version,"executable":sys.executable,
      "platform":platform.platform(),"packages":{p:importlib.metadata.version(p) for p in packages},
      "thread_limit":1},indent=2),encoding="utf-8")
    start=time.perf_counter(); feature_streams=delay_streams=0
    try:
      with online_path.open("w",newline="",encoding="utf-8") as of, \
           eval_path.open("w",newline="",encoding="utf-8") as ef, \
           release_path.open("w",newline="",encoding="utf-8") as rf, threadpool_limits(limits=1):
        warnings.simplefilter("error", ConvergenceWarning)
        ow=csv.DictWriter(of,fieldnames=online_fields); ew=csv.DictWriter(ef,fieldnames=eval_fields); rw=csv.DictWriter(rf,fieldnames=release_fields)
        ow.writeheader();ew.writeheader();rw.writeheader()
        for task,condition in TASK_CONDITION:
          for index,seed in enumerate(seeds,1):
            t0=time.perf_counter()
            train,calibration,reference,monitor=make_stream(seed,task,condition)
            if any(np.intersect1d(a.ids,b.ids).size for i,a in enumerate((train,calibration,reference,monitor))
                   for b in (train,calibration,reference,monitor)[i+1:]):
              raise AssertionError("partition IDs overlap")
            scaler=StandardScaler().fit(train.x); train_x=scaler.transform(train.x)
            frozen=LogisticRegression(C=1.0,max_iter=500,random_state=seed).fit(train_x,train.y)
            all_predictions={}; all_probabilities={}
            for delay in DELAYS:
              broker=LabelReleaseBroker(monitor,delay); adapter=CumulativeLogisticAdapter(seed,train)
              pred_stack=np.empty((2,BATCHES,BATCH_SIZE),dtype=np.int8)
              prob_stack=np.empty((2,BATCHES,BATCH_SIZE),dtype=np.float64)
              released_batches=set()
              for batch in range(1,BATCHES+1):
                sl=slice((batch-1)*BATCH_SIZE,batch*BATCH_SIZE)
                x=monitor.x[sl]; ids=monitor.ids[sl]
                frozen_p=frozen.predict_proba(scaler.transform(x))[:,1]
                frozen_y=(frozen_p>=.5).astype(np.int8)
                adapted_y,adapted_p=adapter.predict(x)
                budget=label_budget_before(batch,delay); audit=adapter.audit(batch)
                if audit["released_monitoring_n"]!=budget: raise AssertionError("preprediction label budget mismatch")
                predictions=(frozen_y,adapted_y); probabilities=(frozen_p,adapted_p)
                for mi,(method,py,pp) in enumerate(zip(("frozen","cumulative_refit"),predictions,probabilities)):
                  pred_stack[mi,batch-1]=py;prob_stack[mi,batch-1]=pp
                  record={"task":task,"seed":seed,"condition":condition,"delay":delay,"method":method,
                    "batch":batch,"n":BATCH_SIZE,"available_monitoring_labels":budget,
                    "training_rows":4000 if method=="frozen" else audit["training_n"],
                    "fit_count":1 if method=="frozen" else audit["fit_count"],
                    "monitor_id_sha256":digest_ids([]) if method=="frozen" else audit["monitor_id_sha256"],
                    "max_available_at":None if method=="frozen" else audit["max_available_at"],
                    "prediction_persisted":True,"probability_mean":float(pp.mean()),
                    "predicted_positive_rate":float(py.mean()),
                    "prediction_id_sha256":hashlib.sha256(np.asarray(ids,dtype="<i8").tobytes()).hexdigest()}
                  ow.writerow(record)
                of.flush();os.fsync(of.fileno())
                # Prediction log is durable before same-clock releases, including delay zero.
                due=broker.release(batch)
                for event in due:
                  if event.source_batch in released_batches: raise AssertionError("duplicate release")
                  released_batches.add(event.source_batch)
                  rw.writerow({"task":task,"seed":seed,"condition":condition,"delay":delay,
                    "source_batch":event.source_batch,"available_at":event.available_at,
                    "release_clock":batch,"queue_draining":False,"n":len(event.ids),
                    "ids_sha256":digest_ids(event.ids)})
                  adapter.add_released([event])
              for clock in range(BATCHES+1,BATCHES+delay+1):
                for event in broker.release(clock):
                  if event.source_batch in released_batches: raise AssertionError("duplicate drained release")
                  released_batches.add(event.source_batch)
                  rw.writerow({"task":task,"seed":seed,"condition":condition,"delay":delay,
                    "source_batch":event.source_batch,"available_at":event.available_at,
                    "release_clock":clock,"queue_draining":True,"n":len(event.ids),
                    "ids_sha256":digest_ids(event.ids)})
              if released_batches!=set(range(1,BATCHES+1)): raise AssertionError("queue failed to drain")
              if len(broker.released_batches)!=BATCHES: raise AssertionError("release count mismatch")
              all_predictions[delay]=pred_stack;all_probabilities[delay]=prob_stack
              delay_streams+=1
            # Compact cache is evaluator support, written outside the online interface.
            cache_path=out/"predictions"/("original" if task=="v1_synthetic" else "xor")/f"{seed}.npz"
            cache_path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(cache_path,ids=monitor.ids,
                **{f"pred_d{d}":a for d,a in all_predictions.items()},
                **{f"prob_d{d}":a for d,a in all_probabilities.items()})
            for delay in DELAYS:
              truth=monitor.y.reshape(BATCHES,BATCH_SIZE)
              pred=all_predictions[delay]
              for mi,method in enumerate(("frozen","cumulative_refit")):
                for batch in range(BATCHES):
                  cm=confusion_matrix(truth[batch],pred[mi,batch],labels=(0,1));tn,fp,fn,tp=(int(x) for x in cm.ravel())
                  macro=float(np.mean([2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else np.nan,
                                       2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan]))
                  correct=int((truth[batch]==pred[mi,batch]).sum())
                  ew.writerow({"task":task,"seed":seed,"condition":condition,"delay":delay,
                    "method":method,"batch":batch+1,"n":BATCH_SIZE,"correct":correct,
                    "accuracy":correct/BATCH_SIZE,"macro_f1":macro,
                    "class0_n":int((truth[batch]==0).sum()),"class1_n":int((truth[batch]==1).sum())})
            ef.flush();rf.flush();feature_streams+=1
            if index%25==0 or index==len(seeds):
              print(f"{task} seed {seed} ({index}/{len(seeds)}): {time.perf_counter()-t0:.1f}s; elapsed {time.perf_counter()-start:.1f}s",flush=True)
      lock.update(status="complete",completed_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-start,feature_streams=feature_streams,
        delay_streams=delay_streams,online_batch_records=len(seeds)*2*BATCHES*len(TASK_CONDITION)*len(DELAYS),
        evaluator_batch_records=len(seeds)*2*BATCHES*len(TASK_CONDITION)*len(DELAYS))
      (out/"protocol_lock.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")
    except BaseException as exc:
      lock.update(status="failed",failed_at_utc=datetime.now(timezone.utc).isoformat(),
        elapsed_seconds=time.perf_counter()-start,error=repr(exc))
      (out/"protocol_lock.json").write_text(json.dumps(lock,indent=2),encoding="utf-8")
      raise


if __name__=="__main__":
  p=argparse.ArgumentParser(description=__doc__);p.add_argument("--mode",choices=("development","final"),required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();run(a.mode,a.out)
