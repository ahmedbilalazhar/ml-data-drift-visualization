"""Matched-cap static, periodic and released-error-triggered adaptation audit."""
from __future__ import annotations
import argparse,hashlib,json,platform,sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy,sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from drift_monitoring.phase3.strengthening_generators import generate_historical,generate_monitoring  # noqa: E402
from drift_monitoring.phase3.replay import LabelBroker, release_feedback  # noqa: E402

BATCHES=96;BATCH_SIZE=500;PERIODIC_CLOCKS={25,49,73};MAX_UPDATES=3;MAX_LABEL_ROWS_PER_FIT=2000
METHODS=("frozen","periodic_refit","global_error_trigger_refit","region_error_trigger_refit")

def fit_model(seed:int,x:np.ndarray,y:np.ndarray)->LogisticRegression:
    if len(np.unique(y))!=2:raise ValueError("every refit must contain both classes")
    model=LogisticRegression(C=1.0,max_iter=1000,random_state=seed)
    model.fit(x,y)
    return model

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--config",type=Path,default=ROOT/"configs/phase3_strengthening_v3.json")
 p.add_argument("--output",type=Path,default=ROOT/"results/phase3/adaptation_schedule_audit_20261118_v4")
 p.add_argument("--seed-limit",type=int,default=None,help="smoke-test only; default uses the full locked bank")
 a=p.parse_args();a.config=a.config.resolve();a.output=a.output.resolve()
 if a.output.exists():raise FileExistsError(f"refusing to overwrite {a.output}")
 cfg=json.loads(a.config.read_text(encoding="utf-8"));seeds=list(map(int,cfg["seed_banks"]["evaluation"]))
 if a.seed_limit is not None:
  if a.seed_limit<1:raise ValueError("seed limit must be positive")
  seeds=seeds[:a.seed_limit]
 train_n=int(cfg["history"]["train_rows"]);cal_n=int(cfg["history"]["calibration_rows"])
 batch_rows=[];run_rows=[];release_rows=[];delays=(0,5,20);mechanisms=("M0_stationary","M4_abrupt_conditional_reversal")
 mech_code={m:i for i,m in enumerate(mechanisms)}
 for seed in seeds:
  train=generate_historical("linear_probabilistic",seed,1,train_n,8)
  calibration=generate_historical("linear_probabilistic",seed,2,cal_n,8)
  scaler=StandardScaler().fit(train.x);tx=scaler.transform(train.x);cx=scaler.transform(calibration.x)
  base_model=fit_model(seed,tx,train.y)
  base_error=float(np.mean(base_model.predict(cx)!=calibration.y))
  regioner=MiniBatchKMeans(n_clusters=3,n_init=3,batch_size=256,random_state=seed).fit(tx)
  train_regions=regioner.predict(tx);cal_regions=regioner.predict(cx)
  train_region_error={r:float(np.mean(base_model.predict(cx[cal_regions==r])!=calibration.y[cal_regions==r]))
                      for r in range(3) if int(np.sum(cal_regions==r))>=20}
  for mechanism in mechanisms:
   stream=generate_monitoring("linear_probabilistic",mechanism,seed,feature_count=8)
   mx=scaler.transform(stream.monitor.x);my=stream.evaluator.labels;ids=stream.monitor.ids
   monitor_regions=regioner.predict(mx)
   for delay in delays:
    states={}
    for method in METHODS:
      states[method]={"model":fit_model(seed,tx,train.y),"broker":LabelBroker(ids,my,BATCH_SIZE,delay),
       "released_x":[],"released_y":[],"released_ids":[],
       "released_sources":[],"released_errors":[],"predictions":{},"fit_count":1,"update_count":0,
       "fit_rows_total":len(train.y),"last_update_clock":0,"last_trigger_source":0,"last_available_source":0,
       "last_forecast_available_source":0,
       "local_models":{},"local_released_x":{r:[] for r in range(3)},"local_released_y":{r:[] for r in range(3)},
       "local_errors":{r:[] for r in range(3)},"local_last_trigger_source":{r:0 for r in range(3)}}
    for batch in range(1,BATCHES+1):
     lo=(batch-1)*BATCH_SIZE;hi=lo+BATCH_SIZE
     for method,state in states.items():
      # Release only at the previous clock: source+delay <= batch-1, so no
      # label from the current forecast cohort can affect its own prediction.
      feedback=release_feedback(state["broker"],mx,ids,clock=batch-1)
      for item in feedback:
       source=item.source_batch;slo=(source-1)*BATCH_SIZE;shi=slo+BATCH_SIZE
       if source not in state["predictions"]:
        raise RuntimeError(f"released cohort {source} has no persisted prediction")
       state["released_x"].append(item.features);state["released_y"].append(item.labels.copy())
       state["released_ids"].append(item.ids.copy());state["released_sources"].append(source)
       state["last_available_source"]=max(state["last_available_source"],source)
       error=float(np.mean(state["predictions"][source]!=item.labels))
       state["released_errors"].append((source,error))
       release_rows.append({"seed":seed,"mechanism":mechanism,"delay":delay,"method":method,
        "source_batch":source,"available_at":item.available_at,"release_processed_clock":batch-1,
        "prediction_clock":batch,"rows":len(item.labels),
        "ids_sha256":hashlib.sha256(np.asarray(item.ids).tobytes()).hexdigest(),"queue_drain":False})
       if method=="region_error_trigger_refit":
        source_region=monitor_regions[slo:shi];source_prediction=state["predictions"][source]
        for region in range(3):
         mask=source_region==region
         if int(mask.sum())>=20:
          state["local_errors"][region].append((source,float(np.mean(source_prediction[mask]!=my[slo:shi][mask]))))
          state["local_released_x"][region].append(mx[slo:shi][mask])
          state["local_released_y"][region].append(my[slo:shi][mask].copy())
      update_reason="none";target_region=None
      if method=="periodic_refit" and batch in PERIODIC_CLOCKS:update_reason="periodic_clock"
      elif method=="global_error_trigger_refit" and state["update_count"]<MAX_UPDATES:
       recent=state["released_errors"][-2:]
       if (len(recent)==2 and all(err>base_error+.10 for _,err in recent)
           and recent[-2][0]>state["last_trigger_source"]
           and batch-state["last_update_clock"]>=8):update_reason="two_released_cohorts_above_calibration_error"
      elif method=="region_error_trigger_refit" and state["update_count"]<MAX_UPDATES and batch-state["last_update_clock"]>=8:
       candidates=[]
       for region,history in state["local_errors"].items():
        recent=history[-2:]
        if (region in train_region_error and len(recent)==2
            and all(err>train_region_error[region]+.10 for _,err in recent)
            and recent[-2][0]>state["local_last_trigger_source"][region]):
         candidates.append((recent[-1][1]-train_region_error[region],region,recent[-1][0]))
       if candidates:
        _,target_region,trigger_source=max(candidates)
        update_reason=f"region_{target_region}_two_released_cohorts_above_region_calibration_error"
       else:target_region=None
      update_n=0
      should_update=(method!="frozen" and update_reason!="none" and state["update_count"]<MAX_UPDATES
                     and state["released_y"])
      if should_update:
       if method=="region_error_trigger_refit":
        local_x=state["local_released_x"][target_region]
        local_y=state["local_released_y"][target_region]
        train_mask=train_regions==target_region
        if local_y and int(train_mask.sum())>=20:
         all_ry=np.concatenate(local_y);all_rx=np.vstack(local_x)
         update_n=min(MAX_LABEL_ROWS_PER_FIT,len(all_ry));rx=all_rx[-update_n:];ry=all_ry[-update_n:]
         fit_x=np.vstack([tx[train_mask],rx]);fit_y=np.r_[train.y[train_mask],ry]
         # A region refit needs both classes in its own training and released evidence.
         if len(np.unique(fit_y))==2:
          state["local_models"][target_region]=fit_model(seed,fit_x,fit_y)
          state["fit_count"]+=1;state["update_count"]+=1;state["fit_rows_total"]+=len(fit_y)
          state["last_update_clock"]=batch
          state["local_last_trigger_source"][target_region]=trigger_source
         else:update_reason+="_skipped_single_class_region"
        else:update_reason+="_skipped_insufficient_region_data"
       else:
        all_rx=np.concatenate(state["released_x"]);all_ry=np.concatenate(state["released_y"])
        # Both updating arms use the same recent-label cap at each fit.
        update_n=min(MAX_LABEL_ROWS_PER_FIT,len(all_ry));rx=all_rx[-update_n:];ry=all_ry[-update_n:]
        state["model"]=fit_model(seed, np.vstack([tx,rx]),np.r_[train.y,ry])
        state["fit_count"]+=1;state["update_count"]+=1;state["fit_rows_total"]+=len(train.y)+update_n
        state["last_update_clock"]=batch
        if method=="global_error_trigger_refit":state["last_trigger_source"]=state["released_errors"][-1][0]
      elif update_reason!="none":update_reason+="_skipped_no_due_labels_or_cap"
      prediction=state["model"].predict(mx[lo:hi]).astype(np.int8)
      if method=="region_error_trigger_refit":
       batch_regions=monitor_regions[lo:hi]
       for region,local_model in state["local_models"].items():
        local_mask=batch_regions==region
        if local_mask.any():prediction[local_mask]=local_model.predict(mx[lo:hi][local_mask]).astype(np.int8)
      state["predictions"][batch]=prediction
      batch_rows.append({"seed":seed,"mechanism":mechanism,"delay":delay,"method":method,"batch":batch,
       "prediction_persisted":True,"available_source_through":state["last_available_source"],
       "max_available_at":state["last_available_source"]+delay if state["last_available_source"] else None,
       "available_label_rows":len(state["released_ids"])*BATCH_SIZE,"fit_count":state["fit_count"],
       "updates":state["update_count"],"training_rows_at_forecast":len(train.y)+min(MAX_LABEL_ROWS_PER_FIT,len(np.concatenate(state["released_y"]))) if state["released_y"] else len(train.y),
       "fit_row_cost_cumulative":state["fit_rows_total"],"update_row_count":update_n,"update_reason":update_reason})
    for state in states.values():
     state["last_forecast_available_source"]=state["last_available_source"]
    for method,state in states.items():
     # Drain delayed cohorts after the final prediction. Drain events create no
     # additional prediction/adaptation opportunity.
     for item in release_feedback(state["broker"],mx,ids,clock=BATCHES+delay):
      source=item.source_batch;slo=(source-1)*BATCH_SIZE;shi=slo+BATCH_SIZE
      if source not in state["predictions"]:
       raise RuntimeError(f"drained cohort {source} has no persisted prediction")
      state["released_x"].append(item.features);state["released_y"].append(item.labels.copy())
      state["released_ids"].append(item.ids.copy());state["released_sources"].append(source)
      state["last_available_source"]=max(state["last_available_source"],source)
      release_rows.append({"seed":seed,"mechanism":mechanism,"delay":delay,"method":method,
       "source_batch":source,"available_at":item.available_at,"release_processed_clock":BATCHES+delay,
       "prediction_clock":None,"rows":len(item.labels),
       "ids_sha256":hashlib.sha256(np.asarray(item.ids).tobytes()).hexdigest(),"queue_drain":True})
     if state["last_available_source"]!=BATCHES:
      raise RuntimeError(f"label queue did not drain for {method}, delay {delay}")
     pred=np.concatenate([state["predictions"][b] for b in range(1,BATCHES+1)])
     correct=(pred==my).astype(np.int8)
     post=correct[24*BATCH_SIZE:]
     tp=int(np.sum((pred==1)&(my==1)));fp=int(np.sum((pred==1)&(my==0)));fn=int(np.sum((pred==0)&(my==1)))
     run_rows.append({"seed":seed,"task":"linear_probabilistic","mechanism":mechanism,"delay":delay,"method":method,
      "accuracy":float(correct.mean()),"post_onset_accuracy":float(post.mean()),
      "macro_f1":float((2*tp/max(1,2*tp+fp+fn)+2*(int(np.sum((pred==0)&(my==0))))/max(1,2*int(np.sum((pred==0)&(my==0)))+fp+fn))/2),
      "balanced_accuracy":float(((tp/max(1,tp+fn))+(int(np.sum((pred==0)&(my==0)))/max(1,int(np.sum((pred==0)&(my==0)))+fp)))/2),
      "fit_count":state["fit_count"],"update_count":state["update_count"],
       "released_labels_by_last_forecast":state["last_forecast_available_source"]*BATCH_SIZE,
       "released_labels_after_queue_drain":state["last_available_source"]*BATCH_SIZE,
       "fit_row_cost_cumulative":state["fit_rows_total"],"historical_calibration_error_threshold":base_error+.10,
       "local_models_active":len(state["local_models"]),"local_regions_updated":json.dumps(sorted(state["local_models"]))})
 a.output.mkdir(parents=True);batch=pd.DataFrame(batch_rows);runs=pd.DataFrame(run_rows)
 batch.to_csv(a.output/"online_batch_audit.csv",index=False);runs.to_csv(a.output/"seed_metrics.csv",index=False)
 releases=pd.DataFrame(release_rows);releases.to_csv(a.output/"label_release_events.csv",index=False)
 # Paired-seed inference; no batch is treated as an independent replicate.
 rng=np.random.default_rng(261117);summaries=[]
 for (mechanism,delay,method),g in runs.groupby(["mechanism","delay","method"],sort=True):
  for metric in ["accuracy","post_onset_accuracy","macro_f1","balanced_accuracy","update_count","fit_row_cost_cumulative"]:
   v=g.groupby("seed")[metric].mean().to_numpy(float)
   # A one-seed smoke run is an execution check, not an inferential result.
   if len(v)>=2:
    boot=v[rng.integers(0,len(v),(5000,len(v)))].mean(axis=1)
    lower=float(np.quantile(boot,.025));upper=float(np.quantile(boot,.975));seed_sd=float(v.std(ddof=1))
    inference_status="paired_seed_bootstrap"
   else:
    lower=upper=seed_sd=float("nan");inference_status="insufficient_seeds_smoke_only"
   summaries.append({"mechanism":mechanism,"delay":delay,"method":method,"metric":metric,"seeds":len(v),
    "mean":float(v.mean()),"lower":lower,"upper":upper,"seed_sd":seed_sd,"inference_status":inference_status})
 summary=pd.DataFrame(summaries);summary.to_csv(a.output/"seed_bootstrap_summary.csv",index=False)
 sources=[Path(__file__),a.config,ROOT/"src/drift_monitoring/phase3/strengthening_generators.py",
          ROOT/"src/drift_monitoring/phase3/replay.py"]
 outputs=[a.output/f for f in ["online_batch_audit.csv","seed_metrics.csv","seed_bootstrap_summary.csv","label_release_events.csv"]]
 sha=lambda q:hashlib.sha256(q.read_bytes()).hexdigest()
 receipt={"status":"complete","protocol":"matched-cap-global-and-region-local-released-error-trigger-adaptation-v4-labelbroker-drain-accounting",
  "seed_bank":seeds,"task":"linear_probabilistic","mechanisms":list(mechanisms),"delays":list(delays),"batch_size":BATCH_SIZE,
  "periodic_clocks":sorted(PERIODIC_CLOCKS),"max_update_fits":MAX_UPDATES,"max_recent_label_rows_per_fit":MAX_LABEL_ROWS_PER_FIT,
  "global_trigger":"both most recent released source cohorts exceed calibration error + 0.10; threshold uses calibration only; 8-clock cooldown",
  "label_clock":"LabelBroker releases are consumed at clock forecast_batch-1 before scoring; source_batch + delay <= release_processed_clock; same-clock labels are unavailable before scoring; post-stream queue drain is logged without an adaptation opportunity",
  "label_release_event_rows":len(releases),"label_release_event_sha256":sha(a.output/"label_release_events.csv"),
  "release_accounting":"released_labels_by_last_forecast excludes labels delivered only during post-stream queue drain; released_labels_after_queue_drain records the full evaluator accounting total separately",
  "training_data":"frozen training scaler; initial training plus the most recent at most 2000 released rows per update",
  "monitoring_labels":"used for outcomes only after each prediction and for detector input only at scheduled release",
  "adaptation_scope":"global trigger and frozen-training-KMeans three-region local trigger; local error baselines use calibration rows only, require at least 20 calibration rows per region and at least 20 rows in each released cohort, trigger on two consecutive eligible cohorts exceeding region calibration error by 0.10, with 8-clock global cooldown and at most three total updates; each local refit uses only that region's historical training rows plus at most 2000 most recent released labels for that region, and routes other regions to the global frozen fallback",
  "seed_bootstrap_replicates":5000,"bootstrap_unit":"seed; batch rows aggregated within each seed",
  "seed_bootstrap_inference_valid":len(seeds)>=2,
  "source_sha256":{q.relative_to(ROOT).as_posix():sha(q) for q in sources},
  "output_sha256":{q.relative_to(ROOT).as_posix():sha(q) for q in outputs},
  "environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__}}
 (a.output/"audit_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
 print(summary[summary.metric.isin(["accuracy","post_onset_accuracy","update_count","fit_row_cost_cumulative"])].to_string(index=False))
 print(f"Completed {len(batch):,} forecast rows and {len(runs):,} seed-policy cells in {a.output}")

if __name__=="__main__":main()
