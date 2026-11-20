"""Block-bootstrap reference and isotonic-calibrator uncertainty for NOAA."""
from __future__ import annotations
import argparse, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy, sklearn
from sklearn.isotonic import IsotonicRegression

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"src"))
from drift_monitoring.phase3.core import Partition, fit_artifacts, impact  # noqa: E402
from scripts.run_noaa_weather_transfer import FEATURES, split_counts  # noqa: E402

def moving_block_indices(n:int, block:int, rng:np.random.Generator)->np.ndarray:
    """Noncircular moving-block bootstrap index with an exact requested length."""
    if n<1 or block<1: raise ValueError("n and block must be positive")
    starts=rng.integers(0,max(1,n-block+1),size=(n+block-1)//block)
    return np.concatenate([np.arange(s,min(s+block,n)) for s in starts])[:n]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,default=ROOT/"data/processed/weather_noaa_1949_1999_v2/weather_ordered.csv")
    p.add_argument("--manifest",type=Path,default=ROOT/"data/processed/weather_noaa_1949_1999_v2/weather_manifest.json")
    p.add_argument("--output",type=Path,default=ROOT/"results/phase3/noaa_reference_calibration_uncertainty_20261117_v2")
    p.add_argument("--replicates",type=int,default=200)
    p.add_argument("--block-length",type=int,default=30)
    a=p.parse_args(); a.data=a.data.resolve(); a.manifest=a.manifest.resolve(); a.output=a.output.resolve()
    if a.output.exists(): raise FileExistsError(f"refusing to overwrite {a.output}")
    manifest=json.loads(a.manifest.read_text(encoding="utf-8"))
    import hashlib as _h
    if _h.sha256(a.data.read_bytes()).hexdigest()!=manifest["processed"]["sha256"]: raise ValueError("data checksum mismatch")
    df=pd.read_csv(a.data,parse_dates=["timestamp"]); counts=split_counts(len(df))
    a0=counts["train"]; b0=a0+counts["calibration"]; c0=b0+counts["reference"]
    x=df.loc[:,FEATURES].to_numpy(float); y=df.label_rain.to_numpy(np.int8)
    part=lambda lo,hi:Partition(x[lo:hi],y[lo:hi],df.row_id.to_numpy(np.int64)[lo:hi])
    train,cal,ref=part(0,a0),part(a0,b0),part(b0,c0)
    fitted=fit_artifacts(0,train,cal,ref,n_pca_components=3,n_clusters=3,model_max_iter=2000)
    cp=fitted.model.predict_proba(fitted.scaler.transform(cal.x))[:,1]
    cal_conf=np.maximum(cp,1-cp); cal_correct=((cp>=.5)==cal.y).astype(np.int8)
    rp=fitted.model.predict_proba(fitted.scaler.transform(ref.x))[:,1]
    ref_correct=((rp>=.5)==ref.y).astype(np.int8)
    mon=df.iloc[c0:].reset_index(drop=True); n_batches=len(mon)//500; mon=mon.iloc[:n_batches*500]
    mp=fitted.model.predict_proba(fitted.scaler.transform(mon.loc[:,FEATURES].to_numpy(float)))[:,1]
    raw_conf=np.maximum(mp,1-mp); monitor_y=mon.label_rain.to_numpy(np.int8)
    pred=(mp>=.5).astype(np.int8); actual_correct=(pred==monitor_y).astype(np.int8)
    baseline_q=fitted.calibration.predict(raw_conf)
    baseline=[]
    for b in range(n_batches):
        sl=slice(b*500,(b+1)*500); res,_=impact(baseline_q[sl],int(ref_correct.sum()),len(ref_correct))
        baseline.append(res)
    rng=np.random.default_rng(261117); rows=[]
    modes=("calibrator_only","reference_only","joint")
    for mode in modes:
      for rep in range(a.replicates):
        if mode in ("calibrator_only","joint"):
            ci=moving_block_indices(len(cal_conf),a.block_length,rng)
            iso=IsotonicRegression(out_of_bounds="clip").fit(cal_conf[ci],cal_correct[ci])
            q=iso.predict(raw_conf)
        else: q=baseline_q
        if mode in ("reference_only","joint"):
            ri=moving_block_indices(len(ref_correct),a.block_length,rng)
            nr=int(ref_correct[ri].sum()); nn=len(ri)
        else: nr=int(ref_correct.sum()); nn=len(ref_correct)
        for b in range(n_batches):
            sl=slice(b*500,(b+1)*500); result,_=impact(q[sl],nr,nn)
            rows.append({"component":mode,"replicate":rep+1,"batch":b+1,
                         "bootstrap_accuracy_estimate":float(np.mean(q[sl])),
                         "concern_probability":result["concern"],"concern_flag":bool(result["concern_flag"]),
                         "plug_in_lower":result["lower"],"plug_in_upper":result["upper"],
                         "bootstrap_reference_accuracy":nr/nn,"reference_n":nn,
                         "verified_accuracy":float(actual_correct[sl].mean()),
                         "baseline_accuracy_estimate":baseline[b]["estimate"],
                         "baseline_concern_probability":baseline[b]["concern"],
                         "baseline_concern_flag":baseline[b]["concern_flag"]})
    a.output.mkdir(parents=True)
    frame=pd.DataFrame(rows); frame.to_csv(a.output/"bootstrap_draws.csv",index=False)
    summary=frame.groupby(["component","batch"],as_index=False).agg(
        draws=("replicate","nunique"),estimate_mean=("bootstrap_accuracy_estimate","mean"),
        estimate_lower=("bootstrap_accuracy_estimate",lambda v:float(np.quantile(v,.025))),
        estimate_upper=("bootstrap_accuracy_estimate",lambda v:float(np.quantile(v,.975))),
        concern_mean=("concern_probability","mean"),concern_lower=("concern_probability",lambda v:float(np.quantile(v,.025))),
        concern_upper=("concern_probability",lambda v:float(np.quantile(v,.975))),
        flag_rate=("concern_flag","mean"),verified_accuracy=("verified_accuracy","first"),
        baseline_estimate=("baseline_accuracy_estimate","first"),baseline_concern=("baseline_concern_probability","first"),
        baseline_flag=("baseline_concern_flag","first"),reference_accuracy_sd=("bootstrap_reference_accuracy","std"))
    summary["concern_flag_disagreement_rate"]=np.where(summary.component.eq("joint"),
          np.where(summary.baseline_flag,1-summary.flag_rate,summary.flag_rate),np.nan)
    summary.to_csv(a.output/"uncertainty_summary.csv",index=False)
    sources=[Path(__file__),a.data,a.manifest,ROOT/"src/drift_monitoring/phase3/core.py",ROOT/"scripts/run_noaa_weather_transfer.py"]
    outputs=[a.output/"bootstrap_draws.csv",a.output/"uncertainty_summary.csv"]
    sha=lambda f:_h.sha256(f.read_bytes()).hexdigest()
    receipt={"status":"complete","protocol":"NOAA-fixed-classifier-moving-block-reference-calibrator-bootstrap-v1",
      "bootstrap_replicates_per_component":a.replicates,"block_length_rows":a.block_length,
      "components":list(modes),"classifier":"frozen original chronological training fit; its parameter uncertainty is not bootstrapped",
      "calibrator":"isotonic refit on resampled chronological calibration confidence/correctness pairs",
      "reference":"reference accuracy resampled from chronological correctness blocks",
      "monitoring_labels":"evaluator-only scoring and descriptive coverage comparisons; no fit or threshold selection",
      "dataset_scope":"one NOAA station and chronological same-day classification holdout; bootstrap is conditional, not external replication",
      "source_sha256":{f.relative_to(ROOT).as_posix():sha(f) for f in sources},
      "output_sha256":{f.relative_to(ROOT).as_posix():sha(f) for f in outputs},
      "environment":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"scikit_learn":sklearn.__version__}}
    (a.output/"audit_receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(summary.groupby("component").agg(estimate_width=("estimate_upper",lambda s:float((s-summary.loc[s.index,"estimate_lower"]).mean())),
      concern_width=("concern_upper",lambda s:float((s-summary.loc[s.index,"concern_lower"]).mean())),
      flag_disagreement=("concern_flag_disagreement_rate","mean")).to_string())
    print(f"Completed {len(frame)} batch-level block-bootstrap draws at {a.output}")

if __name__=="__main__": main()
