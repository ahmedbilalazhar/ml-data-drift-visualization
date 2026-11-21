"""Fit frozen development predictors and summarize completed stress-test blocks."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from run_future_harm_unseen_family import FEATURES, DEV, TRAIN_RUN
from run_phase3_extended_shift_stress import LAWS, DELAYS

ROOT=Path(__file__).resolve().parents[1]

def models():
    x=pd.read_csv(TRAIN_RUN/"predictions/online_features.csv")
    y=pd.read_csv(TRAIN_RUN/"evaluator_truth/future_targets.csv")
    x=x[x.seed.isin(DEV)];y=y[y.seed.isin(DEV)]
    train=x.merge(y,on=["seed","condition","delay","batch"],validate="one_to_one")
    fitted={}
    with threadpool_limits(limits=1):
        for name,cols in FEATURES.items():
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=2026))
            fitted[name]=model.fit(train[cols].replace([np.inf,-np.inf],np.nan).fillna(-1),train.future_harm)
    return fitted

def main(blocks,out):
    out.mkdir(parents=True,exist_ok=False)
    xs=[];ys=[];locks=[]
    for path in blocks:
        lock=json.loads((path/"protocol_lock.json").read_text())
        xfile=path/"online_features.csv";yfile=path/"evaluator_targets.csv"
        if lock["status"]!="complete" or not xfile.exists() or not yfile.exists():
            raise ValueError(f"Incomplete block {path}")
        xs.append(pd.read_csv(xfile));ys.append(pd.read_csv(yfile));locks.append(lock)
    x=pd.concat(xs,ignore_index=True);y=pd.concat(ys,ignore_index=True)
    keys=["seed","onset","law","delay","origin"]
    if x.duplicated(keys).any() or y.duplicated(keys).any():raise ValueError("Duplicate cells")
    d=x.merge(y,on=keys,validate="one_to_one")
    if len(d)!=len(x) or len(d)!=len(y):raise ValueError("Feature/target keys differ")
    expected=len(set(d.seed))*len(LAWS)*len(DELAYS)*19
    if len(d)!=expected:raise ValueError(f"Missing cells: {len(d)} versus {expected}")
    if set(d.seed)&set(DEV):raise ValueError("Train/test seed overlap")
    if not d.groupby("seed").onset.nunique().eq(1).all():raise ValueError("Onset assignment changed within seed")
    # Reconstruct released-label features from outcomes that are available under
    # the strict source-batch + delay < forecast-origin rule.
    d=d.sort_values(keys).reset_index(drop=True)
    d["released_harm_fraction"]=np.nan;d["latest_age"]=25
    for _,idx in d.groupby(["seed","law","delay"],sort=False).groups.items():
        ix=list(idx)
        for row in ix:
            t=int(d.at[row,"origin"]);delay=int(d.at[row,"delay"])
            released=[j for j in ix if int(d.at[j,"origin"])+delay<t]
            if released:
                tail=released[-5:]
                d.at[row,"rolling_accuracy"]=float(d.loc[tail,"accuracy"].mean())
                d.at[row,"released_harm_fraction"]=float((d.loc[tail,"accuracy"] < d.loc[tail,"reference_accuracy"]-.05).mean())
                d.at[row,"latest_age"]=t-int(d.at[released[-1],"origin"])
    # Exact identical X-monitor forecasts are required for conditional-only laws.
    fields=["estimate","concern","alarm_count","min_adjusted_p","input_alarm"]
    base=d[d.law.eq("stationary")][keys[:2]+["delay","origin"]+fields].set_index(["seed","onset","delay","origin"])
    identity=[]
    for law in ("flatten_035","flatten_070","flatten_100","regional_neg075","regional_zero","regional_pos075"):
        other=d[d.law.eq(law)].set_index(["seed","onset","delay","origin"])
        common=base.index.intersection(other.index)
        equal=base.loc[common,fields].equals(other.loc[common,fields])
        identity.append(dict(law=law,matched_cells=len(common),monitor_fields_exactly_identical=equal))
        if not equal:raise AssertionError(f"Inputs changed in conditional-only law {law}")
    fitted=models()
    for name,cols in FEATURES.items():
        q=d[cols].replace([np.inf,-np.inf],np.nan).fillna(-1)
        d[name]=fitted[name].predict_proba(q)[:,1]
    prevalence=float(pd.read_csv(TRAIN_RUN/"evaluator_truth/future_targets.csv").query("seed in @DEV").future_harm.mean())
    d["development_prevalence"]=prevalence
    d["last_released_harm"]=((d.latest_accuracy < d.reference_accuracy-.05).astype(float)).where(d.released_n.gt(0),prevalence)
    models_all=["time_only","unlabeled_monitor","full_released","development_prevalence","last_released_harm"]
    seed_loss=d.assign(**{f"{m}_loss":(d[m]-d.future_harm)**2 for m in models_all})
    grouped=seed_loss.groupby(["seed","law","delay"],sort=False)
    per_seed=grouped[[f"{m}_loss" for m in models_all]].mean().reset_index()
    summary=[];contrasts=[]
    for (law,delay),g in d.groupby(["law","delay"],sort=True):
        for model in models_all:
            yy=g.future_harm.to_numpy();pp=g[model].to_numpy()
            summary.append(dict(law=law,delay=delay,seeds=g.seed.nunique(),rows=len(g),prevalence=float(yy.mean()),
                brier=float(np.mean((pp-yy)**2)),auroc=(float(roc_auc_score(yy,pp)) if np.unique(yy).size==2 else None)))
    wide=per_seed.pivot(index="seed",columns=["law","delay"],values=[f"{m}_loss" for m in models_all])
    contrast_cols=[];contrast_names=[]
    for law in LAWS:
        for delay in DELAYS:
            for name,a,b in (("monitor_vs_time","time_only_loss","unlabeled_monitor_loss"),
                             ("released_vs_monitor","unlabeled_monitor_loss","full_released_loss")):
                col=wide[(a,law,delay)]-wide[(b,law,delay)]
                contrast_cols.append(col.to_numpy());contrast_names.append((name,law,delay))
    mat=np.column_stack(contrast_cols);means=mat.mean(axis=0);rng=np.random.default_rng(973999)
    boots=mat[rng.integers(0,len(mat),(2000,len(mat)))].mean(axis=1)
    width=float(np.quantile(np.max(np.abs(boots-means),axis=1),.95))
    for (name,law,delay),mu in zip(contrast_names,means):
        contrasts.append(dict(contrast=name,law=law,delay=delay,seed_count=len(mat),brier_gain=float(mu),
            simultaneous_low=float(mu-width),simultaneous_high=float(mu+width)))
    d[ keys+["accuracy","reference_accuracy","estimate","concern","alarm_count","input_alarm","released_n","released_harm_fraction","latest_age",
       "future_accuracy","future_harm"]+models_all].to_csv(out/"heldout_probabilities.csv",index=False)
    pd.DataFrame(summary).to_csv(out/"metrics_by_law_delay.csv",index=False)
    per_seed.to_csv(out/"paired_seed_losses.csv",index=False)
    pd.DataFrame(contrasts).to_csv(out/"simultaneous_contrasts.csv",index=False)
    receipt=dict(status="passed",seed_count=d.seed.nunique(),law_count=d.law.nunique(),delay_count=d.delay.nunique(),
       rows=len(d),complete_cells=True,training_seeds=list(DEV),test_seed_overlap=False,
       conditional_feature_identity=identity,source_blocks=[str(p) for p in blocks],
       block_locks=[lock["source_sha256"] for lock in locks])
    (out/"analysis_validation.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(json.dumps(receipt,indent=2),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--blocks",nargs="+",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();main(a.blocks,a.out)
