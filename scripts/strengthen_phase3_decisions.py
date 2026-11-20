"""Outcome-informed, equal-label-budget audit policies on preserved cached streams.

Not an adaptation experiment: actions request labels, never refit the model.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
POLICIES=('uniform','input_only','concern_only','combined','released_label')

def audit_schedule(scores, threshold, budget, delay, released_accuracy=None, reference_accuracy=None):
    """Choose before current-clock release; equal cost via explicit tail fill.

    The label provider can only be called for previously selected batches whose
    release clock is strictly before the current decision clock.
    """
    scores=np.asarray(scores,float)
    if len(scores)!=24 or budget not in (2,4) or delay not in (0,5,20):
        raise ValueError('unsupported locked audit configuration')
    selected=[]; decisions=[]; known=[]
    for clock in range(1,25):
        for batch in selected:
            if batch+delay<clock and batch not in known: known.append(batch)
        value=float(scores[clock-1])
        if released_accuracy is not None:
            value=max((reference_accuracy-released_accuracy(b) for b in known),default=-np.inf)
        remaining=budget-len(selected)
        forced=remaining>0 and (25-clock)<=remaining
        probe=released_accuracy is not None and clock==int(np.ceil(24/budget))
        flag=remaining>0 and (value>=threshold or forced or probe)
        if flag:selected.append(clock)
        decisions.append(dict(batch=clock,selected=flag,score=value if np.isfinite(value) else None,
                              reason='initial_probe' if probe else ('quota_tail_fill' if forced else ('signal' if flag else 'none')),
                              prior_released_selected_batches=json.dumps(known),
                              selected_labels_available_at=clock+delay if flag else None))
    assert len(selected)==budget
    return decisions

def percentile(values, reference):
    return np.searchsorted(np.sort(reference),values,side='right')/len(reference)

def main(out):
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    dev=pd.concat([pd.read_csv(ROOT/f'results/phase3/{name}/online_batches.csv')
                   for name in ('development_v1','development_remaining_v1')],ignore_index=True)
    if dev.duplicated(['seed','condition','batch']).any():raise ValueError('duplicate development cohorts')
    if set(dev.seed)!={90001,90002,90003,90004,90005}:raise ValueError('development seeds differ from protocol')
    final=pd.read_csv(ROOT/'results/phase3/final_v1/online_batches.csv')
    truth=pd.read_csv(ROOT/'results/phase3/final_v1/released_verification.csv')
    truth=truth[truth.delay==0].set_index('cohort',verify_integrity=True)
    assert not set(dev.seed)&set(final.seed)
    for d in (dev,final):
        d['input_score']=-np.log10(np.maximum(d.min_adjusted_p,1e-300))
        d['combined_score']=np.minimum(percentile(d.input_score,dev.input_score),
                                      percentile(d.cal_concern,dev.cal_concern))
    thresholds={}
    for budget in (2,4):
        thresholds[str(budget)]={name:float(np.quantile(dev[col],1-budget/24,method='higher'))
            for name,col in [('input_only','input_score'),('concern_only','cal_concern'),('combined','combined_score')]}
    lock=dict(study='equal-budget label-audit extension',outcome_informed=True,
        development_seeds=sorted(map(int,dev.seed.unique())),final_seeds=sorted(map(int,final.seed.unique())),
        budget_batches=[2,4],label_cost_per_audit=500,delays=[0,5,20],thresholds=thresholds,
        policy_rules='threshold crossing with fixed quota; force remaining audits in final remaining slots; no future ranking',
        released_label_rule='initial probe at ceil(24/budget); then max observed reference-minus-accuracy over prior selected labels; threshold .05; no current-clock release',
        combined_rule='minimum of development empirical percentiles of input and concern scores (conjunction)',
        uniform_rule='evenly spaced fixed batch positions',
        missing_claims=['no accuracy improvement from audits','no human decision benefit','no future-horizon forecast'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'protocol_lock.json').write_text(json.dumps(lock,indent=2))
    rows=[];logs=[]
    for (seed,condition),g in final.groupby(['seed','condition']):
        g=g.sort_values('batch');t=truth.loc[g.cohort];harm=t.harm.to_numpy(bool);accuracy=t.accuracy.to_numpy(float)
        for delay in (0,5,20):
            for budget in (2,4):
                for policy in POLICIES:
                    provider=None;ref=None
                    if policy=='uniform':
                        scores=np.zeros(24);scores[np.ceil(np.arange(1,budget+1)*24/budget).astype(int)-1]=1;threshold=.5
                    elif policy=='released_label':
                        scores=np.zeros(24);threshold=.05;provider=lambda b:float(accuracy[b-1]);ref=float(g.reference_accuracy.iloc[0])
                    else:
                        col={'input_only':'input_score','concern_only':'cal_concern','combined':'combined_score'}[policy]
                        scores=g[col].to_numpy();threshold=thresholds[str(budget)][policy]
                    decisions=audit_schedule(scores,threshold,budget,delay,provider,ref)
                    selected=np.array([r['selected'] for r in decisions]);tp=int((selected&harm).sum());hn=int(harm.sum())
                    confirmed=[r['batch']+delay for r in decisions if r['selected'] and harm[r['batch']-1]]
                    first_harm=np.flatnonzero(harm)
                    first_confirmation=min(confirmed) if confirmed else np.nan
                    rows.append(dict(seed=seed,condition=condition,delay=delay,budget=budget,policy=policy,
                        audited_batches=int(selected.sum()),labels_requested=500*budget,harmful_batches=hn,
                        harmful_audits=tp,unnecessary_audits=budget-tp,missed_harmful_batches=hn-tp,
                        audit_precision=tp/budget,harm_recall=tp/hn if hn else np.nan,
                        harmful_confirmations_during_input=sum(c<=24 for c in confirmed),
                        first_harm_confirmation_clock=first_confirmation,
                        first_confirmation_latency=first_confirmation-int(first_harm[0]+1) if len(first_harm) else np.nan,
                        quota_fill_audits=sum(r['selected'] and r['reason']=='quota_tail_fill' for r in decisions)))
                    logs.extend(dict(seed=int(seed),condition=condition,delay=delay,budget=budget,policy=policy,**r) for r in decisions)
    data=pd.DataFrame(rows);data.to_csv(out/'policy_per_seed.csv',index=False)
    pd.DataFrame(logs).to_csv(out/'online_audit_decisions.csv',index=False)
    summary=data.groupby(['condition','delay','budget','policy']).mean(numeric_only=True).reset_index()
    summary.to_csv(out/'policy_summary.csv',index=False)
    differences=[];rng=np.random.default_rng(160112026)
    for keys,g in data.groupby(['condition','delay','budget']):
        base=g[g.policy=='uniform'].set_index('seed')
        for policy in POLICIES[1:]:
            other=g[g.policy==policy].set_index('seed').loc[base.index]
            values=other.harmful_audits.to_numpy()-base.harmful_audits.to_numpy()
            boot=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(axis=1)
            differences.append(dict(condition=keys[0],delay=keys[1],budget=keys[2],policy=policy,
                metric='harmful_audits_minus_uniform',mean=float(values.mean()),lower=float(np.quantile(boot,.025)),upper=float(np.quantile(boot,.975)),paired_seeds=len(values)))
    pd.DataFrame(differences).to_csv(out/'paired_bootstrap.csv',index=False)
    # Existing localization metrics already have valid ground truth for C1/C2.
    loc=pd.read_csv(ROOT/'results/phase3/final_v1/localization_batches.csv')
    loc=loc[(loc.changed_marginal_n>0)]
    loc.groupby(['seed','condition'])[['precision_at_1','recall_at_3','feature_flag_recall']].mean().reset_index().to_csv(out/'localization_per_seed.csv',index=False)
    loc.groupby('condition')[['precision_at_1','recall_at_3','feature_flag_recall']].mean().reset_index().to_csv(out/'localization_summary.csv',index=False)
    (out/'validation.json').write_text(json.dumps(dict(status='passed',policy_rows=len(data),
        identical_label_budget=bool((data.labels_requested==data.budget*500).all()),
        decision_rows=len(logs),source_results_unchanged=True,
        localization_scope='C1/C2 original marginal interventions only; no causal or dependence localization claim'),indent=2))
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
