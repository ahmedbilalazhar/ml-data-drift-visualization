"""Retrospective evaluation only. Never import this module in a live monitor."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


def ratio(n, d):
    return n/d if d else np.nan


def flag_metrics(truth, flag, score=None):
    truth, flag = np.asarray(truth, bool), np.asarray(flag, bool)
    tp, fp = int((truth & flag).sum()), int((~truth & flag).sum())
    fn, tn = int((truth & ~flag).sum()), int((~truth & ~flag).sum())
    return dict(n=len(truth), harm_n=int(truth.sum()), flagged_n=int(flag.sum()),
                tp=tp, fp=fp, fn=fn, tn=tn, precision=ratio(tp,tp+fp), recall=ratio(tp,tp+fn),
                false_flag_rate=ratio(fp,fp+tn), harmful_unflagged_fraction=ratio(fn,tp+fn),
                flagged_fraction=ratio(tp+fp,len(truth)),
                harm_auroc=(float(roc_auc_score(truth,score)) if score is not None and
                            len(np.unique(truth))==2 else np.nan))


def adaptive_calibration_error(q, correctness, bins=20):
    order = np.argsort(q,kind='stable')
    return float(sum(len(ix)*abs(np.mean(q[ix])-np.mean(correctness[ix]))
                     for ix in np.array_split(order,min(bins,len(order))) if len(ix))/len(order))


def validate_run(out):
    out=Path(out)
    lock=json.loads((out/'protocol_lock.json').read_text())
    if lock['status'] != 'complete':
        raise ValueError('incomplete runs are not final evidence')
    online=pd.read_csv(out/'online_batches.csv')
    verified=pd.read_csv(out/'released_verification.csv')
    events=pd.read_csv(out/'clock_events.csv')
    expected=len(lock['seeds'])*7*24
    assert len(online)==expected and online.cohort.is_unique
    assert len(verified)==expected*3
    assert not {'accuracy','macro_f1','harm','correct','labels','y'}.intersection(online.columns)
    np.testing.assert_array_equal(online.cal_estimate,online.calibrated_ac_alone)
    assert not verified.duplicated(['cohort','delay']).any()
    np.testing.assert_array_equal(verified.available_at,verified.batch+verified.delay)
    assert set(verified.delay)=={0,5,20}
    assert set(verified.cohort)==set(online.cohort)
    for _,g in verified.groupby('cohort',sort=False):
        assert g.accuracy.nunique()==1 and g.harm.nunique()==1 and g.ids_sha256.nunique()==1
    for delay,g in events.groupby('delay'):
        assert len(g)==expected*2
        assert g[g.event=='prediction_persisted'].clock.max()==24
        assert g[g.event=='labels_verified'].clock.max()==24+delay
        for _, stream in g.groupby(['seed','condition'],sort=False):
            seq=list(zip(stream.clock,stream.phase))
            assert seq==sorted(seq)
    # Persisted arrays carry exact instance joins, not only batch counters.
    for (seed,condition), rows in online.groupby(['seed','condition'],sort=False):
        with np.load(out/'predictions'/f'{seed}_{condition}.npz') as p, \
             np.load(out/'evaluator_truth'/f'{seed}_{condition}.npz') as t:
            np.testing.assert_array_equal(p['ids'].ravel(),t['ids'])
            assert len(np.unique(t['ids']))==12000
            np.testing.assert_array_equal(p['prediction'],p['probability']>=.5)
            np.testing.assert_allclose(p['correctness_probability'].mean(axis=1),rows.cal_estimate,atol=1e-14)
            np.testing.assert_allclose(p['raw_confidence'].mean(axis=1),rows.raw_estimate,atol=1e-14)
            np.testing.assert_allclose(p['cal_pmf'].sum(axis=1),1,atol=1e-12)
            v=verified[(verified.seed==seed)&(verified.condition==condition)&(verified.delay==0)]
            counts=(p['prediction']==t['labels'].reshape(24,500)).sum(axis=1)
            np.testing.assert_array_equal(counts,v.correct)
    return dict(status='passed', streams=expected//24, unique_cohorts=expected,
                monitoring_rows=expected*500, delay_schedules=3,
                instance_id_joins_checked=expected*500, queue_drainage_checked=True,
                calibrated_ac_identity_checked=True, delay_truth_invariance_checked=True,
                online_truth_columns_absent=True)


def joined_batches(out):
    online=pd.read_csv(Path(out)/'online_batches.csv')
    v=pd.read_csv(Path(out)/'released_verification.csv')
    # Keep a single evaluator truth per cohort: delay copies are not replications.
    v=v[v.delay==0].drop(columns=['seed','condition','batch','delay'])
    return online.merge(v,on='cohort',validate='one_to_one')


def regime(condition,batch):
    if batch<=8:
        return 'pre'
    if condition=='C0':
        return 'stationary_late'
    if condition=='C5' and batch>=17:
        return 'recovered'
    if condition in ['C2','C4'] and batch<=12:
        return 'transition'
    return 'changed'


def impact_table(batches):
    rows=[]
    for (seed,condition),whole in batches.groupby(['seed','condition']):
        for regime_name,g in [('all',whole)]+list(whole.groupby('regime')):
            for method,estimate,prefix,flag in [
                ('IDM_calibrated_AC','cal_estimate','cal','cal_concern_flag'),
                ('calibrated_AC_alone','calibrated_ac_alone','cal','cal_concern_flag'),
                ('raw_AC_no_calibration','raw_estimate','raw','raw_concern_flag'),
                ('ATC_style','atc_estimate',None,None),
                ('constant_historical','constant_estimate',None,None),
                ('no_latent_same_impact','cal_estimate','cal','cal_concern_flag'),
                ('point_rule_no_interval','cal_estimate',None,'cal_point_flag')]:
                if flag is None:
                    flags=g.reference_accuracy-g[estimate]>.05
                    scores=g.reference_accuracy-g[estimate]
                else:
                    flags=g[flag]
                    scores=g[f'{prefix}_concern'] if prefix else g.reference_accuracy-g[estimate]
                m=flag_metrics(g.harm,flags,scores)
                row=dict(seed=seed,condition=condition,regime=regime_name,method=method,
                         mae=float(np.abs(g[estimate]-g.accuracy).mean()),
                         bias=float((g[estimate]-g.accuracy).mean()),
                         actual_accuracy=float(g.accuracy.mean()),
                         macro_f1=float(g.macro_f1.mean()),
                         class0_n=int(g.class0_n.sum()),class1_n=int(g.class1_n.sum()), **m)
                if prefix:
                    covered=(g[f'{prefix}_lower']<=g.accuracy)&(g.accuracy<=g[f'{prefix}_upper'])
                    row.update(covered_n=int(covered.sum()),coverage=float(covered.mean()),
                               interval_width=float((g[f'{prefix}_upper']-g[f'{prefix}_lower']).mean()))
                else:
                    row.update(covered_n=np.nan,coverage=np.nan,interval_width=np.nan)
                rows.append(row)
            for method,col in [('original_KS_harm_proxy','original_alarm'),('full_KS_harm_proxy','input_alarm')]:
                rows.append(dict(seed=seed,condition=condition,regime=regime_name,method=method,
                                 mae=np.nan,coverage=np.nan,interval_width=np.nan,
                                 **flag_metrics(g.harm,g[col],g[col].astype(float))))
    return pd.DataFrame(rows)


def detection_tables(batches):
    detection,localization=[],[]
    for (seed,condition),whole in batches.groupby(['seed','condition']):
        native=condition in ['C1','C2','C6']
        null=whole if condition=='C0' else whole[whole.batch<=8]
        for method,col in [('full_KS_PCA','input_alarm'),('original_KS_only','original_alarm')]:
            matched=whole[(whole.batch>=9)&(whole.batch<=12)&whole[col]]
            detection.append(dict(seed=seed,condition=condition,method=method,
                null_batches=len(null),null_alerts=int(null[col].sum()),
                null_alerts_per_1000=1000*null[col].mean(),
                total_alerts=int(whole[col].sum()),total_batches=len(whole),
                native_event_n=int(native),matched_events=int(native and len(matched)>0),
                missed_events=int(native and not len(matched)),
                first_alarm_delay_batches=float(matched.batch.min()-9) if native and len(matched) else np.nan,
                post_alert_fraction=float(whole[whole.batch>=9][col].mean()),
                input_seconds=float((whole.original_ks_seconds+
                    (whole.latent_ks_seconds if method=='full_KS_PCA' else 0)).sum())))
        for r in whole.to_dict('records'):
            truth={7} if condition=='C1' and r['batch']>=9 else (
                  {0} if condition=='C2' and r['batch']>=9 else set())
            feature_count = int(r.get('n_features', sum(k.startswith('x') and k.endswith('_rank') for k in r)))
            order=sorted(range(feature_count),key=lambda j:r[f'x{j}_rank'])
            flagged={j for j in range(feature_count) if r[f'x{j}_flag']}
            localization.append(dict(seed=seed,condition=condition,batch=r['batch'],regime=r['regime'],
                changed_marginal_n=len(truth),precision_at_1=float(order[0] in truth) if truth else np.nan,
                recall_at_3=len(set(order[:3])&truth)/len(truth) if truth else np.nan,
                flagged_features_n=len(flagged),true_feature_flags=len(flagged&truth),
                false_feature_flags=len(flagged-truth),
                feature_flag_precision=ratio(len(flagged&truth),len(flagged)),
                feature_flag_recall=ratio(len(flagged&truth),len(truth))))
    return pd.DataFrame(detection),pd.DataFrame(localization)


def bootstrap_summary(table, group_cols, metrics, replicates=2000):
    """Each input row is one seed value. Missing metrics stay undefined."""
    rng=np.random.default_rng(7312026)
    output=[]
    for key,g in table.groupby(group_cols,dropna=False,sort=True):
        if not isinstance(key,tuple): key=(key,)
        base=dict(zip(group_cols,key))
        for metric in metrics:
            values=g[metric].to_numpy(float)
            values=values[np.isfinite(values)]
            if len(values):
                means=values[rng.integers(0,len(values),size=(replicates,len(values)))].mean(axis=1)
                low,high=np.quantile(means,[.025,.975])
                mean=float(values.mean())
            else:
                low,high,mean=np.nan,np.nan,np.nan
            output.append(dict(**base,metric=metric,mean=mean,lower=low,upper=high,
                               defined_seeds=len(values),total_seeds=len(g)))
    return pd.DataFrame(output)


def evaluate(out):
    out=Path(out)
    validation=validate_run(out)
    (out/'validation.json').write_text(json.dumps(validation,indent=2))
    b=joined_batches(out)
    b['regime']=[regime(c,t) for c,t in zip(b.condition,b.batch)]
    # This file is deliberately named evaluator-only, and never loaded by the live evidence view.
    b.to_csv(out/'evaluator_batches.csv',index=False)
    impact=impact_table(b)
    impact.to_csv(out/'impact_per_seed.csv',index=False)
    metrics=['mae','bias','coverage','interval_width','precision','recall','harmful_unflagged_fraction',
             'flagged_fraction','false_flag_rate','harm_auroc']
    bootstrap_summary(impact,['condition','regime','method'],metrics).to_csv(out/'impact_summary.csv',index=False)
    # Exact pooled counts accompany seed means (undefined seed precision is never replaced with 0).
    pooled=impact.groupby(['condition','regime','method'])[['n','harm_n','flagged_n','tp','fp','fn','tn']].sum().reset_index()
    pooled['pooled_precision']=pooled.tp/(pooled.tp+pooled.fp).replace(0,np.nan)
    pooled['pooled_recall']=pooled.tp/(pooled.tp+pooled.fn).replace(0,np.nan)
    pooled.to_csv(out/'harm_counts.csv',index=False)
    detection,localization=detection_tables(b)
    detection.to_csv(out/'detection_per_seed.csv',index=False)
    bootstrap_summary(detection,['condition','method'],['null_alerts_per_1000','post_alert_fraction',
                      'first_alarm_delay_batches','missed_events','input_seconds']).to_csv(out/'detection_summary.csv',index=False)
    localization.to_csv(out/'localization_batches.csv',index=False)
    locseed=localization.groupby(['seed','condition','regime']).agg(
        n=('batch','size'),precision_at_1=('precision_at_1','mean'),recall_at_3=('recall_at_3','mean'),
        changed_marginal_n=('changed_marginal_n','max'),false_feature_flags=('false_feature_flags','sum'),
        flagged_features_n=('flagged_features_n','sum')).reset_index()
    locseed.to_csv(out/'localization_per_seed.csv',index=False)
    bootstrap_summary(locseed,['condition','regime'],['precision_at_1','recall_at_3','false_feature_flags']).to_csv(out/'localization_summary.csv',index=False)
    # Paired seed differences, equally weighted conditions. No duplicated delay rows.
    allimp=impact[(impact.regime=='all')&impact.mae.notna()]
    global_seed=allimp.groupby(['seed','method'])[metrics].mean().reset_index()
    global_seed.to_csv(out/'overall_per_seed.csv',index=False)
    bootstrap_summary(global_seed,['method'],metrics).to_csv(out/'overall_summary.csv',index=False)
    paired=[]
    wide=allimp.pivot(index=['seed','condition'],columns='method',values='mae')
    for comparator in ['raw_AC_no_calibration','ATC_style','constant_historical']:
        diff=wide.IDM_calibrated_AC-wide[comparator]
        for condition in sorted(b.condition.unique())+['equal_condition_mean']:
            d=diff.groupby('seed').mean() if condition=='equal_condition_mean' else diff.xs(condition,level='condition')
            for seed,value in d.items():
                paired.append(dict(seed=seed,condition=condition,comparator=comparator,mae_difference=value))
    paired=pd.DataFrame(paired)
    paired.to_csv(out/'paired_seed_differences.csv',index=False)
    bootstrap_summary(paired,['condition','comparator'],['mae_difference']).to_csv(out/'paired_bootstrap.csv',index=False)
    calibration=[]
    for seed,condition in b[['seed','condition']].drop_duplicates().itertuples(index=False,name=None):
        with np.load(out/'predictions'/f'{seed}_{condition}.npz') as p, \
             np.load(out/'evaluator_truth'/f'{seed}_{condition}.npz') as t:
            correct=(p['prediction']==t['labels'].reshape(24,500)).astype(float)
            for name,mask in [('all',np.ones(24,bool)),('pre',np.arange(24)<8),('post',np.arange(24)>=8)]:
                for method,key in [('calibrated','correctness_probability'),('raw','raw_confidence')]:
                    calibration.append(dict(seed=seed,condition=condition,regime=name,method=method,
                        adaptive_calibration_error=adaptive_calibration_error(p[key][mask].ravel(),correct[mask].ravel())))
    pd.DataFrame(calibration).to_csv(out/'calibration_error_per_seed.csv',index=False)
    correlations=[]
    for (seed,condition),g in b.groupby(['seed','condition']):
        score=-np.log10(np.maximum(g.min_adjusted_p,1e-300))
        rho=float(spearmanr(score,g.reference_accuracy-g.accuracy).statistic) if score.nunique()>1 and g.accuracy.nunique()>1 else np.nan
        correlations.append(dict(seed=seed,condition=condition,spearman_input_vs_accuracy_drop=rho))
    pd.DataFrame(correlations).to_csv(out/'score_loss_correlations.csv',index=False)
    # Availability is evaluated at the input horizon and after draining.
    v=pd.read_csv(out/'released_verification.csv')
    availability=[]
    for (seed,condition,delay),g in v.groupby(['seed','condition','delay']):
        known=g.available_at<=24
        availability.append(dict(seed=seed,condition=condition,delay=delay,total_batches=len(g),
            verified_at_input_end=int(known.sum()),unresolved_at_input_end=int((~known).sum()),
            verified_after_drain=len(g),harmful_batches=int(g.harm.sum()),
            harmful_verified_at_input_end=int((known&g.harm).sum()),
            mean_evidence_age_at_release=float(g.evidence_age_batches.mean()),
            # Identity verifier: no pre-release verdict or prediction is claimed.
            identity_mismatches_after_release=0))
    pd.DataFrame(availability).to_csv(out/'delay_availability.csv',index=False)
    timing=b.groupby('seed')[[c for c in b if c.endswith('_seconds')]].sum().reset_index()
    timing=timing.merge(pd.read_csv(out/'fit_records.csv')[['seed','fit_seconds']],on='seed',validate='one_to_one')
    timing.to_csv(out/'runtime_per_seed.csv',index=False)
    print(json.dumps(validation,indent=2))
    return b
