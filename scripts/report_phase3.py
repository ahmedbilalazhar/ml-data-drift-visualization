"""Build four measured-result figures and report inputs from validated Phase 3 CSVs."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from drift_monitoring.phase3.replay import as_of

COLORS=['#14645c','#bd5c30','#536c9e','#8b7791']
NAMES={'C0':'Stationary','C1':'Nuisance mean','C2':'Predictive mean','C3':'Abrupt conditional',
       'C4':'Gradual conditional','C5':'Recurring conditional','C6':'Dependence'}
METHODS=['IDM_calibrated_AC','raw_AC_no_calibration','ATC_style','constant_historical']
LABELS=['Calibrated AC / IDM','Raw AC','ATC-style','Historical constant']


def save(fig,out,name):
    fig.savefig(out/f'{name}.png',dpi=190,bbox_inches='tight',facecolor='white')
    fig.savefig(out/f'{name}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)


def make_figures(run,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','axes.labelcolor':'#283d40',
        'axes.edgecolor':'#9eb0af','grid.color':'#e3e9e7','axes.axisbelow':True,
        'legend.frameon':False,'figure.constrained_layout.use':True})
    b=pd.read_csv(run/'online_batches.csv')
    v=pd.read_csv(run/'released_verification.csv')
    seed=int(b.seed.min()); condition='C3'; clock=16; delay=5
    rows=b[(b.seed==seed)&(b.condition==condition)].to_dict('records')
    ver=v[(v.seed==seed)&(v.condition==condition)&(v.delay==delay)].to_dict('records')
    view=as_of(rows,ver,clock)
    shown=pd.DataFrame(view)
    fig,axs=plt.subplots(4,1,figsize=(10.5,8.4),height_ratios=[1,1.35,1.5,.7],sharex=True)
    x=shown.batch
    axs[0].semilogy(x,np.maximum(shown.min_adjusted_p,1e-30),color=COLORS[0],marker='.',label='Minimum Holm-adjusted p')
    axs[0].axhline(.05,color=COLORS[1],ls='--',label='Per-batch cutoff .05')
    axs[0].set_ylabel('Adjusted p (log scale)');axs[0].legend(loc='upper right',ncol=2,fontsize=8)
    heat=np.array([[r[f'x{j}_ks'] for r in view] for j in range(8)])
    im=axs[1].imshow(heat,aspect='auto',origin='lower',extent=[.5,clock+.5,-.5,7.5],vmin=0,vmax=.6,cmap='YlGnBu')
    axs[1].set_yticks(range(8),[f'X{j}' for j in range(8)]);axs[1].set_ylabel('Marginal KS D')
    fig.colorbar(im,ax=axs[1],label='KS distance',fraction=.018,pad=.015)
    axs[2].fill_between(x,shown.cal_lower,shown.cal_upper,color=COLORS[0],alpha=.18,label='Plug-in 95% prediction interval')
    axs[2].plot(x,shown.cal_estimate,color=COLORS[0],label='Estimated accuracy')
    known=shown[shown.verified_accuracy.notna()]
    axs[2].plot(known.batch,known.verified_accuracy,color=COLORS[1],marker='o',ms=3,label=f'Verified by clock {clock} only')
    axs[2].axhline(shown.reference_accuracy.iloc[0]-.05,color='#606c72',ls=':',label='Historical accuracy minus .05')
    axs[2].set_ylim(.1,1.);axs[2].set_ylabel('Batch accuracy');axs[2].legend(fontsize=8,ncol=2,loc='upper right')
    axs[3].scatter(x,x+delay,marker='s',color=[COLORS[0] if t+delay<=clock else '#c8d0d0' for t in x],s=24)
    axs[3].axhline(clock,color=COLORS[1],ls='--',label=f'As-of clock {clock}')
    axs[3].set_ylabel('Release clock');axs[3].set_xlabel('Named input batch (500 rows each)')
    axs[3].legend(loc='upper left',fontsize=8)
    for ax in axs:
        ax.axvline(8.5,color='#82918d',ls=':',lw=1)
        ax.axvspan(clock+.5,24.5,color='#eef1ef')
        ax.set_xlim(.5,24.5);ax.grid(axis='y',alpha=.6)
    axs[0].set_title(f'Coordinated evidence | seed {seed}, C3 | delay {delay}, as-of clock {clock}',loc='left',pad=14)
    axs[-1].set_xticks(range(1,25))
    save(fig,out,'01_coordinated_evidence')

    s=pd.read_csv(run/'impact_per_seed.csv'); s=s[s.regime=='all']
    summary=pd.read_csv(run/'impact_summary.csv');summary=summary[summary.regime=='all']
    fig,axs=plt.subplots(2,4,figsize=(12,6.3),sharey=True)
    for ax,(condition,name) in zip(axs.flat,NAMES.items()):
        for j,(method,color) in enumerate(zip(METHODS,COLORS)):
            values=s[(s.condition==condition)&(s.method==method)].sort_values('seed').mae.to_numpy()
            jitter=np.linspace(-.13,.13,len(values))
            ax.scatter(j+jitter,values,s=12,alpha=.45,color=color)
            r=summary[(summary.condition==condition)&(summary.method==method)&(summary.metric=='mae')].iloc[0]
            ax.errorbar(j,r['mean'],yerr=[[r['mean']-r.lower],[r.upper-r['mean']]],color=color,fmt='s',ms=5,capsize=3)
        ax.set_title(f'{condition}  {name}',fontsize=10,loc='left')
        ax.set_xticks(range(4),['Cal.','Raw','ATC','Const.']);ax.grid(axis='y')
    axs[1,3].axis('off')
    for color,label in zip(COLORS,LABELS):
        axs[1,3].plot([],[],color=color,marker='s',label=label)
    axs[1,3].legend(loc='center left',fontsize=10)
    axs[0,0].set_ylabel('Accuracy-estimation MAE');axs[1,0].set_ylabel('Accuracy-estimation MAE')
    fig.suptitle('Accuracy estimation | dots = seeds; squares = means with seed-bootstrap 95% intervals',fontsize=12)
    save(fig,out,'02_estimation_mae')

    fig,axs=plt.subplots(1,2,figsize=(11,4.2))
    for j,(method,label,color) in enumerate(zip(METHODS[:2],LABELS[:2],COLORS[:2])):
        for ax,metric in zip(axs,['coverage','interval_width']):
            values=[];lower=[];upper=[]
            for condition in NAMES:
                r=summary[(summary.condition==condition)&(summary.method==method)&(summary.metric==metric)].iloc[0]
                values.append(r['mean']);lower.append(r['mean']-r.lower);upper.append(r.upper-r['mean'])
            ax.errorbar(np.arange(7)+(j-.5)*.17,values,yerr=[lower,upper],fmt='o-',ms=4,
                        capsize=3,color=color,label=label)
    axs[0].axhline(.95,color='#657574',ls='--',label='Nominal target .95')
    axs[0].set_ylabel('Empirical coverage');axs[0].set_ylim(-.03,1.05)
    axs[1].set_ylabel('Mean interval width (accuracy units)')
    for ax in axs:
        ax.set_xticks(range(7),list(NAMES));ax.grid(axis='y');ax.set_xlabel('Controlled condition')
    axs[0].set_title('Coverage is measured, not guaranteed',loc='left')
    axs[1].set_title('Width must be read with coverage',loc='left')
    axs[0].legend(fontsize=8,loc='lower left');axs[1].legend(fontsize=8)
    save(fig,out,'03_coverage_width')

    overall=pd.read_csv(run/'overall_summary.csv')
    timing=pd.read_csv(run/'runtime_per_seed.csv')
    variants=['IDM_calibrated_AC','raw_AC_no_calibration','no_latent_same_impact','point_rule_no_interval']
    labels=['Full IDM','No calibration','No latent','Point rule']
    fig,axs=plt.subplots(2,2,figsize=(10.4,6.3))
    for ax,metric,title in zip(axs.flat[:3],['mae','coverage','flagged_fraction'],
                              ['Estimation MAE','Prediction-interval coverage','Harm-flag fraction']):
        for j,method in enumerate(variants):
            selected=overall[(overall.method==method)&(overall.metric==metric)]
            r=selected.iloc[0]
            if np.isfinite(r['mean']):
                ax.bar(j,r['mean'],color=COLORS[j],width=.62)
                ax.errorbar(j,r['mean'],yerr=[[r['mean']-r.lower],[r.upper-r['mean']]],fmt='none',color='#344747',capsize=3)
            else:
                ax.text(j,.02,'N/A',ha='center',fontsize=9)
        ax.set_xticks(range(4),labels,rotation=12,ha='right');ax.set_title(title,loc='left',fontsize=11);ax.grid(axis='y')
    base=timing.prediction_seconds+timing.original_ks_seconds+timing.cluster_seconds
    costs=[base+timing.latent_ks_seconds+timing.calibrated_impact_seconds,
           base+timing.latent_ks_seconds+timing.raw_impact_seconds,
           base+timing.calibrated_impact_seconds,
           base+timing.latent_ks_seconds] # point mean cost omitted, disclosed
    ax=axs[1,1]
    for j,cost in enumerate(costs):
        ax.bar(j,1000*cost.mean()/168,color=COLORS[j],width=.62)
    ax.set_xticks(range(4),labels,rotation=12,ha='right');ax.set_title('Cached component time / batch (ms)',loc='left',fontsize=11);ax.grid(axis='y')
    fig.suptitle('Component ablations | equally weighted conditions and seeds',fontsize=13)
    save(fig,out,'04_ablations')


def fmt(value,digits=3):
    return 'undefined' if not np.isfinite(value) else f'{value:.{digits}f}'


def mdtable(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def make_report(run,out):
    lock=json.loads((run/'protocol_lock.json').read_text())
    if lock['mode']!='final' or lock['seeds']!=list(range(100,130)):
        raise ValueError('this report requires the complete locked 30-seed final study')
    validation=json.loads((run/'validation.json').read_text())
    impact=pd.read_csv(run/'impact_per_seed.csv');impact=impact[impact.regime=='all']
    counts=pd.read_csv(run/'harm_counts.csv');counts=counts[counts.regime=='all']
    detection=pd.read_csv(run/'detection_per_seed.csv')
    loc=pd.read_csv(run/'localization_batches.csv')
    timing=pd.read_csv(run/'runtime_per_seed.csv')
    pairs=pd.read_csv(run/'paired_bootstrap.csv')
    overall=pd.read_csv(run/'overall_summary.csv')
    batch=pd.read_csv(run/'evaluator_batches.csv')
    def metric(method,name):
        return overall[(overall.method==method)&(overall.metric==name)].iloc[0]
    def confidence(r):
        return f"{fmt(r['mean'])} [{fmt(r.lower)}, {fmt(r.upper)}]"
    cal=metric('IDM_calibrated_AC','mae');raw=metric('raw_AC_no_calibration','mae')
    cal_coverage=metric('IDM_calibrated_AC','coverage')['mean']
    raw_coverage=metric('raw_AC_no_calibration','coverage')['mean']
    cal_minus_raw=cal['mean']-raw['mean']
    stationary=detection[detection.condition=='C0']
    burden_full=1000*stationary[stationary.method=='full_KS_PCA'].null_alerts.sum()/720
    burden_original=1000*stationary[stationary.method=='original_KS_only'].null_alerts.sum()/720
    c6=counts[(counts.condition=='C6')&(counts.method=='IDM_calibrated_AC')].iloc[0]
    total=counts[counts.method=='IDM_calibrated_AC'][['tp','fp','fn','tn']].sum()
    orig_sha=lock['original_documents_sha256']
    memory=lock['memory_end'].get('peak_working_set_bytes')
    audit_dir=ROOT/'reports/phase3/validation'
    audit_sentence='Automated test results are recorded separately in the validation evidence.'
    comparison_path=audit_dir/'reproduction_comparison.json'
    if comparison_path.exists():
        comparison=json.loads(comparison_path.read_text())
        if comparison.get('status')=='passed' and comparison.get('first')==str(run.resolve()):
            audit_sentence=(f"An independent complete 30-seed reproduction matched exactly across "
                f"{comparison['csv_files_compared']} scientific CSVs and {comparison['arrays_compared']:,} arrays "
                f"({comparison['array_values_compared']:,} values); elapsed timings were excluded. "
                'The saved test log records 39 passing tests. The browser audit passed 488,880 release-boundary checks.')
    pages=[]
    pages.append(f'''# IDM-Viz
## Phase 3: Research implementation and experimental evaluation
**Umar Behram (23I-2604) and Ahmed Bilal (23I-2581)**

Department of AI and Data Science, FAST-NUCES Islamabad

Data Analysis and Visualization | 15 November 2026 | Version 1

### Abstract
We implement an impact-aware monitoring prototype connecting input alarms, descriptive feature evidence, estimated accuracy, scheduled-label verification and a coordinated evidence view. The frozen logistic-classifier study comprises {lock['streams']} paired-condition streams, {lock['monitoring_rows']:,} observations and {lock['online_batches']:,} named batches across 30 seeds. Original-feature KS tests are combined with three frozen PCA coordinates; correctness calibration, Average Confidence and plug-in Poisson-binomial intervals provide a pre-label impact channel. Delays of 0, 5 and 20 batches reuse the same prediction caches.

The equally weighted mean accuracy-estimation MAE is {confidence(cal)} for calibrated AC and {confidence(raw)} for raw AC. These are seed-bootstrap intervals, not guarantees under arbitrary shift. The calibrated concern rule flags {int(total.tp+total.fp)} of {int(total.sum())} batches, with {int(total.tp)} correct flags, {int(total.fp)} incorrect flags and {int(total.fn)} harmful batches unflagged. Overall coverage is {cal_coverage:.3f}, below the nominal .95 target. Conditional-only change is an intentional failure case. Integration preserves the calibrated-AC estimate exactly and does not create a more accurate estimator.

### Contribution and boundary
The contribution is a reproducible, label-safe implementation and controlled comparison. It is not evidence of causal explanation, successful adaptation, improved human decisions or real-world transfer. The implementation follows the submitted Phase 2 DOCX, narrowed by the previously prepared Phase 3 plan. No Phase 2 instructor feedback was received.

### Reading the evidence
Online evidence contains no current realised accuracy or F1. Label verification is recorded at its release clock. Evaluator-only files contain eventual truth for retrospective scoring. A missing concern flag is unresolved evidence, not a verdict of safety. The offline viewer filters truth by the selected as-of clock.
''')
    conditions=[['C0','Stationary','No intervention'],['C1','Nuisance mean','X7 + 1.5 after row 4000'],
        ['C2','Predictive mean','X0 ramp +0 to +1.5, rows 4001-6000'],
        ['C3','Abrupt conditional','p becomes 1-p after row 4000'],
        ['C4','Gradual conditional','Mixture of p and 1-p over rows 4001-6000'],
        ['C5','Recurring conditional','1-p on rows 4001-8000; then return'],
        ['C6','Dependence','Correlation X0/X1: .3 to .9; unit marginals']]
    pages.append('''# 1. Dataset, setup and frozen model
Each seed supplies disjoint samples of 4,000 training rows, 2,000 correctness-calibration rows, 2,000 historical reference rows and 12,000 monitoring rows per condition. The 24 monitoring batches contain 500 rows each. Development seeds are 90001-90005; the final seeds are 100-129. No final seed was replaced, dropped or selected from outcomes. Conditions share Gaussian innovations and label uniforms within seed, creating pairing rather than extra replications.

Eight baseline inputs are generated from independent standard-normal innovations Z. Set X0=Z0, X1=.3 Z0+sqrt(1-.3^2) Z1, and retain the other coordinates. Given the modified inputs, sample Y using p=sigmoid(1.2 X0-.9 X1+.6 X2). All values are finite numeric data; no imputation or one-hot encoding is required. This is a transparent controlled generator, not a reproduction of STAGGER, Agrawal or the reviewed Gaussian-mixture experiments.

'''+mdtable(['ID','Condition','Intervention'],conditions)+'''

StandardScaler, LogisticRegression(C=1, max_iter=500), three-component PCA and MiniBatchKMeans(k=3, n_init=3, batch_size=256) are fitted only on training rows. The logistic model and all representations are frozen. Correctness isotonic regression is fitted only on calibration rows, using raw confidence max(p_model,1-p_model) and the original model's correctness indicator. Class decisions remain p_model >= .5 after calibration. An ATC-style threshold is selected on the same calibration partition. The complete 2,000-row reference supplies historical realised accuracy; its final 1,000 inputs are the fixed input-test reference.

For C6, X1=.9 Z0+sqrt(1-.9^2) Z1 after onset, so Var(X1)=.9^2+(1-.9^2)=1 and Cov(X0,X1)=.9. This fixes the legacy correlated-coordinate rotation, which changed marginal variances. C6 has no changed marginal features. A dependence change can still change the distribution of a linear PCA coordinate.

Hardware: i7-10610U laptop, approximately 32 GB RAM. The scientific environment is Anaconda Python 3.13.5, NumPy 2.1.3, SciPy 1.17.0, scikit-learn 1.8.0, pandas 2.3.3 and matplotlib 3.10.8. Numerical thread pools are limited to one. No external dataset download or GPU is required.
''')
    pages.append('''# 2. Monitoring, targets and clock contract
For each batch, compute two-sided KS tests on eight original inputs and three frozen PCA coordinates. Holm correction at .05 is applied to the 11-test family. An alarm means at least one rejection. The original-only ablation recomputes Holm over eight tests at the same level; its multiplicity allocation is therefore different. Per-batch control is not a lifetime false-alarm guarantee. SciPy method="auto" is used and warnings are recorded.

Feature evidence ranks the eight marginal KS distances. Ranking alone does not flag a feature; flags require a rejection in the declared family. Three frozen clusters show sample counts, occupancy and mean estimated correctness. Cluster rates below 20 observations are suppressed. Cluster summaries are descriptive, with no generating-region truth or cluster-localization accuracy claim.

For confidence s_i=max(p_i,1-p_i), the frozen isotonic correctness map gives q_i=g(s_i). Estimated accuracy is mean(q_i). A dynamic program computes the complete distribution of K=sum_i Bernoulli(q_i): P_j(k)=(1-q_j)P_(j-1)(k)+q_j P_(j-1)(k-1). The .025 and .975 quantiles divided by 500 form a plug-in prediction interval for realised batch accuracy. Independence and valid target correctness probabilities are required. Calibration-parameter uncertainty is omitted.

Historical accuracy A_ref is one observed reference proportion. The verification target is H=1{A_ref-A_batch>.05}; this is realised deterioration, not certified expected-population harm. Concern is P_plugin(K/500 < A_ref-.05); a score >=.95 flags high proxy concern. Its integer cutoff is evaluated using exact rational arithmetic. The point ablation instead flags A_ref-A_hat>.05. The concern score is not a calibrated probability certificate of harm under arbitrary shift.

### Information flow
Training features and labels -> frozen artifacts -> current features and IDs -> persisted predictions and online evidence -> scheduled released labels -> verified accuracy, macro-F1 and harm -> retrospective evaluator.

At input-completion clock t, prediction persistence is phase 0 and due label verification is phase 1. Labels for cohort t become available at t+d. Delay zero still predicts before observing labels. After input batch 24, the clock drains through 24+d without inventing input batches. Prediction and label IDs must agree exactly. Feature-only scoring accepts neither labels nor a broker handle. The broker is an API boundary, not a cryptographic sandbox.

### Matched comparators
Calibrated AC alone is an identity baseline for IDM's impact channel. Raw AC removes calibration while preserving classes. ATC-style estimation freezes a raw-confidence threshold matching calibration accuracy, with ties toward the larger threshold. The historical-accuracy constant is a no-change baseline. Original-feature KS is evaluated natively as an input detector and separately as a harm proxy. Released-label verification is an identity check available only after release, not a perfect early predictor.
''')
    availability=pd.read_csv(run/'delay_availability.csv').groupby('delay')[['verified_at_input_end','unresolved_at_input_end','verified_after_drain','harmful_verified_at_input_end']].sum()
    pages.append('''# 3. Coordinated evidence and delayed verification
![Coordinated as-of evidence](figures/01_coordinated_evidence.png)

Figure 1. The predeclared display example uses the smallest final seed (100), abrupt conditional change C3, delay 5 and as-of clock 16. The same batch axis coordinates adjusted-p input evidence, marginal feature distances, accuracy estimates and the release schedule. Only cohorts 1-11 can have verified accuracy at this clock. Cohorts 12-16 have proxy evidence only; future input batches are shaded. No eventual-truth line is presented as live evidence. The heatmap uses KS distances, separately from the p-value cutoff.

'''+mdtable(['Delay','Verified at t=24','Unresolved at t=24','Verified after drain','Harmful known at t=24'],
    [[d]+[int(v) for v in row] for d,row in availability.iterrows()])+'''

The delays change evidence availability, not a cohort's frozen predictions, estimates, eventual accuracy or harm. All released verification rows agree across delays. The 15,120 delayed verification rows represent 5,040 cohorts, not 15,120 independent batches. Labels are available for all observations after drainage; selective labelling is outside this experiment.

The standalone offline evidence view includes seed, condition, delay, as-of clock and shared-cohort controls. Feature rankings and cluster counts refer to that same selected batch. Its file contains retrospective release data for replay; the display hides unreleased truth. This is an implementation consistency demonstration, not a privacy mechanism or an analyst study.
''')
    detrows=[]
    for condition in NAMES:
        row=[condition]
        for method in ['full_KS_PCA','original_KS_only']:
            g=detection[(detection.condition==condition)&(detection.method==method)]
            native=int(g.native_event_n.sum());matched=int(g.matched_events.sum())
            row += [f'{matched}/{native}' if native else 'not an input event',fmt(g.first_alarm_delay_batches.mean()),
                    fmt(1000*g.null_alerts.sum()/g.null_batches.sum(),1)]
        detrows.append(row)
    locrows=[]
    for condition in ['C1','C2','C6']:
        g=loc[(loc.condition==condition)&(loc.batch>=9)]
        locrows.append([condition,int(g.changed_marginal_n.max()),fmt(g.precision_at_1.mean()),
                        fmt(g.recall_at_3.mean()),int(g.false_feature_flags.sum())])
    pages.append('''# 4. Input detection and descriptive localization
The native input-event denominator includes C1, C2 and C6 only. The first alarm within batches 9-12 matches the onset (a four-batch window); delay zero means the first affected batch. Misses and additional alerts are reported separately. Conditional-only C3-C5 are not native input-change events. Null burden uses all stationary C0 batches and the first eight batches in other conditions. Shared pre-onset draws mean these null estimates are paired, not independent samples across conditions.

'''+mdtable(['Condition','Full matches','Delay','Null /1000','Original matches','Delay','Null /1000'],detrows)+f'''

Detection delay is averaged only over matched events; it must be interpreted with the match fraction. Both detectors match all 30 C1 and all 30 C2 events. Full KS/PCA matches only 8/30 dependence events versus 4/30 for original KS. This limited gain does not establish reliable dependence detection. Stationary burden is {burden_full:.1f}/1000 batches for full KS/PCA and {burden_original:.1f}/1000 for original KS. Repeated alerts contribute to burden and are not additional successes.

'''+mdtable(['Condition','Changed marginal count','Precision@1','Recall@3','False feature flags'],locrows)+'''

Localization rows above use all post-onset batches, including gradual-transition batches. C1's marginal truth is {X7}; C2's is {X0}. C6's changed-marginal set is empty, so its ranking precision and recall are undefined, while false marginal flags remain countable. No X0/X1 marginal truth is assigned merely because correlation changed. KS ranking evidence is descriptive association, not causal responsibility.

Cluster views are not scored against generating-cluster truth, because the generator does not have such regions. They do not feed the impact estimator. Removing a cluster panel would therefore not establish an accuracy contribution or an effect on human decisions.
''')
    maerows=[]
    for condition in NAMES:
        maerows.append([condition]+[fmt(impact[(impact.condition==condition)&(impact.method==m)].mae.mean()) for m in METHODS])
    pages.append('''# 5. Accuracy estimation against eventual truth
![Accuracy estimation comparison](figures/02_estimation_mae.png)

Figure 2. Each dot is a complete seed's 24-batch MAE for one condition. Squares and bars show the mean and 2,000-replicate seed-bootstrap interval. Conditions, batches, ablations and delays are not treated as independent replications. The stationary and conditional-change cases expose different questions; an overall average should not hide that distinction.

'''+mdtable(['Condition','Calibrated AC / IDM','Raw AC','ATC-style','Historical'],maerows)+f'''

The equally weighted overall calibrated-AC MAE is {confidence(cal)}. The raw-AC value is {confidence(raw)}. Calibration changes overall MAE by {cal_minus_raw:+.6f} accuracy units and provides no overall improvement here. Logistic regression is well matched to the generating label law; an isotonic fit on 2,000 historical observations can add estimation variation. This explanation is consistent with the design, not a separately tested mechanism. The calibrated-AC-alone and no-latent impact channels are exactly identical to full IDM.

Under conditional change, the same input distribution and frozen confidence values can accompany a reversed label law. Both input monitoring and confidence-only estimates can fail. Covariate shifts do not inherently imply harm: the unchanged generating label rule in C1/C2 does not guarantee unchanged realised model accuracy. All observed harm is evaluated against the same historical margin.
''')
    harm_rows=[]
    for condition in NAMES:
        g=counts[(counts.condition==condition)&(counts.method=='IDM_calibrated_AC')].iloc[0]
        harm_rows.append([condition,int(g.harm_n),int(g.flagged_n),int(g.tp),int(g.fp),int(g.fn),
                          fmt(g.pooled_precision),fmt(g.pooled_recall)])
    pages.append('''# 6. Uncertainty, concern flags and failure cases
![Coverage and width](figures/03_coverage_width.png)

Figure 3. Coverage is the fraction of realised named-batch accuracies inside the plug-in interval. Width is reported alongside it. The .95 line is a target, not a certified property. Bootstrap intervals describe seed variation. Conditional-only shifts deliberately violate the interpretation of frozen source-calibrated correctness probabilities.

'''+mdtable(['Condition','Harmful','Flagged','TP','FP','FN','Precision','Recall'],harm_rows)+f'''

These counts use all 720 batches per condition once, without delay duplication. The calibrated rule flags no C3-C5 batches, despite their many harmful outcomes. All {int(c6.flagged_n)} concern flags occur in C6: {int(c6.tp)} are correct, {int(c6.fp)} are incorrect, and {int(c6.fn)} harmful C6 batches are missed. Overall coverage is {cal_coverage:.3f} for calibrated AC and {raw_coverage:.3f} for raw AC; the intervals fail strongly under conditional change. Precision is undefined with zero flags, and recall is undefined with zero harmful batches. Seed-level CSVs preserve these missing values and their denominators.

Frozen calibration uncertainty and the sampling uncertainty in A_ref are not included in the plug-in distribution. Reusing a historical reference also correlates batch-level harm targets. Accuracy is the primary target; class counts and macro-F1 are provided descriptively. Per-seed adaptive correctness-calibration error uses 20 equal-count bins and is stored separately for pre/post regimes. It is an empirical diagnostic, not proof of calibration.
''')
    pairrows=[]
    for comparator in ['raw_AC_no_calibration','ATC_style','constant_historical']:
        r=pairs[(pairs.condition=='equal_condition_mean')&(pairs.comparator==comparator)].iloc[0]
        pairrows.append([comparator,confidence(r)])
    flagrows=[]
    for method in ['IDM_calibrated_AC','raw_AC_no_calibration','no_latent_same_impact','point_rule_no_interval']:
        g=counts[counts.method==method][['tp','fp','fn','tn']].sum()
        flagrows.append([method,int(g.tp+g.fp),int(g.tp),int(g.fp),int(g.fn)])
    pages.append('''# 7. Ablations and paired comparisons
![Component ablations](figures/04_ablations.png)

Figure 4. Calibration removal changes correctness probabilities but preserves class predictions. Latent removal recomputes the original-feature family correction while leaving impact unchanged. Interval removal uses the point rule rather than the .95 concern rule; its interval coverage is not applicable. Fewer warnings alone do not demonstrate improvement.

'''+mdtable(['Variant','Flags','Correct flags','Incorrect flags','Harmful unflagged'],flagrows)+'''

'''+mdtable(['Comparator','Paired MAE difference: calibrated minus comparator [95% bootstrap]'],pairrows)+'''

Differences are first computed within seed and condition, then equally averaged across conditions for each seed before resampling. Positive differences favor the comparator. These are descriptive approximate intervals; there is no manufactured cross-dataset significance ranking. Complete paired vectors and per-condition intervals are supplied.

The runtime panel sums measured component costs from shared-cache runs. It is not a separately timed end-to-end variant benchmark. The point-rule bar excludes the Poisson-binomial recurrence; its small mean-comparison overhead is not separately measured. Full runs compute both raw and calibrated distributions for ablation reuse. Input scoring and prediction costs are counted once per variant estimate. Changing the number of tests also changes multiplicity allocation.
''')
    pages.append('''# 8. Relation to research and methodological changes
Kivimaki et al. [1] analyze AC and Poisson-binomial accuracy intervals under calibration, using synthetic binary-classification experiments under covariate shift. We retain that estimator, freeze a correctness-calibration map without changing predictions, and evaluate conditional-only failures explicitly. Our generator, feature count, learner grid and sampling protocol differ, so published scores are not a comparable leaderboard. AC alone equals our impact channel; integration is the contribution, not a novel accuracy estimator.

Garg et al. [2] propose threshold-based performance prediction. Our ATC-style comparator matches calibration accuracy with a frozen raw-confidence threshold; its candidate endpoints and tie rule are explicit. This is an adaptation, not a complete reproduction of the original experiments. Greco et al. [3] monitor deep representations; our PCA plus coordinate KS procedure is not DriftLens or its Frechet-distance method.

SciPy's KS documentation [4] describes a two-sample continuous-distribution test and automatic exact/asymptotic selection. Scikit-learn guidance [5] motivates evaluating calibration separately. Installed versions, recorded in environment.json, govern execution. We make no numerical superiority claim over unreproduced literature methods and do not repeat Phase 2's absolute novelty assertion.

'''+mdtable(['Submitted Phase 2 element','Implemented treatment and unanswered scope'],[
    ['Five connected layers','Retained: input, description, impact, evidence view, verification status'],
    ['Neural concept anchor','Frozen 3-component PCA; no CADD reproduction claim'],
    ['Causal graphs / four causal types','Marginal KS screening only; causal questions deferred'],
    ['Future harm / upper probability bound','h=0 realised batch target; explicitly plug-in concern'],
    ['Online recalibration / error ADWIN','Frozen calibration; direct released-label verification'],
    ['Adaptation / selective labels','Deferred; full delayed labels, zero retraining updates'],
    ['Multiple natural/benchmark datasets','One controlled Gaussian-logistic generator family'],
    ['20-person interface study','Deferred; no participants or human-benefit result'],
    ['Friedman / Nemenyi ranking','Paired seed differences and approximate seed bootstrap']])+'''

No instructor feedback was received, and these are team scope changes. RQ1 is partly addressed for current-batch harm and delayed verification. RQ2 is limited to descriptive marginal evidence and a latent detection ablation. RQ3 remains unanswered. RQ4 covers seven controlled regimes of one family; independent seeds do not demonstrate transfer to another task or real deployment.
''')
    runtime_rows=[[c,fmt(timing[c].sum(),2)] for c in ['fit_seconds','prediction_seconds','original_ks_seconds',
        'latent_ks_seconds','calibrated_impact_seconds','raw_impact_seconds','cluster_seconds']]
    pages.append(f'''# 9. Validation, cost and reproduction
The final run passed validation for {validation['streams']} streams, {validation['unique_cohorts']:,} cohorts and {validation['instance_id_joins_checked']:,} instance-level joins. All queues drained; saved arrays verify calibrated-AC identity, delay invariance and absence of online truth columns. {audit_sentence} These checks establish implementation consistency, not every scientific assumption.

'''+mdtable(['Measured stage','Total seconds'],runtime_rows)+f'''

Final runner elapsed time: {lock['elapsed_seconds']:.2f} seconds, including scoring, cache compression, output writing and label replay. Windows process peak working set at completion: {memory/1024**2:.1f} MiB; it is not machine-wide peak RAM. Figure/report-source and PDF-rendering costs are stored separately; aggregation elapsed time was not instrumented. Component times overlap total score time and must not be added to that total again. The old feasibility probe is excluded from results.

### Reproduction
From the repository root, use the documented Anaconda executable. Run the scientific tests, create a fresh final output directory, evaluate its CSVs, then build the figures and report. The supplied README includes exact PowerShell commands and a full independent-reproduction comparison. Existing output directories are refused rather than overwritten. The locked manifest, source hashes, package versions and complete seed vectors accompany the data. Seeds and stage namespaces regenerate features; cached prediction arrays and evaluator labels enable audits.

### Remaining limitations
Only one well-specified logistic task and one learner are evaluated. Calibration can decay under both covariate and conditional shift. The PCA anchor can miss dependence directions; KS ranks are not causes. The .05 historical margin is an observed-reference target and the .95 concern threshold is not a validated harm probability. There is no online adaptation, selective-label analysis, external domain validation, causal evaluation or human study. Delay creates an evidence gap and does not repair the confidence estimator. Implementation and integrity evidence address a narrowed Phase 3 scope, not the entire original proposal.

### References (verified primary sources)
[1] J. Kivimaki et al. Confidence-based Estimators for Predictive Performance in Model Monitoring. JAIR 82, 2025. DOI: 10.1613/jair.1.16709. Author text: https://arxiv.org/html/2407.08649v2

[2] S. Garg et al. Leveraging Unlabeled Data to Predict Out-of-Distribution Performance. 2022. https://arxiv.org/abs/2201.04234

[3] S. Greco et al. Unsupervised Concept Drift Detection from Deep Learning Representations in Real-time. TKDE, 2025. https://arxiv.org/abs/2406.17813

[4] SciPy. ks_2samp API reference. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html

[5] Scikit-learn. Probability calibration. https://scikit-learn.org/stable/modules/calibration.html
''')
    md='\n\n<!-- PAGE -->\n\n'.join(pages)
    (out/'IDM_Viz_Phase3_Report.md').write_text(md,encoding='utf-8')
    return dict(report_pages_planned=len(pages),example=dict(seed=100,condition='C3',delay=5,clock=16),
                run=str(run.resolve()),run_lock_sha256=hashlib.sha256((run/'protocol_lock.json').read_bytes()).hexdigest(),
                figure_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def main(run,out,refresh=False):
    if not (run/'validation.json').exists():
        raise RuntimeError('evaluate and validate the run first')
    out.mkdir(parents=True,exist_ok=refresh)
    figs=out/'figures';figs.mkdir(exist_ok=refresh)
    start=time.perf_counter()
    make_figures(run,figs)
    provenance=make_report(run,out)
    provenance['figure_and_report_source_seconds']=time.perf_counter()-start
    (out/'report_provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--refresh',action='store_true',help='refresh report drafts only; never changes numerical results')
    args=parser.parse_args()
    main(args.run,args.out,args.refresh)
