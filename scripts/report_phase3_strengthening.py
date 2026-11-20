"""Consolidate preserved studies and the explicit label-audit extension."""
from pathlib import Path
import argparse,json,re,shutil,sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from continue_phase3_overnight import latex_source
from build_phase3_view import build

def table(d):return d.to_markdown(index=False,floatfmt='.3f')

def main(run,out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    s=pd.read_csv(run/'policy_summary.csv');p=pd.read_csv(run/'paired_bootstrap.csv')
    loc=pd.read_csv(run/'localization_summary.csv')
    # Show all conditions at a declared example delay/budget, never select by result.
    chosen=p[(p.delay==5)&(p.budget==4)]
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    policies=['input_only','concern_only','combined','released_label']
    colors=['#087e8b','#d87a35','#687b45','#7a66a2']
    for i,(policy,color) in enumerate(zip(policies,colors)):
        g=chosen[chosen.policy==policy].sort_values('condition');x=list(range(len(g)))
        ax.errorbar([a+(i-1.5)*.15 for a in x],g['mean'],yerr=[g['mean']-g.lower,g.upper-g['mean']],fmt='o',capsize=2,label=policy,color=color)
    ax.axhline(0,color='#444',lw=1);ax.set_xticks(range(7),[f'C{i}' for i in range(7)])
    ax.set_ylabel('Additional harmful cohorts audited versus uniform')
    ax.set_title('Equal 2,000-label budget; delay 5; 95% paired seed intervals')
    ax.legend(ncol=2,fontsize=8);ax.grid(axis='y',alpha=.2)
    for ext in ('pdf','svg','png'):fig.savefig(out/f'audit_comparison.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for policy,color in zip(policies,colors):
        g=s[(s.condition=='C3')&(s.budget==4)&(s.policy==policy)].sort_values('delay')
        ax.plot(g.delay,g.harmful_confirmations_during_input,'o-',label=policy,color=color)
    ax.set_xlabel('Label delay (batches)');ax.set_ylabel('Harmful audits confirmed by input clock 24')
    ax.set_title('Conditional-change verification under a fixed label budget');ax.legend(fontsize=8)
    for ext in ('pdf','svg','png'):fig.savefig(out/f'delay_verification.{ext}',dpi=180)
    plt.close(fig)
    build(ROOT/'results/phase3/final_v1',out/'evidence_view.html')
    view=(out/'evidence_view.html').read_text(encoding='utf-8')
    view=view.replace('Labels unresolved. Scheduled release:', 'Unverified estimate. Source calibration can fail under drift. Scheduled release:')
    view=view.replace('Plug-in interval [','Current-cohort estimate (horizon 0), not a future-loss forecast. Plug-in interval ignores calibration uncertainty: [')
    view=view.replace('<small>Estimate</small>','<small>Source-calibrated estimate</small>')
    (out/'evidence_view.html').write_text(view,encoding='utf-8')
    intro='''# IDM-Viz: consolidated implementation and experimental evaluation

## Research basis and scope

The submitted Phase 2 DOCX is the authoritative methodological basis. No instructor feedback was received. The implementation connects input-distribution alarms, descriptive feature/cluster evidence, source-calibrated current-cohort accuracy estimates, delayed-label verification and coordinated visualization. Its concern score is a plug-in proxy for current-cohort harm relative to historical accuracy; horizon h=0. It is not a validated future-loss forecast. The proposed analyst study, causal discovery and neural anchor remain deferred. The frozen PCA anchor is an explicit substitution. No human benefit is established.

The new label-audit comparison is outcome-informed. All existing evidence remains unchanged. Audit actions purchase verification; they do not refit classifiers or increase their accuracy. It evaluates usefulness of automated audit allocation, not analyst decision quality.

## Equal-budget audit protocol

Five policies are compared: fixed uniform, input evidence, calibrated concern, conjunction of development-percentile input/concern, and prior released-label evidence. Thresholds use only development seeds 90001-90005. Final seeds 100-129 were already studied, so this is an exploratory extension rather than an untouched confirmatory test. Budget is exactly two or four 500-label cohorts per 24-batch stream. Delays are 0, 5 and 20. Selected predictions precede same-clock release, including zero delay. Remaining quota is filled at the tail, explicitly logged. The released-label comparator makes an initial scheduled probe and may use only selected labels released strictly before the current clock.

Equal budget does not imply equal scheduling. Tail filling can favor persistent late drift and hurt recurrence; it is part of the policy, not evidence that an input detector detected a conditional change. The comparison is batch-audit allocation, not episode-level deployment utility. Costs are label counts, not monetary or human costs. Harm targets are model-specific realised accuracy drops greater than 0.05 against the frozen historical reference. Reference uncertainty is omitted.

![Equal-budget comparison](audit_comparison.png)

![Delayed verification](delay_verification.png)

## Paired audit results

The table includes every condition at the predeclared example delay 5/budget 4. Intervals use 2,000 paired seed bootstrap replicates. These many exploratory comparisons have no familywise significance guarantee; effect sizes are the principal evidence. All delays and budgets are included in the machine-readable results.

'''
    body=intro+table(chosen[['condition','policy','mean','lower','upper']])+'\n\n## Feature-localization evidence\n\n'+table(loc)+'''

These values are aggregated over batches with known changed marginals in C1/C2. They are not causal attribution. C6 changes dependence with unchanged marginals; C3-C5 alter labels without a known changed input marginal. Their feature-localization metric is undefined, not zero or successful. Frozen clusters are descriptive regions, not validated causal explanations.

## External calibration transfer failure

The preserved UCI 224 cohort evaluation uses batch 1 split into 267 training, 89 calibration and 89 reference rows; batches 2-10 remain whole monitoring cohorts. Mean cohort accuracy is 0.432, macro-F1 0.290, calibrated estimated accuracy 0.976, MAE 0.545, approximate interval coverage 0/9 and input alarms 9/9. This is one external dataset and not nine independent replications. Class composition and sensor conditions change together. Small source calibration, normal-approximation intervals, a different correctness mapping and absent simulated release timing limit comparability with the synthetic study. Historical batch-1 class-proportion metadata has a denominator defect; monitoring outcomes are unaffected, but that metadata must not be used as a probability distribution. The class-mix sensitivity is descriptive and outcome-informed.

## Research comparison and interpretation

Average Thresholded Confidence predicts target accuracy using thresholded model confidence. Garg et al. show that unlabeled accuracy estimation requires assumptions about distribution shift; comparison to their published numeric scores is not fair on our different generators and learners. Our ATC-style baseline is therefore a within-study comparator. Ovadia et al. benchmark uncertainty under dataset shift, supporting evaluation of calibration transfer rather than an assumption that source calibration persists. Neither paper establishes IDM-Viz's superiority or analyst usefulness. Sources: https://arxiv.org/abs/2201.04234 and https://proceedings.neurips.cc/paper/2019/hash/8558cb408c1d76621371888657d2eb1d-Abstract.html .

## Original and model/task extension evidence

The following preserved study text supplies the classifier setup, baseline and component-ablation evidence. It predates the audit extension; its scope statements refer to that study version. Timings and historical receipts must not be presented as newly repeated tests.

'''
    for index,(source,label) in enumerate([
        (ROOT/'reports/phase3/v2_final/23I-2581_23I-2604_Phase3_Report.md','Model/task study'),
        (ROOT/'reports/phase3/adaptation_final/23I-2581_23I-2604_Phase3_Adaptation_Report.md','Selected adaptation study')]):
        text=source.read_text(encoding='utf-8')
        def image(m):
            caption,path=m.groups();p=source.parent/path
            if not p.is_file():return m.group(0)
            target=f'preserved_{index}_{p.name}';shutil.copy2(p,out/target)
            return f'![{caption}]({target})'
        text=re.sub(r'!\[(.*?)\]\((.*?)\)',image,text)
        body+='\n\n## '+label+'\n\n'+text
    body+='''

## Validation and unresolved claims

The audit extension tests equal quotas, prefix decisions unaffected by future scores, strict zero-delay release ordering and absence of unavailable feedback at long delay. All budget rows are checked. No full fresh reproduction of the earlier classifier experiments is claimed by this report. The long 1,500-seed-per-task robustness study is still a separate running job and has no final findings here. A larger number of seeds cannot repair calibration transfer, prove future forecasting, establish causal explanations, or replace a participant study.

The evidence view labels unresolved estimates and current-cohort scope. Existing as-of release logic is retained; revised wording does not certify current calibration. Inspect the generated pages and dashboard before final submission. Original source documents and submitted archive remain preserved. New results and manuscript are prepared research improvements, not uploaded submissions.
'''
    body=body.translate(str.maketrans({'’':"'",'–':'-','−':'-','—':'-','“':'"','”':'"'}))
    horizon=ROOT/'results/phase3/horizon_metadata_strengthening_v2'
    if horizon.is_dir():
        h=pd.read_csv(horizon/'horizon_summary.csv')
        body+='\n\n## Exploratory future-horizon persistence benchmark\n\nAt input clock t, each current estimate is persisted to t+1 or t+5 without future features or labels. These are explicit forecasting baselines, not a learned future-risk model. Estimates and issue/target clocks are exported separately from truth. Methods are compared within each horizon because available cohorts differ. No future-interval guarantee is claimed. Below are C3 results; all conditions are exported.\n\n'+table(h[h.condition=='C3'])
        body+='\n\nThe external historical class-composition denominator error is corrected in separate source-versus-reference metadata, with proportions checked to sum to one within each population. Preserved original monitoring outcomes are unchanged.\n'
    md=out/'REPORT.md';md.write_text(body,encoding='utf-8');tex=latex_source(md)
    content=tex.read_text(encoding='utf-8')
    content=content.replace(r'\begin{document}',r'\usepackage{xurl,seqsplit}\setlength{\tabcolsep}{2pt}'+'\n'+r'\begin{document}')
    content=content.replace('IDM-Viz: Large-Seed Robustness Evaluation','IDM-Viz: Consolidated Research Evaluation')
    content=re.sub(r'`([^`\n]+)`',lambda m:r'\texttt{\seqsplit{'+m.group(1)+'}}',content)
    content=re.sub(r'https://[^\s]+',lambda m:r'\url{'+m.group(0).rstrip('.').replace(r'\_','_')+'}',content)
    content=content.replace(r'\end{document}',r'\section{Verified external references}\cite{garg2022,ovadia2019}\bibliographystyle{plain}\bibliography{REFERENCES}'+'\n'+r'\end{document}')
    content=content.replace(r'\_',r'\_\allowbreak{}')
    content=content.replace('**','')
    content=content.replace('Visual inspection of this automatically compiled edition remains pending. Scientific validation and sample reproduction are recorded separately.',
                            'Scientific validation, reproduction checks and visual review are recorded in the accompanying receipts.')
    content=re.sub(r'(\\begin\{longtable\}\{[^\n]*\}\n\\toprule\n)([^\n]+)(\n\\midrule)',
        lambda m:m.group(1)+m.group(2)+m.group(3)+'\n'+r'\endfirsthead'+'\n'+r'\toprule'+'\n'+m.group(2)+m.group(3)+'\n'+r'\endhead',content)
    tex.write_text(content,encoding='utf-8')
    (out/'REFERENCES.bib').write_text('''@misc{garg2022,title={Leveraging Unlabeled Data to Predict Out-of-Distribution Performance},author={Garg, Saurabh and others},year={2022},url={https://arxiv.org/abs/2201.04234}}
@inproceedings{ovadia2019,title={Can You Trust Your Model\'s Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift},author={Ovadia, Yaniv and others},year={2019},booktitle={NeurIPS}}
''',encoding='utf-8')
    print(out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.run,a.out)
