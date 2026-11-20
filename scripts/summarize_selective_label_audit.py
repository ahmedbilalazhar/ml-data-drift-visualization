"""Seed-bootstrap summary, figures and TeX report for selective-label audit."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument("--run",type=Path,default=ROOT/"results/phase3/selective_label_audit_20261117_v2")
 p.add_argument("--output",type=Path,default=ROOT/"reports/phase3/selective_label_audit_20261117_v5")
 p.add_argument("--bootstrap",type=int,default=5000)
 a=p.parse_args();a.run=a.run.resolve();a.output=a.output.resolve()
 if a.output.exists():raise FileExistsError(f"refusing to overwrite {a.output}")
 a.output.mkdir(parents=True);frame=pd.read_csv(a.run/"batch_estimates.csv")
 # Seeds are independent simulation units. First average the 96 dependent batches.
 byseed=frame.groupby(["task","mechanism","policy","seed"],as_index=False).agg(
  full_stream_accuracy=("full_stream_accuracy","mean"),selected_naive_accuracy=("selected_naive_accuracy","mean"),
  horvitz_thompson_accuracy=("horvitz_thompson_accuracy","mean"),selected_rows=("selected_rows","mean"),
  ht_ci_covers=("ht_ci_covers","mean"),effective_sample_size=("effective_sample_size","mean"))
 rng=np.random.default_rng(20261117);rows=[]
 for (task,mech,policy),g in byseed.groupby(["task","mechanism","policy"],sort=True):
  for metric in ["full_stream_accuracy","selected_naive_accuracy","horvitz_thompson_accuracy","selected_rows","ht_ci_covers","effective_sample_size"]:
   v=g[metric].dropna().to_numpy(float)
   if not len(v):continue
   boot=v[rng.integers(0,len(v),(a.bootstrap,len(v)))].mean(axis=1)
   rows.append({"task":task,"mechanism":mech,"policy":policy,"metric":metric,"seeds":len(v),
    "mean":float(v.mean()),"seed_sd":float(v.std(ddof=1)),"lower":float(np.quantile(boot,.025)),"upper":float(np.quantile(boot,.975))})
 summary=pd.DataFrame(rows);summary.to_csv(a.output/"seed_bootstrap_summary.csv",index=False)
 colors={"targeted_10pct":"#D06A45","stratified_random_audit":"#176B87","simple_random_budget_matched":"#6A6C9B"}
 labels={"targeted_10pct":"Targeted only (naive)","stratified_random_audit":"Targeted + random audit","simple_random_budget_matched":"Uniform budget matched"}
 ests=[("full_stream_accuracy","Full-stream evaluator truth","#172B4D","D"),
       ("selected_naive_accuracy","Selected-label naive","#AAB3BD","o"),
       ("horvitz_thompson_accuracy","Horvitz–Thompson","#176B87","s")]
 plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,"axes.spines.right":False,
  "figure.facecolor":"#F4F7FA","axes.facecolor":"white","savefig.facecolor":"#F4F7FA","axes.edgecolor":"#D6DEE8",
  "axes.labelcolor":"#344054","text.color":"#172B4D","xtick.color":"#475467","ytick.color":"#475467","grid.color":"#E7ECF2"})
 keys=[("linear_probabilistic","M0_stationary"),("linear_probabilistic","M4_abrupt_conditional_reversal"),
       ("five_region","M0_stationary"),("five_region","M4_abrupt_conditional_reversal")]
 fig,axs=plt.subplots(2,2,figsize=(11.2,7.3),constrained_layout=True)
 for ax,(task,mech) in zip(axs.flat,keys):
  gold=summary[(summary.task==task)&(summary.mechanism==mech)&(summary.metric=="full_stream_accuracy")]
  base=gold[gold.policy=="stratified_random_audit"].iloc[0]
  for j,policy in enumerate(colors):
   for metric,name,color,marker in ests[1:]:
    q=summary[(summary.task==task)&(summary.mechanism==mech)&(summary.policy==policy)&(summary.metric==metric)]
    if q.empty:continue
    row=q.iloc[0];offset={"selected_naive_accuracy":-.12,"horvitz_thompson_accuracy":.12}[metric]
    ax.errorbar(j+offset,row["mean"],yerr=[[row["mean"]-row.lower],[row.upper-row["mean"]]],fmt=marker,
     color=color if metric=="selected_naive_accuracy" else colors[policy],ms=6,capsize=3,lw=1.5)
  ax.axhline(base["mean"],color="#172B4D",ls="--",lw=1.3,label="Full-stream accuracy")
  ax.set_xticks(range(3),["Targeted only","Targeted + audit","Uniform audit"],rotation=12,ha="right")
  ax.set_ylim(.25,.9);ax.set_ylabel("Accuracy estimate (95% seed-bootstrap CI)");ax.grid(axis="y",alpha=.55)
  task_name="Linear" if task=="linear_probabilistic" else "Five-region"
  mech_name="Stationary" if mech=="M0_stationary" else "Conditional reversal"
  ax.set_title(f"{task_name} · {mech_name}",weight="bold")
  from matplotlib.lines import Line2D
  handles=[Line2D([],[],color="#172B4D",ls="--",label="Full-stream truth"),
           Line2D([],[],color="#AAB3BD",marker="o",ls="",label="Naive selected-label mean"),
           Line2D([],[],color="#176B87",marker="s",ls="",label="Horvitz–Thompson estimate")]
  ax.legend(handles=handles,frameon=False,fontsize=7,loc="lower left")
 fig.suptitle("Known-probability random audits correct targeted-label bias",fontsize=14,weight="bold")
 fig.savefig(a.output/"selective_label_bias.png",dpi=250,bbox_inches="tight");fig.savefig(a.output/"selective_label_bias.svg",bbox_inches="tight");plt.close(fig)
 def get(task,mech,policy,metric):return summary[(summary.task==task)&(summary.mechanism==mech)&(summary.policy==policy)&(summary.metric==metric)].iloc[0]
 gold=get("linear_probabilistic","M4_abrupt_conditional_reversal","stratified_random_audit","full_stream_accuracy")
 targ=get("linear_probabilistic","M4_abrupt_conditional_reversal","targeted_10pct","selected_naive_accuracy")
 combnaive=get("linear_probabilistic","M4_abrupt_conditional_reversal","stratified_random_audit","selected_naive_accuracy")
 combht=get("linear_probabilistic","M4_abrupt_conditional_reversal","stratified_random_audit","horvitz_thompson_accuracy")
 randht=get("linear_probabilistic","M4_abrupt_conditional_reversal","simple_random_budget_matched","horvitz_thompson_accuracy")
 cover=get("linear_probabilistic","M4_abrupt_conditional_reversal","stratified_random_audit","ht_ci_covers")
 sampled=get("linear_probabilistic","M4_abrupt_conditional_reversal","stratified_random_audit","selected_rows")
 tex=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}\usepackage[T1]{fontenc}\usepackage{lmodern,microtype,booktabs,graphicx,xcolor,hyperref,tcolorbox,parskip}
\setlength{\emergencystretch}{3em}
\definecolor{ink}{HTML}{172B4D}\definecolor{teal}{HTML}{176B87}\definecolor{coral}{HTML}{D06A45}\definecolor{mist}{HTML}{F1F6F8}
\hypersetup{colorlinks=true,linkcolor=teal,urlcolor=teal}
\title{\vspace{-2em}\color{ink}\bfseries IDM-Viz selective-label bias audit\\[-.2em]\large Feature-targeted labels with a known-probability random audit}
\author{Phase 3 controlled synthetic addendum}\date{17 November 2026}
\begin{document}\maketitle\vspace{-1em}
\begin{tcolorbox}[colback=mist,colframe=teal,boxrule=0pt,leftrule=2mm]\textbf{Scope.} Simulated acquisition on two synthetic families and two conditions. This is not a prospective natural-stream label audit.
\end{tcolorbox}
\section*{Protocol}
We evaluated 30 fixed seeds, linear and five-region generators, stationary and abrupt conditional reversal, and 96 consecutive batches of 500 rows. Acquisition sees only the frozen classifier's confidence, never labels. A targeted-only policy selects exactly the 10\% least-confident rows; its inclusion probability is zero outside that set, so population accuracy is not identifiable from these labels and no IPW estimate is reported. The combined policy includes that certainty sample and Bernoulli-samples each remaining row with probability 0.10 (logged \(\pi_i=1\) or \(0.10\)). A simple random audit with probability 0.19 matches the combined policy's expected 19\% label budget. We compute the untruncated Horvitz--Thompson accuracy \(N^{-1}\sum_{i\in S} c_i/\pi_i\) and its Bernoulli design variance. Inference summaries average batches within seed and bootstrap the 30 seeds (\BOOT{} replicates). Batch-level normal intervals are also checked against fully available synthetic truth, used only after selection for evaluation.
\section*{Findings}
For linear conditional reversal, the evaluator's mean full-stream accuracy was \GOLD{} (95\% seed-bootstrap CI \GOLDLO{}--\GOLDHI{}). The targeted-only selected-label estimate was \TARGET{} (\TARGETLO{}--\TARGETHI{}), showing that the direction and size of selection bias depend on the stream. With the random-audit component, the naive selected-label estimate was \COMBNAIVE{} (\CNAIVELO{}--\CNAIVEHI{}), while the logged-probability Horvitz--Thompson estimate was \COMBHT{} (\CHTLO{}--\CHTHI{}). The budget-matched simple-random Horvitz--Thompson estimate was \RANDHT{} (\RHTLO{}--\RHTHI{}). The combined policy sampled a mean of \SAMPLED{} labels per 500-row batch. Its normal design interval covered the finite-batch truth in \COVER{} of batch-seed cases. This empirical coverage is below or near nominal and is not a guarantee.
\begin{center}\includegraphics[width=\linewidth]{selective_label_bias.png}\end{center}
\section*{Interpretation and limits}
Targeted labels can efficiently find errors but their unweighted mean does not estimate population accuracy. A known-probability random component restores positivity and permits design-based weighting in this simulated setting; it does not make the deterministic targeted-only sample representative. Horvitz--Thompson estimates and normal intervals are intentionally not clipped to \([0,1]\). Only these synthetic selection laws, inclusion probabilities, fixed classifier, and batch sizes are evaluated. Natural acquisition behavior, label costs, nonresponse, misspecified probabilities, and calibration refits from selected outcomes remain untested. The logged outcomes contain only selected rows; all unseen truth is used solely for evaluator scoring.
\section*{Reproduction}
Run these commands from the project root:
\begin{verbatim}
python scripts/run_selective_label_audit.py
python scripts/summarize_selective_label_audit.py
\end{verbatim}
Full selected-label inclusion probabilities, batch estimates, seed-bootstrap summaries, source hashes and the editable TeX source are included.
\end{document}
'''
 vals={"\\BOOT":str(a.bootstrap),"\\GOLD":f"{gold['mean']:.3f}","\\GOLDLO":f"{gold['lower']:.3f}","\\GOLDHI":f"{gold['upper']:.3f}",
  "\\TARGET":f"{targ['mean']:.3f}","\\TARGETLO":f"{targ['lower']:.3f}","\\TARGETHI":f"{targ['upper']:.3f}",
  "\\COMBNAIVE":f"{combnaive['mean']:.3f}","\\CNAIVELO":f"{combnaive['lower']:.3f}","\\CNAIVEHI":f"{combnaive['upper']:.3f}",
  "\\COMBHT":f"{combht['mean']:.3f}","\\CHTLO":f"{combht['lower']:.3f}","\\CHTHI":f"{combht['upper']:.3f}",
  "\\RANDHT":f"{randht['mean']:.3f}","\\RHTLO":f"{randht['lower']:.3f}","\\RHTHI":f"{randht['upper']:.3f}",
  "\\SAMPLED":f"{sampled['mean']:.1f}","\\COVER":f"{cover['mean']:.3f}"}
 for old,new in sorted(vals.items(),key=lambda q:-len(q[0])):
  tex=tex.replace(old+"{}",new);tex=tex.replace(old,new)
 texpath=a.output/"SELECTIVE_LABEL_AUDIT.tex";texpath.write_text(tex,encoding="utf-8")
 subprocess.run(["tectonic","-X","compile",str(texpath),"--outdir",str(a.output)],check=True)
 sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 inputs=[a.run/"audit_receipt.json",a.run/"batch_estimates.csv",a.run/"selected_label_log.csv"]
 outputs=[a.output/n for n in ["seed_bootstrap_summary.csv","selective_label_bias.png","selective_label_bias.svg","SELECTIVE_LABEL_AUDIT.tex","SELECTIVE_LABEL_AUDIT.pdf"]]
 (a.output/"REPORT_RECEIPT.json").write_text(json.dumps({"status":"complete","bootstrap_replicates":a.bootstrap,"bootstrap_unit":"evaluation seed",
  "input_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in inputs},"output_sha256":{p.relative_to(ROOT).as_posix():sha(p) for p in outputs}},indent=2)+"\n",encoding="utf-8")
 print(f"Wrote seed-bootstrap table, vector/raster figures and TeX report to {a.output}")
if __name__=="__main__":main()
