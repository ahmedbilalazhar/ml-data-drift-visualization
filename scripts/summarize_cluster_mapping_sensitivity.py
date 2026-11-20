"""Seed-bootstrap summary, figures and TeX report for cluster mapping sensitivity."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "results/phase3/cluster_mapping_sensitivity_20261117_v2")
    p.add_argument("--output", type=Path, default=ROOT / "reports/phase3/cluster_mapping_sensitivity_20261117_v1")
    p.add_argument("--bootstrap", type=int, default=5000)
    a = p.parse_args(); a.run = a.run.resolve(); a.output = a.output.resolve()
    if a.output.exists(): raise FileExistsError(f"refusing to overwrite {a.output}")
    a.output.mkdir(parents=True)
    raw = pd.read_csv(a.run / "cluster_mapping_sensitivity.csv")
    metrics = ["ari", "many_to_one_macro_f1", "region2_ap", "region2_brier", "region2_recall", "region2_precision"]
    rng = np.random.default_rng(261117)
    rows=[]
    for (rep,k), g in raw.groupby(["representation","clusters"], sort=True):
        for metric in metrics:
            vals=g.groupby("seed")[metric].mean().to_numpy(float); vals=vals[np.isfinite(vals)]
            boot=vals[rng.integers(0,len(vals),(a.bootstrap,len(vals)))].mean(axis=1)
            rows.append({"representation":rep,"clusters":int(k),"metric":metric,"seeds":len(vals),
                         "mean":float(vals.mean()),"sd":float(vals.std(ddof=1)),
                         "lower":float(np.quantile(boot,.025)),"upper":float(np.quantile(boot,.975))})
    summary=pd.DataFrame(rows); summary.to_csv(a.output/"seed_bootstrap_summary.csv",index=False)
    colors={"scaled_full":"#176B87","frozen_pca2":"#C46B43","frozen_pca3":"#6A6C9B"}
    labels={"scaled_full":"Scaled full space","frozen_pca2":"Frozen PCA-2","frozen_pca3":"Frozen PCA-3"}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(11.5,4.0),constrained_layout=True)
    for ax,metric,title in zip(axes,["ari","many_to_one_macro_f1","region2_ap"],
                               ["Agreement with generating regions","Many-to-one macro F1","Region 2 ranking (average precision)"]):
        d=summary[summary.metric==metric]
        for rep in colors:
            q=d[d.representation==rep].sort_values("clusters")
            ax.errorbar(q.clusters,q["mean"],yerr=[q["mean"]-q.lower,q.upper-q["mean"]],
                        marker="o",capsize=3,lw=1.7,color=colors[rep],label=labels[rep])
        ax.set(title=title,xlabel="Clusters",xticks=[3,5,10]); ax.grid(axis="y",alpha=.22)
    axes[0].set_ylabel("Mean (95% seed-bootstrap CI)"); axes[1].legend(frameon=False,fontsize=8)
    fig.suptitle("Region localization stays weak across frozen representations",weight="bold",fontsize=13)
    fig.savefig(a.output/"cluster_mapping_sensitivity.png",dpi=260,bbox_inches="tight")
    fig.savefig(a.output/"cluster_mapping_sensitivity.svg",bbox_inches="tight"); plt.close(fig)
    def cell(rep,k,m): return summary[(summary.representation==rep)&(summary.clusters==k)&(summary.metric==m)].iloc[0]
    full5=cell("scaled_full",5,"many_to_one_macro_f1"); pca2_5=cell("frozen_pca2",5,"many_to_one_macro_f1")
    pca3_5=cell("frozen_pca3",5,"many_to_one_macro_f1")
    ari=cell("scaled_full",5,"ari"); ap=cell("scaled_full",5,"region2_ap")
    tex=r'''\documentclass[10pt]{article}
\usepackage[margin=.8in]{geometry}\usepackage{graphicx,booktabs,xcolor,microtype,parskip}
\definecolor{navy}{HTML}{18334B}\definecolor{teal}{HTML}{176B87}
\usepackage[colorlinks=true,linkcolor=teal,urlcolor=teal]{hyperref}
\title{\vspace{-2em}\color{navy}\bfseries IDM-Viz cluster localization sensitivity\\[-.2em]\large Frozen representations and many-to-one historical mappings}
\author{Post-deadline empirical addendum}\date{17 November 2026}
\begin{document}\maketitle\vspace{-1.1em}
\section*{Protocol}
We tested 30 already locked evaluation seeds, five generating regions, and 3, 5, or 10 clusters. Cluster models, standardization, and PCA bases were fit on historical training features only. Region mappings and per-cluster region probabilities were estimated from those same historical rows. Monitoring region IDs were used only for scoring. We compare scaled full-feature clustering with frozen PCA-2 and PCA-3 clustering. The many-to-one mapping permits several clusters to represent one region; Jeffreys-style additive smoothing (0.5 per cell) defines the cluster-to-region probabilities. Intervals resample seeds (\BOOTSTRAP{} replicates), not rows. This is an exploratory diagnostic on the locked seed bank; the primary detector and original results are unchanged.
\section*{Findings}
At five clusters, full-space many-to-one macro F1 was \FULLFONE{} (95\% seed-bootstrap CI \FULLLO{}--\FULLHI{}); PCA-2 was \PCA2{} and PCA-3 was \PCA3{}. Five-cluster adjusted Rand index in the full space was \ARI{} (CI \ARILO{}--\ARIHI{}). Region-2 average precision was \AP{} (CI \APLO{}--\APHI{}). Allowing many-to-one maps avoids the artificial unmapped-cluster penalty, but it does not make region identities recoverable: region-2 precision/recall and ranking remain weak. PCA does not improve the five-cluster macro F1 in this experiment. These results support exposing cluster ambiguity and composition rather than presenting clusters as known generating regions.
\begin{center}\includegraphics[width=\linewidth]{cluster_mapping_sensitivity.png}\end{center}
\section*{Limits and implication}
This sensitivity analysis does not establish a general localization method, causal attribution, or natural-stream regional validity. Region 2 has approximately one-fifth prevalence, so average precision must be read against that baseline. The results are exploratory, use one designed five-region generator, and do not justify retuning the primary method. Cluster assignments remain descriptive summaries. Raw seed-level metrics, historical contingency tables, probability maps, environment details, and SHA-256 receipts accompany this report.
\end{document}
'''
    rep={"\\BOOTSTRAP":str(a.bootstrap),"\\FULLFONE":f"{full5['mean']:.3f}","\\FULLLO":f"{full5['lower']:.3f}","\\FULLHI":f"{full5['upper']:.3f}",
         "\\PCA2":f"{pca2_5['mean']:.3f}","\\PCA3":f"{pca3_5['mean']:.3f}","\\ARI":f"{ari['mean']:.3f}","\\ARILO":f"{ari['lower']:.3f}","\\ARIHI":f"{ari['upper']:.3f}",
         "\\AP":f"{ap['mean']:.3f}","\\APLO":f"{ap['lower']:.3f}","\\APHI":f"{ap['upper']:.3f}"}
    for old,new in sorted(rep.items(),key=lambda x:-len(x[0])): tex=tex.replace(old,new)
    (a.output/"CLUSTER_MAPPING_SENSITIVITY.tex").write_text(tex,encoding="utf-8")
    sources=[a.run/"audit_receipt.json",a.run/"cluster_mapping_sensitivity.csv",Path(__file__)]
    outputs=[a.output/n for n in ["seed_bootstrap_summary.csv","cluster_mapping_sensitivity.png","cluster_mapping_sensitivity.svg","CLUSTER_MAPPING_SENSITIVITY.tex"]]
    sha=lambda x:hashlib.sha256(x.read_bytes()).hexdigest()
    rec={"status":"complete","bootstrap_replicates":a.bootstrap,"bootstrap_unit":"seed",
         "input_sha256":{x.relative_to(ROOT).as_posix():sha(x) for x in sources},
         "output_sha256":{x.relative_to(ROOT).as_posix():sha(x) for x in outputs}}
    (a.output/"REPORT_RECEIPT.json").write_text(json.dumps(rec,indent=2)+"\n",encoding="utf-8")
    print(f"Wrote bootstrap tables, figures and TeX report to {a.output}")

if __name__=="__main__": main()
