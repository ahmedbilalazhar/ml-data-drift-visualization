"""Create a concise report for the fresh-seed adaptation replication."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS={"v1_synthetic":"#087e8b","xor_synthetic":"#ff5a5f"}

def build(run:Path,out:Path):
    out.mkdir(parents=True,exist_ok=False)
    summary=pd.read_csv(run/"paired_bootstrap.csv")
    validation=pd.read_json(run/"validation.json",typ="series")
    seeds=int(validation["seeds_per_task"])
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True,constrained_layout=True)
    for ax,regime in zip(axes,("post","late_post")):
        d=summary[(summary.regime==regime)&(summary.metric=="accuracy")]
        for task in ("v1_synthetic","xor_synthetic"):
            g=d[d.task==task].sort_values("delay")
            if g.empty:continue
            y=[0,1,2]
            ax.errorbar(g["mean"],y,xerr=[g["mean"]-g.lower,g.upper-g["mean"]],fmt="o",
                        color=COLORS[task],capsize=3,label=task.replace("_synthetic",""))
        ax.axvline(0,color="#263a38",lw=1,ls="--");ax.set_yticks([0,1,2],["delay 0","delay 5","delay 20"])
        ax.invert_yaxis();ax.set_xlabel("Cumulative refit minus frozen accuracy\n(95% seed bootstrap interval)")
        ax.set_title("All post-onset" if regime=="post" else "Late post-onset")
        ax.grid(axis="x",alpha=.2);ax.legend(frameon=False)
    for ext in ("png","svg"):fig.savefig(out/f"adaptation_replication.{ext}",dpi=180 if ext=="png" else None)
    plt.close(fig)
    lines=["# Fresh-seed replication of delayed-label adaptation","",
      "**Ahmed Bilal (23I-2581) and Umar Behram (23I-2604)**  ",
      "Data Analysis and Visualization · Phase 3 supplementary experiment","",
      "**Protocol status:** Outcome-informed replication locked before its fresh seed run; selected C3/X3 cases were chosen after the earlier pilot. This is not external validation.","",
      "## Purpose","",
      f"The original adaptation pilot had 30 seeds per task. Its zero-delay C3 late-post gain was larger than its all-post estimate, and the XOR effects were small. This replication tests how those patterns vary over {seeds:,} fresh seeds per task using the same two synthetic conditions and cumulative logistic update rule. It does not introduce an updater or tune thresholds.","",
      "## Protocol and validation","",
      "The frozen and cumulative-refit arms start from the same 4,000 training rows and training-only scaler. The adaptive fit uses only unique monitoring labels released strictly before the next prediction. Same-clock predictions are flushed before release, including delay zero. At forecast batch 24, the preprediction label budgets are 11,500, 9,000 and 1,500 for delays 0, 5 and 20; for delay 20 none of those labels are post-onset. The queues drain without new inputs.","",
      f"Validation passed {int(validation['prediction_truth_joins']):,} prediction/truth batch checks, {int(validation['release_rows']):,} scheduled release rows, exact budgets, no future-label fit access, identical release IDs across delays and complete queue drainage. Online rows omit outcomes; evaluator rows are retrospective. Paired uncertainty resamples complete seeds (2,000 replicates).","",
      "## Paired accuracy differences","",
      "![Fresh-seed paired adaptation differences](adaptation_replication.png)","",
      "| Task / condition | Delay | Regime | Adapted minus frozen accuracy [95% seed interval] | Seeds |","|---|---:|---|---:|---:|"]
    wanted={("v1_synthetic","C3"),("xor_synthetic","X3")}
    for row in summary[(summary.metric=="accuracy") & summary.apply(lambda r:(r.task,r.condition) in wanted,axis=1)].sort_values(["task","delay","regime"]).itertuples(index=False):
        lines.append(f"| {row.task}/{row.condition} | {row.delay} | {row.regime} | {row.mean:+.5f} [{row.lower:+.5f}, {row.upper:+.5f}] | {row.defined_seeds} |")
    lines += ["","Intervals crossing zero do not establish a directional effect. The intervention cases are selected synthetic examples, not a broad test of adaptation.","",
      "## Interpretation limits","",
      "This replication changes only the fresh seed count. It uses one linear logistic classifier, one cumulative refit procedure and two outcome-informed synthetic conditional-change cases. The frozen comparator uses zero monitoring labels for fitting, so this is an access-and-update contrast, not an adaptive-method leaderboard at equal consumed-label budgets. The design does not claim calibration repair, calibrated harm probabilities, external validity, safety or analyst benefit. A possible reason a linear updater may struggle with XOR is its inability to represent the interaction; that mechanism is a hypothesis unless directly tested against a nonlinear updater.","",
      "## Reproduction","",
      "From the workspace root, use fresh paths:","","```powershell",
      "powershell -NoProfile -ExecutionPolicy Bypass `",
      "  -File .\\scripts\\reproduce_phase3_adaptation_replication.ps1 `",
      "  -RunDir results/phase3/adaptation_seed_replication_reproduction `",
      "  -ReportDir reports/phase3/adaptation_seed_replication_reproduction",
      "```","","Exact environment and hashes are recorded in the run. Prior final results and prepared archives are unchanged."]
    dest=out/"23I-2581_23I-2604_Phase3_Adaptation_Replication.md";dest.write_text("\n".join(lines)+"\n",encoding="utf-8");return dest

if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args();print(build(a.run,a.out))
