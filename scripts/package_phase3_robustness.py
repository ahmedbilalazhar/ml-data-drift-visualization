"""Package the two new robustness experiments without touching submitted archives."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"output/phase3"
COMPACT=OUT/"23I-2581_23I-2604_Phase3_Research_Robustness.zip"
FULL=OUT/"23I-2581_23I-2604_Phase3_Research_Robustness_Full.zip"
SUBMITTED=OUT/"23I-2581_23I-2604_Phase3.zip"
SUBMITTED_SHA="279d9867d1c75720a51df12caae2ac0b4d490641b83752e32642dcb87e1c8877"
V1=OUT/"IDM_Viz_Phase3_v1_submission.zip"
V1_SHA="ddf91fa89ab9a9d45c20059c1dbea5cfdd966f3eea62a605769a371ae55e0ebd"
V1_FULL=OUT/"IDM_Viz_Phase3_v1_full_reproducibility.zip"
V1_FULL_SHA="0baba912c5f3e48a8a08a606e2a77a303f67e884ddccfef0907046d94dca5519"


def sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()


def sha_bytes(data:bytes)->str:return hashlib.sha256(data).hexdigest()


def main()->None:
    for path,expected in ((SUBMITTED,SUBMITTED_SHA),(V1,V1_SHA),(V1_FULL,V1_FULL_SHA)):
        if sha(path)!=expected:raise RuntimeError(f"preserved archive checksum changed: {path.name}")
    if COMPACT.exists() or FULL.exists():
        raise FileExistsError("refusing to overwrite a previous supplementary archive")
    robust=ROOT/"results/phase3/large_seed_robustness_final"
    rr=ROOT/"reports/phase3/large_seed_robustness_final"
    sample=rr/"sample_reproduction/sample_reproduction_comparison.json"
    required=(robust/"validation.json",sample,
      rr/"23I-2581_23I-2604_Phase3_Robustness_Report.pdf")
    missing=[str(p) for p in required if not p.is_file()]
    if missing:raise FileNotFoundError("incomplete final evidence: "+", ".join(missing))

    with tempfile.TemporaryDirectory(prefix="phase3_robustness_pkg_") as tmp:
      stage=Path(tmp)/"stage";stage.mkdir()
      def copy(source:Path,target:str):
        dest=stage/target;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dest)
      def write(target:str,content:str):
        dest=stage/target;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(content,encoding="utf-8",newline="\n")
      # Short, roll-numbered reports, prior comparison PDFs and validation receipts.
      copy(rr/"23I-2581_23I-2604_Phase3_Robustness_Report.pdf","Robustness.pdf")
      copy(rr/"23I-2581_23I-2604_Phase3_Robustness_Report.md","Robustness.md")
      prior=ROOT/"reports/phase3/v2_final"
      prior_adapt=ROOT/"reports/phase3/adaptation_final"
      copy(prior/"23I-2581_23I-2604_Phase3_Report.pdf","Prior_Report.pdf")
      copy(prior_adapt/"23I-2581_23I-2604_Phase3_Adaptation_Report.pdf","Prior_Adaptation.pdf")
      copy(prior_adapt/"23I-2581_23I-2604_Phase3_Adaptation_Report.md","Prior_Adaptation.md")
      copy(ROOT/"docs/phase3/LARGE_SEED_ROBUSTNESS_PROTOCOL.md","Protocol_Robustness.md")
      copy(ROOT/"docs/phase3/V2_EXTENSION_MANIFEST.md","Protocol_Previous.md")
      decision=(ROOT/"docs/phase3/V2_DECISION_LOG.md").read_text(encoding="utf-8")
      decision=decision.replace("IDM-Viz","Phase 3 research").replace("v2","extension").replace("V2","Extension")
      write("Decisions.md",decision)
      copy(robust/"protocol_lock.json","Robustness_Lock.json")
      copy(robust/"environment.json","Robustness_Environment.json")
      copy(robust/"validation.json","Robustness_Validation.json")
      copy(robust/"analysis_source_amendment.json","Analysis_Amendment.json")
      copy(sample,"Sample_Reproduction.json")
      copy(rr/"pdf_render_record.json","Robustness_Render.json")
      copy(rr/"visual_review.json","Robustness_Visual.json")
      copy(ROOT/"requirements-phase3.txt","Requirements.txt")
      for name in ("paired_accuracy.png","paired_accuracy.svg","estimation_and_coverage.png",
                   "estimation_and_coverage.svg"):
        copy(rr/name,"Figures/"+name)
      # Complete seed vectors and paired uncertainty are included in the compact archive.
      for source,target in (("per_seed_condition.csv","Results/Seed_Condition.csv"),
          ("seed_bootstrap_summary.csv","Results/Seed_Bootstrap.csv"),
          ("paired_model_seed_differences.csv","Results/Model_Pairs.csv"),
          ("paired_model_bootstrap.csv","Results/Model_Bootstrap.csv"),
          ("input_detection_summary.csv","Results/Input_Detection.csv")):
        src=(rr/source if source in {"per_seed_condition.csv","seed_bootstrap_summary.csv",
          "paired_model_seed_differences.csv","paired_model_bootstrap.csv","input_detection_summary.csv"} else robust/source)
        copy(src,target)
      # Source aliases keep all archive paths short and free of historical version tags.
      code_src=stage/"Code/src/drift_monitoring/phase3";code_src.mkdir(parents=True,exist_ok=True)
      source_root=ROOT/"src/drift_monitoring/phase3"
      for name in ("core.py","replay.py","evaluation.py"):
        copy(source_root/name,"Code/src/drift_monitoring/phase3/"+name)
      v2_text=(source_root/"v2.py").read_text(encoding="utf-8")
      (code_src/"extension.py").write_text(v2_text.replace("IDM-Viz","Phase 3 research"),encoding="utf-8",newline="\n")
      write("Code/src/drift_monitoring/phase3/__init__.py",'"""Phase 3 research package."""\n')
      adaptation=(source_root/"adaptation.py").read_text(encoding="utf-8")
      adaptation=adaptation.replace("from .v2 import","from .extension import").replace("IDM-Viz","Phase 3 research")
      (code_src/"adaptation.py").write_text(adaptation,encoding="utf-8",newline="\n")
      code_scripts=stage/"Code/scripts";code_scripts.mkdir(parents=True,exist_ok=True)
      script_names=("run_phase3_large_seed.py","evaluate_phase3_large_seed.py",
        "report_phase3_large_seed.py","compare_phase3_large_seed_sample.py",
        )
      for name in script_names:
        content=(ROOT/"scripts"/name).read_text(encoding="utf-8")
        content=content.replace("drift_monitoring.phase3.v2","drift_monitoring.phase3.extension")
        content=content.replace("src/drift_monitoring/phase3/v2.py","src/drift_monitoring/phase3/extension.py")
        content=content.replace("IDM-Viz","Phase 3 research")
        (code_scripts/name).write_text(content,encoding="utf-8",newline="\n")
      render=(ROOT/"scripts/render_phase3_v2_pdf.py").read_text(encoding="utf-8")
      (code_scripts/"render.py").write_text(render.replace("IDM-Viz","Phase 3 research"),encoding="utf-8",newline="\n")
      name="reproduce_phase3_large_seed.ps1"
      content=(ROOT/"scripts"/name).read_text(encoding="utf-8").replace("render_phase3_v2_pdf.py","render.py")
      (code_scripts/name).write_text(content,encoding="utf-8",newline="\n")
      copy(ROOT/"docs/phase3/LARGE_SEED_ROBUSTNESS_PROTOCOL.md","Code/docs/phase3/LARGE_SEED_ROBUSTNESS_PROTOCOL.md")
      # Preserve the original manifest as a neutral protocol record.
      manifest=(ROOT/"docs/phase3/V2_EXTENSION_MANIFEST.md").read_text(encoding="utf-8")
      manifest=manifest.replace("IDM-Viz","Phase 3 research").replace("v2","extension").replace("V2","Extension")
      write("Code/docs/phase3/EXTENSION_PROTOCOL.md",manifest)
      for name,target in (("test_v2_extension_contract.py","test_extension_contract.py"),
          ("test_adaptation_contract.py","test_adaptation_contract.py"),
          ("test_large_seed_robustness.py","test_large_seed_robustness.py")):
        content=(ROOT/"tests/phase3"/name).read_text(encoding="utf-8")
        content=content.replace("drift_monitoring.phase3.v2","drift_monitoring.phase3.extension")
        (stage/"Code/tests").mkdir(parents=True,exist_ok=True)
        (stage/"Code/tests"/target).write_text(content,encoding="utf-8",newline="\n")
      write("Reproduce.md", """# Reproduction and contents

This supplementary archive contains the large fresh-seed model-sensitivity study, alongside the prior empirical reports for comparison. The new task remains synthetic; no human-study or natural-domain result is claimed. The earlier 30-seed adaptation pilot is included for context; a new 1,500-seed adaptation replication was not run.

The compact archive includes reports, figures, complete per-seed summaries, paired seed vectors, bootstrap summaries, protocols, environment records, source, tests, and validation. The full archive adds every saved online/evaluator batch row. Large model streams are reproducibly regenerated from their locked seed lists rather than stored as per-instance caches.

From the `Code` directory, run the PowerShell wrappers with fresh output directories:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\\scripts\\reproduce_phase3_large_seed.ps1 -RunDir results/seed_reproduction -ReportDir reports/seed_reproduction
```

Use the same scientific packages listed in `Requirements.txt`. The report PDFs are supplementary prepared research artifacts. Nothing in this package is an upload receipt.
""")
      write("README.txt", """DATA ANALYSIS AND VISUALIZATION - PHASE 3 SUPPLEMENT
Ahmed Bilal (23I-2581) and Umar Behram (23I-2604)

Contents: two empirical reports, prior comparison PDFs, figures, result tables, protocols, source, tests, exact environment information and validation. The additional task is synthetic. Results are outcome-informed robustness replications, not external validation. No instructor approval, human participants or analyst-benefit result is claimed.

The prepared archives are supplemental. The submitted archive and the earlier prepared archives are preserved unchanged. This is not an upload receipt.
""")
      write("Adaptation_Note.txt", "The prior 30-seed adaptation pilot is included for context. A protocol for a 1,500-seed-per-task replication was drafted, but that additional run was not conducted; time was reserved for the model-sensitivity robustness study, clean reproduction, reporting and package verification. Do not call the pilot a fresh-seed replication.\n")

      # Compact payload checksum inventory, then assemble archive.
      inventory={}
      for p in sorted(x for x in stage.rglob("*") if x.is_file()):
        rel=p.relative_to(stage).as_posix()
        if any(tag in rel.lower() for tag in ("v1","v2","idm-viz")):
          raise RuntimeError("archive member name is not neutral: "+rel)
        inventory[rel]={"bytes":p.stat().st_size,"sha256":sha(p)}
      (stage/"Checksums.json").write_text(json.dumps(inventory,indent=2),encoding="utf-8")

      # Full archive includes the entire compact payload and complete new run outputs.
      full_stage=Path(tmp)/"full";shutil.copytree(stage,full_stage)
      for source,prefix in ((robust,"Results/Robustness"),(rr,"Review/Robustness")):
        for p in source.rglob("*"):
          if not p.is_file():continue
          rel="/".join(part.replace("v1","original").replace("v2","extension") for part in p.relative_to(source).parts)
          dest=full_stage/prefix/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
      inv={}
      for p in sorted(x for x in full_stage.rglob("*") if x.is_file() and x.name!="Checksums.json"):
        rel=p.relative_to(full_stage).as_posix()
        if any(tag in rel.lower() for tag in ("v1","v2","idm-viz")):
          raise RuntimeError("full archive member name is not neutral: "+rel)
        inv[rel]={"bytes":p.stat().st_size,"sha256":sha(p)}
      (full_stage/"Checksums.json").write_text(json.dumps(inv,indent=2),encoding="utf-8")

      OUT.mkdir(parents=True,exist_ok=True)
      for staged,target in ((stage,COMPACT),(full_stage,FULL)):
        with zipfile.ZipFile(target,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as zf:
          for p in sorted(x for x in staged.rglob("*") if x.is_file()):
            zf.write(p,p.relative_to(staged).as_posix())
        with zipfile.ZipFile(target) as zf:
          bad=zf.testzip()
          if bad:raise RuntimeError(f"ZIP CRC failure in {target.name}: {bad}")
          names=zf.namelist();inventory=json.loads(zf.read("Checksums.json"))
          if len(inventory)+1!=len(names):raise RuntimeError("checksum inventory member count mismatch")
          for name,record in inventory.items():
            if sha_bytes(zf.read(name))!=record["sha256"]:raise RuntimeError("member hash mismatch: "+name)
        receipt={"archive":str(target),"bytes":target.stat().st_size,"sha256":sha(target),
                 "members":len(names),"crc":"passed","member_sha256":"passed","uploaded":False,
                 "submitted_archive_sha256":SUBMITTED_SHA,"preserved_v1_archive_sha256":V1_SHA,
                 "preserved_full_archive_sha256":V1_FULL_SHA}
        target.with_suffix(".receipt.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
        print(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
