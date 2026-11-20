"""Build and clean-extraction-validate the matched-budget ELEC2 calibration addendum."""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,zipfile
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];PY=Path('C:/Users/hp/anaconda3/python.exe')
OUT=ROOT/'output/phase3/IDM_Viz_ELEC2_Calibration_Family_Addendum_20261116_v2.zip'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 files={}
 def add(rel):
  p=ROOT/rel
  if not p.is_file():raise FileNotFoundError(p)
  files[p.relative_to(ROOT).as_posix()]=p
 for f in ['scripts/run_elec2_platt_recalibration.py','scripts/evaluate_elec2_platt_recalibration.py','scripts/report_elec2_calibration_family.py',
           'scripts/run_elec2_temporal_validation.py','scripts/evaluate_elec2_temporal_validation.py','scripts/package_elec2_calibration_addendum.py',
           'src/drift_monitoring/phase3/__init__.py','src/drift_monitoring/phase3/core.py',
           'tests/phase3/test_elec2_platt_recalibration.py','docs/phase3/ELEC2_TEMPORAL_EXTENSION_20261116.md',
           'docs/phase3/ELEC2_DELAYED_RECALIBRATION_20261116.md','docs/phase3/ELEC2_CALIBRATION_FAMILY_EXTENSION_20261116.md',
           'docs/phase3/PHASE2_TRACEABILITY.md','docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md',
           'docs/phase3/DECISION_STRENGTHENING_CHANGELOG.md','docs/phase3/23I-2581_23I-2604_Phase3_Reproduction.md',
           'docs/23i-2604_23i-2581DAV-phase2.docx','docs/Phase 03 - Implementation.pdf',
           'output/phase3/23I-2581_23I-2604_Phase3.zip',
           'output/phase3/IDM_Viz_Phase3_Final_Strengthening_20261116_v13.receipt.json',
           'data/external/openml_151/electricity.arff','requirements-phase3.txt','pyproject.toml']:
  add(f)
 for folder in ['results/phase3/elec2_temporal_20261116_v2','results/phase3/elec2_recalibration_20261116_v1',
               'results/phase3/elec2_platt_recalibration_20261116_v1','results/phase3/elec2_calibration_family_20261116_v1',
               'reports/phase3/elec2_calibration_family_20261116_v1']:
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tmp'):add(p.relative_to(ROOT))
 protected={
  'docs/23i-2604_23i-2581DAV-phase2.docx':'786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4',
  'docs/Phase 03 - Implementation.pdf':'9a94654bdb772b03d5228993f2cc7eaf1b8552f41c462ea84a4649bb0a828115',
  'output/phase3/23I-2581_23I-2604_Phase3.zip':'279d9867d1c75720a51df12caae2ac0b4d490641b83752e32642dcb87e1c8877',
  'output/phase3/IDM_Viz_Phase3_Final_Strengthening_20261116_v13.zip':'4010ca6baf3cd67ed7516e16fcffff31558c97e17cfc92409f08ef25e0d33c2f'}
 assert all(sha(ROOT/k)==v for k,v in protected.items())
 inventory={k:dict(bytes=v.stat().st_size,sha256=sha(v)) for k,v in files.items()}
 readme='''# ELEC2 matched-budget calibration-family addendum

This small supplementary archive builds on the independently validated parent Phase 3 package v13 (hash in PRESERVATION.json). It adds a matched-budget comparison of frozen isotonic, cumulative isotonic and cumulative Platt-style calibration on the same 55-batch ELEC2 stream. It is an exploratory negative mitigation result, not a general calibration claim.

The Phase 2 submitted DOCX and pre-deadline archive are preserved unchanged. No external upload or submission is made.

Reproduce with `C:/Users/hp/anaconda3/python.exe` from the extracted root:
  python scripts/run_elec2_platt_recalibration.py --run results/phase3/elec2_temporal_20261116_v2 --out results/platt_NEW
  python scripts/evaluate_elec2_platt_recalibration.py --run results/phase3/elec2_temporal_20261116_v2 --platt results/platt_NEW --isotonic results/phase3/elec2_recalibration_20261116_v1 --out results/calibration_NEW
  python scripts/report_elec2_calibration_family.py --run results/phase3/elec2_temporal_20261116_v2 --platt results/platt_NEW --isotonic results/phase3/elec2_recalibration_20261116_v1 --results results/calibration_NEW --out reports/calibration_NEW
  python -m pytest tests/phase3/test_elec2_platt_recalibration.py -q

Protocol and interpretation boundary: docs/phase3/ELEC2_CALIBRATION_FAMILY_EXTENSION_20261116.md. The adjacent JSON receipt records exact clean-extraction checks.
'''
 with zipfile.ZipFile(OUT,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
  for k,v in files.items():z.write(v,k)
  z.writestr('REPRODUCE.md',readme);z.writestr('MEMBER_SHA256.json',json.dumps(inventory,indent=2));z.writestr('PRESERVATION.json',json.dumps(protected,indent=2))
 with zipfile.ZipFile(OUT) as z:
  assert z.testzip() is None
  for k,v in inventory.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256'],k
  with tempfile.TemporaryDirectory(prefix='elec2_calibration_extract_') as td:
   z.extractall(td);work=Path(td);env=os.environ.copy();env['PYTHONPATH']=str(work)+os.pathsep+str(work/'src');env['PYTHONNOUSERSITE']='1';runs=[]
   cmds=[('tests',[str(PY),'-m','pytest','tests/phase3/test_elec2_platt_recalibration.py','-q']),
     ('platt_runner',[str(PY),'scripts/run_elec2_platt_recalibration.py','--run','results/phase3/elec2_temporal_20261116_v2','--out','results/platt_EXTRACTED']),
     ('evaluator',[str(PY),'scripts/evaluate_elec2_platt_recalibration.py','--run','results/phase3/elec2_temporal_20261116_v2','--platt','results/platt_EXTRACTED','--isotonic','results/phase3/elec2_recalibration_20261116_v1','--out','results/calibration_EXTRACTED']),
     ('report',[str(PY),'scripts/report_elec2_calibration_family.py','--run','results/phase3/elec2_temporal_20261116_v2','--platt','results/platt_EXTRACTED','--isotonic','results/phase3/elec2_recalibration_20261116_v1','--results','results/calibration_EXTRACTED','--out','reports/calibration_EXTRACTED'])]
   for name,cmd in cmds:
    r=subprocess.run(cmd,cwd=work,env=env,text=True,capture_output=True,timeout=900);runs.append(dict(stage=name,command=cmd,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
    if r.returncode:raise RuntimeError(json.dumps(runs,indent=2))
   checks=[]
   for d in (0,5,20):
    for f in [f'delay_{d}_fit_audit.csv']:
     saved=f'results/phase3/elec2_platt_recalibration_20261116_v1/{f}';fresh='results/platt_EXTRACTED/'+f
     pd.testing.assert_frame_equal(pd.read_csv(work/saved),pd.read_csv(work/fresh),check_exact=True);checks.append(saved+' exact')
   for f in ['per_batch_metrics.csv','summary.csv']:
    saved='results/phase3/elec2_calibration_family_20261116_v1/'+f;fresh='results/calibration_EXTRACTED/'+f
    pd.testing.assert_frame_equal(pd.read_csv(work/saved),pd.read_csv(work/fresh),check_exact=True);checks.append(saved+' exact')
   for f in ['ELEC2_CALIBRATION_FAMILY.tex','calibration_family.png']:
    saved='reports/phase3/elec2_calibration_family_20261116_v1/'+f;fresh='reports/calibration_EXTRACTED/'+f
    if f.endswith('.tex'):assert (work/saved).read_bytes()==(work/fresh).read_bytes()
    else:
     from PIL import Image
     assert np.array_equal(np.asarray(Image.open(work/saved).convert('RGBA')),np.asarray(Image.open(work/fresh).convert('RGBA')))
    checks.append(saved+' clean-extraction report exact')
 assert all(sha(ROOT/k)==v for k,v in protected.items())
 receipt=dict(status='passed',archive=str(OUT),archive_sha256=sha(OUT),bytes=OUT.stat().st_size,members=len(inventory)+3,
   crc='passed',member_hashes='passed',clean_extraction='passed',fresh_commands=runs,exact_result_comparisons=checks,preserved_hashes=protected,uploaded=False)
 OUT.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
