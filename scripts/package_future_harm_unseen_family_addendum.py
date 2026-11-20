"""Package the held-out synthetic future-harm law and clean-rerun both onset schedules."""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,zipfile
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];PY=Path('C:/Users/hp/anaconda3/python.exe')
OUT=ROOT/'output/phase3/IDM_Viz_Future_Harm_Unseen_Family_Addendum_20261116_v1.zip'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 if OUT.exists():raise FileExistsError(OUT)
 files={}
 def add(rel):
  p=ROOT/rel
  if not p.is_file():raise FileNotFoundError(p)
  files[p.relative_to(ROOT).as_posix()]=p
 for f in ['scripts/run_future_harm_unseen_family.py','scripts/evaluate_future_harm_unseen_family.py','scripts/report_future_harm_unseen_family.py','scripts/package_future_harm_unseen_family_addendum.py',
           'src/drift_monitoring/phase3/__init__.py','src/drift_monitoring/phase3/core.py','tests/phase3/test_future_harm_unseen_family.py',
           'docs/phase3/FUTURE_HARM_UNSEEN_FAMILY_PROTOCOL_20261116.md','docs/phase3/FUTURE_HARM_ONSET_RANDOMIZED_FOLLOWUP_20261116.md',
           'docs/phase3/FUTURE_HARM_FORECAST_EXTENSION_20261116.md','docs/phase3/PHASE2_TRACEABILITY.md','docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md',
           'docs/phase3/DECISION_STRENGTHENING_CHANGELOG.md','docs/phase3/23I-2581_23I-2604_Phase3_Reproduction.md','docs/phase3/experiment_manifest.json',
           'docs/23i-2604_23i-2581DAV-phase2.docx','docs/Phase 03 - Implementation.pdf','output/phase3/23I-2581_23I-2604_Phase3.zip',
           'output/phase3/IDM_Viz_Phase3_Final_Strengthening_20261116_v13.receipt.json',
           'output/phase3/IDM_Viz_ELEC2_Calibration_Family_Addendum_20261116_v2.receipt.json',
           'data/external/openml_151/electricity.arff','requirements-phase3.txt','pyproject.toml']:
  if (ROOT/f).is_file():add(f)
 for folder in ['results/phase3/future_harm_forecast_20261116_v1','results/phase3/future_harm_unseen_family_20261116_v1',
               'results/phase3/future_harm_unseen_family_balanced_20261116_v1','reports/phase3/future_harm_unseen_family_balanced_20261116_v1']:
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tmp'):add(p.relative_to(ROOT))
 protected={
  'docs/23i-2604_23i-2581DAV-phase2.docx':'786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4',
  'docs/Phase 03 - Implementation.pdf':'9a94654bdb772b03d5228993f2cc7eaf1b8552f41c462ea84a4649bb0a828115',
  'output/phase3/23I-2581_23I-2604_Phase3.zip':'279d9867d1c75720a51df12caae2ac0b4d490641b83752e32642dcb87e1c8877',
  'output/phase3/IDM_Viz_Phase3_Final_Strengthening_20261116_v13.zip':'4010ca6baf3cd67ed7516e16fcffff31558c97e17cfc92409f08ef25e0d33c2f',
  'output/phase3/IDM_Viz_ELEC2_Calibration_Family_Addendum_20261116_v2.zip':'e9a48f4708fdd932f9e5243a5ed941e1e6b60f0e98bdc47737aea6e47bcfb1fb'}
 assert all(sha(ROOT/k)==v for k,v in protected.items())
 inv={k:dict(bytes=v.stat().st_size,sha256=sha(v)) for k,v in files.items()}
 readme='''# Future-harm held-out synthetic-family addendum

This supplement builds on main Phase 3 archive v13 (preserved by hash). It includes the original aligned-onset C7 pilot and a balanced-onset follow-up, both using 30 disjoint evaluation seeds and an unchanged model trained only on the original C0-C6 development set. The pilot's near-perfect time-only AUC is retained as a confound warning. The balanced follow-up still shows time predictability and is not external or general family validation.

From the extracted root run:
  python scripts/run_future_harm_unseen_family.py --onset-policy fixed9 --out results/fixed_NEW
  python scripts/evaluate_future_harm_unseen_family.py --run results/fixed_NEW
  python scripts/run_future_harm_unseen_family.py --onset-policy balanced_onsets --out results/balanced_NEW
  python scripts/evaluate_future_harm_unseen_family.py --run results/balanced_NEW
  python scripts/report_future_harm_unseen_family.py --run results/balanced_NEW --out reports/balanced_NEW
  python -m pytest tests/phase3/test_future_harm_unseen_family.py -q

All output folders must be new. Label-free predictor and evaluator truth artifacts are separate. The receipt beside this archive records exact clean-extraction matches and protected hashes.
'''
 with zipfile.ZipFile(OUT,'x',zipfile.ZIP_DEFLATED,compresslevel=5) as z:
  for k,v in files.items():z.write(v,k)
  z.writestr('REPRODUCE.md',readme);z.writestr('MEMBER_SHA256.json',json.dumps(inv,indent=2));z.writestr('PRESERVATION.json',json.dumps(protected,indent=2))
 with zipfile.ZipFile(OUT) as z:
  assert z.testzip() is None
  for k,v in inv.items():assert hashlib.sha256(z.read(k)).hexdigest()==v['sha256'],k
  with tempfile.TemporaryDirectory(prefix='future_family_extract_') as td:
   work=Path(td);z.extractall(work);env=os.environ.copy();env['PYTHONPATH']=str(work)+os.pathsep+str(work/'src');env['PYTHONNOUSERSITE']='1';runs=[]
   cmds=[('tests',[str(PY),'-m','pytest','tests/phase3/test_future_harm_unseen_family.py','-q']),
    ('pilot_runner',[str(PY),'scripts/run_future_harm_unseen_family.py','--onset-policy','fixed9','--out','results/fixed_EXTRACTED']),
    ('pilot_evaluator',[str(PY),'scripts/evaluate_future_harm_unseen_family.py','--run','results/fixed_EXTRACTED']),
    ('balanced_runner',[str(PY),'scripts/run_future_harm_unseen_family.py','--onset-policy','balanced_onsets','--out','results/balanced_EXTRACTED']),
    ('balanced_evaluator',[str(PY),'scripts/evaluate_future_harm_unseen_family.py','--run','results/balanced_EXTRACTED']),
    ('report',[str(PY),'scripts/report_future_harm_unseen_family.py','--run','results/balanced_EXTRACTED','--out','reports/balanced_EXTRACTED'])]
   for name,cmd in cmds:
    r=subprocess.run(cmd,cwd=work,env=env,text=True,capture_output=True,timeout=900);runs.append(dict(stage=name,command=cmd,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
    if r.returncode:raise RuntimeError(json.dumps(runs,indent=2))
   checks=[]
   for saved,fresh in [('future_harm_unseen_family_20261116_v1','fixed_EXTRACTED'),('future_harm_unseen_family_balanced_20261116_v1','balanced_EXTRACTED')]:
    for rel in ['predictions/online_features.csv','evaluator_truth/future_targets.csv','predictions/heldout_probabilities.csv','metrics_overall_by_delay.csv','paired_seed_bootstrap.csv','validation.json']:
     a=work/f'results/phase3/{saved}/{rel}';b=work/f'results/{fresh}/{rel}'
     if rel.endswith('.json'):assert json.loads(a.read_text())==json.loads(b.read_text()),str(a)
     else:pd.testing.assert_frame_equal(pd.read_csv(a),pd.read_csv(b),check_exact=True)
     checks.append(str(a.relative_to(work))+' exact')
   for f in ['FUTURE_HARM_UNSEEN_FAMILY.tex','unseen_family_metrics.png']:
    a=work/'reports/phase3/future_harm_unseen_family_balanced_20261116_v1'/f;b=work/'reports/balanced_EXTRACTED'/f
    if f.endswith('.tex'):assert a.read_bytes()==b.read_bytes()
    else:
     from PIL import Image
     assert np.array_equal(np.asarray(Image.open(a).convert('RGBA')),np.asarray(Image.open(b).convert('RGBA')))
    checks.append(str(a.relative_to(work))+' report exact')
 assert all(sha(ROOT/k)==v for k,v in protected.items())
 rec=dict(status='passed',archive=str(OUT),archive_sha256=sha(OUT),bytes=OUT.stat().st_size,members=len(inv)+3,crc='passed',member_hashes='passed',clean_extraction='passed',fresh_commands=runs,exact_comparisons=checks,preserved_hashes=protected,uploaded=False)
 OUT.with_suffix('.receipt.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
