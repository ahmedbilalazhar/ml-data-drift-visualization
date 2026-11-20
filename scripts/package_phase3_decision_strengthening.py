"""Build and independently rerun the decision/horizon supplementary package."""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,zipfile
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PY='C:/Users/hp/anaconda3/python.exe'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    target=ROOT/'output/phase3/IDM_Viz_Decision_Strengthening_20261116.zip'
    if target.exists():raise FileExistsError(target)
    paths={}
    for base in ('scripts','src','tests/phase3','results/phase3/decision_strengthening_v2',
                 'results/phase3/horizon_metadata_strengthening_v2','reports/phase3/decision_strengthening_v3'):
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and (base not in ('src','scripts') or p.suffix in ('.py','.ps1','.html')):
                paths[p.relative_to(ROOT).as_posix()]=p
    for run,files in {
        'final_v1':['online_batches.csv','released_verification.csv','evaluator_batches.csv','localization_batches.csv'],
        'development_v1':['online_batches.csv'], 'development_remaining_v1':['online_batches.csv'],
        'external_uci224_day2_v2':['protocol_lock.json'],
    }.items():
        for name in files:
            p=ROOT/'results/phase3'/run/name;paths[p.relative_to(ROOT).as_posix()]=p
    for name in ('pyproject.toml','requirements-phase3.txt','data/external/uci224_source/uci224_curl.zip',
                 'docs/phase3/research_improvement_day2/PROTOCOL_LOCK.json','docs/phase3/DECISION_STRENGTHENING_CHANGELOG.md',
                 'docs/23i-2604_23i-2581DAV-phase2.docx'):
        p=ROOT/name;paths[name]=p
    protected={
        'docs/23i-2604_23i-2581DAV-phase2.docx':'786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4',
        'output/phase3/23I-2581_23I-2604_Phase3.zip':'279d9867d1c75720a51df12caae2ac0b4d490641b83752e32642dcb87e1c8877'}
    assert all(sha(ROOT/name)==expected for name,expected in protected.items())
    inventory={name:dict(sha256=sha(p),bytes=p.stat().st_size) for name,p in paths.items()}
    readme='''# Decision and horizon evaluation supplement — prepared, not uploaded

This package adds outcome-informed label-audit policies, localization reporting,
persistence forecasting, corrected external class metadata and a consolidated
LaTeX/PDF report. It does not claim human benefit or general forecast validity.
The independently running 1,500-seed study is not included as completed evidence.

From this extracted root, using the recorded scientific environment:
  python -m pytest tests/phase3 -q
  python scripts/strengthen_phase3_decisions.py --out results/audit_NEW
  python scripts/evaluate_phase3_horizon_extension.py --out results/horizon_NEW

All outputs must be fresh. Compare results/audit_NEW CSVs to
results/phase3/decision_strengthening_v2 and results/horizon_NEW CSVs to
results/phase3/horizon_metadata_strengthening_v2. No classifier refits are needed.
The report source is reports/phase3/decision_strengthening_v3/REPORT.tex;
compile with Tectonic from the root, --outdir reports/phase3/decision_strengthening_v3.
Report regeneration reads earlier report prose not bundled here; compile the
included self-contained TeX and supplied figures directly instead.
'''
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name,p in paths.items():z.write(p,name)
        z.writestr('REPRODUCE_STRENGTHENING.md',readme)
        z.writestr('STRENGTHENING_CHECKSUMS.json',json.dumps(inventory,indent=2))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name,r in inventory.items():assert hashlib.sha256(z.read(name)).hexdigest()==r['sha256']
        with tempfile.TemporaryDirectory(prefix='idm_decision_clean_') as temp:
            z.extractall(temp);work=Path(temp);logs=[]
            env=os.environ.copy();env['PYTHONPATH']=str(work)+os.pathsep+str(work/'src');env['PYTHONNOUSERSITE']='1'
            for name,cmd in [('tests',[PY,'-m','pytest','tests/phase3','-q']),
                ('audit',[PY,'scripts/strengthen_phase3_decisions.py','--out','results/audit_NEW']),
                ('horizon',[PY,'scripts/evaluate_phase3_horizon_extension.py','--out','results/horizon_NEW'])]:
                r=subprocess.run(cmd,cwd=work,env=env,capture_output=True,text=True,timeout=300)
                logs.append(dict(stage=name,command=cmd,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
                if r.returncode:raise RuntimeError(json.dumps(logs,indent=2))
            comparisons=[]
            for old,new in [('results/phase3/decision_strengthening_v2','results/audit_NEW'),
                            ('results/phase3/horizon_metadata_strengthening_v2','results/horizon_NEW')]:
                for p in (work/old).glob('*.csv'):
                    pd.testing.assert_frame_equal(pd.read_csv(p),pd.read_csv(work/new/p.name),check_exact=True)
                    comparisons.append(str(p.relative_to(work)))
    assert all(sha(ROOT/name)==expected for name,expected in protected.items())
    receipt=dict(status='passed',archive=str(target),sha256=sha(target),crc='passed',members_hashed=len(inventory),
        extracted_commands=logs,exact_csv_comparisons=comparisons,preserved_hashes=protected,uploaded=False)
    target.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
