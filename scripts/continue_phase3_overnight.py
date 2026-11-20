"""Detached, fail-closed continuation of the EXISTING locked robustness run.

Does not launch, alter, stop or replace the numerical runner. See status.json.
"""
from __future__ import annotations
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PY = Path('C:/Users/hp/anaconda3/python.exe')
ART = Path('C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe')
RUN = ROOT/'results/phase3/large_seed_robustness_final'
STATE = ROOT/'results/phase3/overnight_continuation_20261115'
REPORT = ROOT/'reports/phase3/overnight_robustness_20261115'
DEADLINE = datetime.fromisoformat('2026-11-16T06:29:00+05:00').timestamp()
PRESERVE = {
 'docs/23i-2604_23i-2581DAV-phase2.docx':'786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4',
 'docs/Phase 03 - Implementation.pdf':'9a94654bdb772b03d5228993f2cc7eaf1b8552f41c462ea84a4649bb0a828115',
 'output/phase3/23I-2581_23I-2604_Phase3.zip':'279d9867d1c75720a51df12caae2ac0b4d490641b83752e32642dcb87e1c8877',
 'output/phase3/IDM_Viz_Phase3_v1_submission.zip':'ddf91fa89ab9a9d45c20059c1dbea5cfdd966f3eea62a605769a371ae55e0ebd',
 'output/phase3/IDM_Viz_Phase3_v1_full_reproducibility.zip':'0baba912c5f3e48a8a08a606e2a77a303f67e884ddccfef0907046d94dca5519',
}

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def write_json(p, data):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2),encoding='utf-8'); os.replace(tmp,p)

def status(stage, **extra):
    write_json(STATE/'status.json',dict(stage=stage,pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**extra))
    print(datetime.now().astimezone().isoformat(),stage,extra,flush=True)

def preservation():
    rows={name:dict(expected=expected,actual=sha(ROOT/name)) for name,expected in PRESERVE.items()}
    write_json(STATE/'preservation.json',rows)
    if any(r['expected']!=r['actual'] for r in rows.values()):
        raise RuntimeError('Preservation hash mismatch')

def execute(name,command,cwd=ROOT,timeout=1800):
    remaining=DEADLINE-time.time()
    if remaining<60: raise TimeoutError('Window exhausted before '+name)
    status(name,command=[str(x) for x in command])
    started=time.time()
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[key]='1'
    with (STATE/(name+'.log')).open('w',encoding='utf-8') as log:
        p=subprocess.run([str(x) for x in command],cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT,
                         timeout=min(timeout,max(1,remaining-30)))
    write_json(STATE/(name+'.receipt.json'),dict(command=[str(x) for x in command],exit_code=p.returncode,
               elapsed_seconds=time.time()-started,cwd=str(cwd)))
    if p.returncode: raise RuntimeError(f'{name} failed, see log; exit {p.returncode}')

def tex_escape(text):
    replacements={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
    return ''.join(replacements.get(c,c) for c in text)

def latex_source(md):
    # Preserve numerical report text and tables; do not invent a narrative or citations.
    out=[r'\documentclass[11pt,a4paper]{article}',r'\usepackage[margin=24mm]{geometry}',
         r'\usepackage[T1]{fontenc}',r'\usepackage{lmodern,microtype,booktabs,longtable,graphicx,xcolor,hyperref}',
         r'\definecolor{teal}{HTML}{087E8B}',r'\hypersetup{colorlinks=true,linkcolor=teal,urlcolor=teal}',
         r'\setlength{\emergencystretch}{4em}',r'\title{IDM-Viz: Large-Seed Robustness Evaluation}',
         r'\author{Ahmed Bilal (23I-2581) \and Umar Behram (23I-2604)}',
         r'\date{Supplementary research edition, 16 November 2026}',r'\begin{document}',r'\maketitle',
         r'\noindent Visual inspection of this automatically compiled edition remains pending. Scientific validation and sample reproduction are recorded separately.\par']
    lines=md.read_text(encoding='utf-8').splitlines(); i=0
    while i<len(lines):
        line=lines[i].strip()
        if line.startswith('```'):
            out.append(r'\begin{verbatim}'); i+=1
            while i<len(lines) and not lines[i].strip().startswith('```'):
                # Avoid unbounded paths extending past the page.
                out.extend(lines[i][j:j+85] for j in range(0,max(1,len(lines[i])),85)); i+=1
            out.append(r'\end{verbatim}')
        elif line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+',c or '-') for c in cells): rows.append(cells)
                i+=1
            n=max(len(r) for r in rows); width=.91/n
            out.extend([r'\begingroup\scriptsize',r'\begin{longtable}{'+''.join('p{'+str(width)+r'\linewidth}' for _ in range(n))+'}',r'\toprule'])
            for k,row in enumerate(rows):
                out.append(' & '.join(tex_escape(c) for c in row)+r' \\')
                if k==0: out.append(r'\midrule')
            out.extend([r'\bottomrule',r'\end{longtable}',r'\endgroup']); continue
        elif line.startswith('!['):
            match=re.match(r'!\[(.*?)\]\((.*?)\)',line)
            if match:
                caption,path=match.groups()
                out.extend([r'\begin{figure}[htbp]\centering',r'\includegraphics[width=.97\linewidth]{'+path+'}',r'\caption{'+tex_escape(caption)+'}',r'\end{figure}'])
        elif line.startswith('<!--'): pass
        elif line.startswith('# '): pass
        elif line.startswith('## '):out.append(r'\section{'+tex_escape(line[3:])+'}')
        elif line.startswith('### '):out.append(r'\subsection{'+tex_escape(line[4:])+'}')
        elif line:out.append(tex_escape(line)+r'\par')
        else:out.append('')
        i+=1
    out.append(r'\end{document}')
    target=md.with_suffix('.tex'); target.write_text('\n'.join(out),encoding='utf-8')
    return target

def package():
    target=ROOT/'output/phase3/IDM_Viz_Robustness_Overnight_20261116.zip'
    if target.exists():raise FileExistsError(target)
    paths={}
    for base in (ROOT/'src/drift_monitoring/phase3', ROOT/'scripts', ROOT/'tests/phase3'):
        for p in base.rglob('*'):
            if p.is_file() and p.suffix in ('.py','.ps1','.html'):paths[p.relative_to(ROOT).as_posix()]=p
    for p in (ROOT/'docs/phase3').glob('*'):
        if p.is_file() and p.suffix in ('.md','.json'): paths[p.relative_to(ROOT).as_posix()]=p
    for p in (ROOT/'pyproject.toml',ROOT/'requirements-phase3.txt',ROOT/'README.md'):
        paths[p.relative_to(ROOT).as_posix()]=p
    for base in (RUN,REPORT,STATE):
        for p in base.rglob('*'):
            if p.is_file() and p.name not in ('overnight.lock','stdout.log','stderr.log') and not p.name.endswith('.tmp'):
                paths[p.relative_to(ROOT).as_posix()]=p
    readme='''# Overnight robustness evidence — prepared, not uploaded

This package preserves the completed 1,500-seed-per-task synthetic study,
its analysis, five-seed-per-task exact sample reproduction, code, figures,
LaTeX/Markdown sources and validation logs. It does not claim a new large
adaptation replication. Visual PDF inspection remains pending.

From this extracted root, with the recorded scientific dependencies:
  python -m pytest tests/phase3 -q
  python scripts/evaluate_phase3_large_seed.py --run results/phase3/large_seed_robustness_final
For an independent sample regeneration, choose a fresh output directory:
  python scripts/compare_phase3_large_seed_sample.py --run results/phase3/large_seed_robustness_final --out results/sample_NEW
For a FULL fresh rerun, use scripts/reproduce_phase3_large_seed.ps1 with new
RunDir and ReportDir paths and configure both interpreter paths. The full
rerun takes hours. Sample reproduction is not full-run reproduction.

The submitted Phase 2 basis and submitted archives were preserved and
their hashes are in the continuation preservation receipt. This is a
supplement, not a replacement for the main consolidated project report.
'''
    inventory={name:dict(sha256=sha(p),bytes=p.stat().st_size) for name,p in paths.items()}
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for name,p in paths.items():z.write(p,name)
        z.writestr('OVERNIGHT_README.md',readme)
        inventory['OVERNIGHT_README.md']=dict(sha256=hashlib.sha256(readme.encode()).hexdigest(),bytes=len(readme.encode()))
        z.writestr('OVERNIGHT_CHECKSUMS.json',json.dumps(inventory,indent=2))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        for name,entry in inventory.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==entry['sha256'],name
        with tempfile.TemporaryDirectory(prefix='idm_overnight_clean_') as tmp:
            z.extractall(tmp)
            execute('clean_package_tests',[PY,'-m','pytest','tests/phase3','-q'],cwd=Path(tmp),timeout=600)
    write_json(target.with_suffix('.receipt.json'),dict(path=str(target),sha256=sha(target),members=len(inventory)+1,
              crc='passed',member_hashes='passed',clean_extracted_phase3_tests='passed',uploaded=False,
              visual_review='pending; no human or agent visual inspection claimed'))

def main():
    STATE.mkdir(parents=True,exist_ok=True)
    # Exclusive launch guard: do not run two postprocessing chains.
    with (STATE/'overnight.lock').open('x') as f:f.write(str(os.getpid()))
    if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        preservation()
        status('waiting_for_existing_run',deadline='2026-11-16 06:29 Asia/Karachi',launches_new_experiment=False)
        last_progress=0
        while True:
            lock=json.loads((RUN/'protocol_lock.json').read_text(encoding='utf-8'))
            if lock['status']=='complete':break
            if lock['status']=='failed':raise RuntimeError('Existing numerical run marked failed')
            if time.time()>DEADLINE-2400:
                raise TimeoutError('Existing run not complete by analysis reserve cutoff; partial evidence preserved, no success claimed')
            if time.time()-last_progress>120:
                # Read only the tail to avoid repeatedly parsing hundreds of MB.
                p=RUN/'per_seed_condition.csv'
                with p.open('rb') as f:
                    f.seek(max(0,p.stat().st_size-6000)); tail=f.read().decode('utf-8',errors='replace').splitlines()
                status('waiting_for_existing_run',last_summary_row=tail[-1] if tail else None,summary_bytes=p.stat().st_size)
                last_progress=time.time()
            time.sleep(30)
        execute('evaluate',[PY,ROOT/'scripts/evaluate_phase3_large_seed.py','--run',RUN],timeout=3600)
        execute('sample_reproduction',[PY,ROOT/'scripts/compare_phase3_large_seed_sample.py','--run',RUN,'--out',REPORT/'sample_reproduction'],timeout=1200)
        execute('tests',[PY,'-m','pytest','tests','-q'],timeout=600)
        execute('report',[PY,ROOT/'scripts/report_phase3_large_seed.py','--run',RUN,'--out',REPORT,
                          '--reproduction',REPORT/'sample_reproduction/sample_reproduction_comparison.json'],timeout=900)
        md=REPORT/'23I-2581_23I-2604_Phase3_Robustness_Report.md'
        marker=Path('C:/Users/hp/.codex/plugins/cache/openai-primary-runtime/pdf/26.1112.11217/skills/pdf/container_tools/mark_artifact_operation_started.mjs')
        execute('artifact_start',['node',marker,'--operation-kind','create','--expected-output-count','1','--output-format','pdf'],timeout=60)
        execute('render',[ART,ROOT/'scripts/render_phase3_v2_pdf.py',md],timeout=600)
        tex=latex_source(md)
        compiler=ROOT/'tmp/tectonic/tectonic.exe'
        # Preserve the successfully rendered PDF if TeX compilation fails.
        shutil.copy2(md.with_suffix('.pdf'),REPORT/'Robustness_Report_ReportLab.pdf')
        try:
            execute('latex_compile',[compiler,tex,'--outdir',REPORT,'--keep-logs'],timeout=600)
            render_code = (
                "import pathlib,json,hashlib,pypdfium2 as p; "
                "f=pathlib.Path(__import__('sys').argv[1]); d=p.PdfDocument(str(f)); "
                "q=f.parent/'qa_latex'; q.mkdir(exist_ok=True); "
                "[(d[i].render(scale=1.5).to_pil().save(q/('page-%02d.png'%(i+1)))) for i in range(len(d))]; "
                "(f.parent/'latex_render_record.json').write_text(json.dumps(dict(pages=len(d),"
                "pdf_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),visual_inspection='pending'),indent=2))"
            )
            execute('latex_render',[ART,'-c',render_code,tex.with_suffix('.pdf')],timeout=600)
        except Exception as exc:
            write_json(STATE/'latex_failure.json',dict(error=repr(exc),editable_tex=str(tex)))
        write_json(REPORT/'visual_review.json',dict(status='pending',reason='Detached automation cannot perform agent visual inspection; review rendered pages before submission'))
        preservation()
        package()
        status('completed_automated_checks',uploaded=False,visual_review='pending',main_consolidated_report='separate follow-up required')
    except BaseException as exc:
        status('blocked_or_failed',error=repr(exc),traceback=traceback.format_exc(),partial_evidence_preserved=True)
        raise
    finally:
        if os.name=='nt':ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true');a=p.parse_args()
    if a.self_check:
        assert PY.is_file() and ART.is_file() and RUN.is_dir()
        assert tex_escape('a_b%')==r'a\_b\%'
        print('Self-check passed: paths and TeX escaping; no experiment launched')
    else:main()
