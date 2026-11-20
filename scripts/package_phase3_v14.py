"""Package current Phase 3 evidence and verify extracted strengthening analyses."""
from pathlib import Path
import hashlib,json,os,subprocess,tempfile,zipfile
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PY=Path('C:/Users/hp/anaconda3/python.exe')
OUT=ROOT/'output/phase3/IDM_Viz_Phase3_Final_Strengthening_20261116_v14.zip'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    if OUT.exists():raise FileExistsError(OUT)
    files={}
    def add(path):
        p=ROOT/path
        if not p.is_file():raise FileNotFoundError(p)
        files[p.relative_to(ROOT).as_posix()]=p
    # Full current method code, selected tests and protocol records.
    for folder in ('src/drift_monitoring/phase3','tests/phase3'):
        for p in (ROOT/folder).rglob('*.py'):add(p.relative_to(ROOT))
    for name in ('scripts/strengthen_phase3_decisions.py','scripts/evaluate_phase3_horizon_extension.py',
                 'scripts/run_phase3_dependence_extension.py','scripts/evaluate_phase3_dependence_extension.py',
                 'scripts/report_phase3_dependence_extension.py',
                 'scripts/evaluate_uci224_exact_interval_extension.py','scripts/report_uci224_exact_interval_extension.py',
                 'scripts/run_uci224_split_model_sensitivity.py','scripts/evaluate_uci224_split_model_sensitivity.py',
                 'scripts/report_uci224_split_model_sensitivity.py','scripts/evaluate_phase3_delay100_availability.py',
                 'scripts/evaluate_uci224_day2.py',
                 'scripts/compare_phase3_large_seed_sample.py','scripts/run_phase3_large_seed.py',
                 'scripts/evaluate_phase3_large_seed.py','scripts/report_phase3_large_seed.py',
                 'scripts/report_phase3_strengthening.py','scripts/build_phase3_consolidated_v5.py',
                 'scripts/run_elec2_temporal_validation.py','scripts/evaluate_elec2_temporal_validation.py',
                 'scripts/run_elec2_delayed_recalibration.py','scripts/evaluate_elec2_delayed_recalibration.py',
                 'scripts/report_elec2_temporal_validation.py','scripts/report_elec2_delayed_recalibration.py',
                 'scripts/run_future_harm_forecast.py','scripts/evaluate_future_harm_forecast.py','scripts/report_future_harm_forecast.py',
                 'scripts/render_phase3_v2_pdf.py',
                 'scripts/build_phase3_view.py',
                 'scripts/phase3_view_template.html','scripts/validate_phase3_view.js',
                 'scripts/package_phase3_final_strengthening.py','pyproject.toml','requirements-phase3.txt',
                 'docs/phase3/experiment_manifest.json','docs/phase3/V2_EXTENSION_MANIFEST.md',
                 'docs/phase3/PHASE2_TRACEABILITY.md','docs/phase3/DEPENDENCE_EXTENSION_PROTOCOL_20261116.md',
                 'docs/phase3/UCI224_EXACT_INTERVAL_EXTENSION_20261116.md',
                 'docs/phase3/UCI224_SPLIT_MODEL_SENSITIVITY_20261116.md',
                 'docs/phase3/DELAY100_AVAILABILITY_EXTENSION_20261116.md',
                 'docs/phase3/ELEC2_TEMPORAL_EXTENSION_20261116.md','docs/phase3/ELEC2_DELAYED_RECALIBRATION_20261116.md',
                 'docs/phase3/FUTURE_HARM_FORECAST_EXTENSION_20261116.md','docs/phase3/PHASE3_REMAINING_WEAKNESSES_20261116.md',
                 'docs/phase3/research_improvement_day2/PROTOCOL_LOCK.json',
                 'docs/phase3/LARGE_SEED_ROBUSTNESS_PROTOCOL.md','docs/phase3/ADAPTATION_PROTOCOL.md',
                 'docs/phase3/DECISION_STRENGTHENING_CHANGELOG.md','docs/phase3/OVERNIGHT_EXECUTION_PLAN_20261115.md',
                 'docs/phase3/NEXT_IMPLEMENTATION_PLAN_20261115.md','docs/23i-2604_23i-2581DAV-phase2.docx',
                 'docs/Phase 03 - Implementation.pdf','docs/phase3/AUTONOMOUS_24H_AUDIT_20261116.md',
                 'results/phase3/autonomous_24h_20261116/STATUS.json','results/phase3/autonomous_24h_20261116/STATUS.md',
                 'scripts/package_phase3_v14.py'):
        if (ROOT/name).is_file():add(name)
    # Compact inputs and outputs needed to reproduce new analyses.
    for folder in ('results/phase3/final_v1','results/phase3/development_v1',
                   'results/phase3/development_remaining_v1','results/phase3/decision_strengthening_v2',
                   'results/phase3/horizon_metadata_strengthening_v2','results/phase3/external_uci224_day2_v2',
                   'results/phase3/large_seed_robustness_final',
                   'results/phase3/overnight_continuation_20261115',
                   'results/phase3/dependence_extension_20261116_development_v4',
                   'results/phase3/dependence_extension_20261116_final_v2',
                   'results/phase3/external_uci224_exact_interval_20261116_v1',
                   'results/phase3/delay100_availability_20261116_v1',
                   'results/phase3/uci224_split_model_sensitivity_20261116_v2',
                   'results/phase3/elec2_temporal_20261116_v2','results/phase3/elec2_recalibration_20261116_v1',
                   'results/phase3/future_harm_forecast_20261116_v1',
                   'reports/phase3/dependence_extension_20261116',
                   'reports/phase3/uci224_exact_intervals_20261116',
                   'reports/phase3/uci224_split_model_sensitivity_20261116_v2',
                   'reports/phase3/elec2_temporal_20261116','reports/phase3/elec2_recalibration_20261116',
                   'reports/phase3/future_harm_forecast_20261116_v10',
                   'reports/phase3/decision_strengthening_v4','reports/phase3/decision_strengthening_v5',
                   'reports/phase3/large_seed_robustness_final'):
        for p in (ROOT/folder).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tmp'):add(p.relative_to(ROOT))
    add('data/external/uci224_source/uci224_curl.zip')
    add('data/external/openml_151/electricity.arff')
    add('reports/phase3/overnight_robustness_20261115/sample_reproduction_tolerant/sample_reproduction_comparison.json')
    protected={
      'docs/23i-2604_23i-2581DAV-phase2.docx':'786d4f23d54d0d09edd5f94740198f43ca3b406eb38626ae754fe04fe26c5ef4',
      'docs/Phase 03 - Implementation.pdf':'9a94654bdb772b03d5228993f2cc7eaf1b8552f41c462ea84a4649bb0a828115',
      'output/phase3/23I-2581_23I-2604_Phase3.zip':'279d9867d1c75720a51df12caae2ac0b4d490641b83752e32642dcb87e1c8877'}
    assert all(sha(ROOT/k)==v for k,v in protected.items())
    inventory={n:dict(bytes=p.stat().st_size,sha256=sha(p)) for n,p in files.items()}
    instructions='''# Phase 3 supplementary strengthening package (prepared, not uploaded)

Main manuscript: reports/phase3/decision_strengthening_v5/REPORT.tex and REPORT.pdf.
Completed large-seed robustness results: results/phase3/large_seed_robustness_final.
The run is 1,500 seeds per synthetic task, not external validation.

Reproduce the complete locked large-seed run only with fresh output locations:
  python scripts/run_phase3_large_seed.py --mode final --out results/seed_run_NEW
  python scripts/evaluate_phase3_large_seed.py --run results/seed_run_NEW
This full run took about 6.2 hours. For a quicker exact sample, use the included
sample comparison script with a fresh directory. Scientific result CSVs,
predictions/timings exclusions and tolerances are recorded in receipts.

Reproduce audit policies/horizon tables from root with fresh outputs:
  python scripts/strengthen_phase3_decisions.py --out results/audit_NEW
  python scripts/evaluate_phase3_horizon_extension.py --out results/horizon_NEW
  python -m pytest tests/phase3 -q
Reproduce the targeted exploratory dependence detector with the locked threshold protocol:
  python scripts/run_phase3_dependence_extension.py --mode development --out results/dev_NEW
  python scripts/run_phase3_dependence_extension.py --mode final --threshold-file results/dev_NEW/threshold.json --out results/final_NEW
  python scripts/evaluate_phase3_dependence_extension.py --run results/final_NEW
The included held-out extension detected C6 on 30/30 seeds versus 16/30 for the
KS/PCA comparator, while stationary alarms averaged 1.70 vs 0.83 per 24-batch
stream. This targeted synthetic result is a sensitivity/burden tradeoff, not a
general detector or harm-prediction claim.
Reproduce the external one-dataset robustness and interval checks:
  python scripts/run_uci224_split_model_sensitivity.py --out results/uci224_NEW
  python scripts/evaluate_uci224_split_model_sensitivity.py --run results/uci224_NEW
  python scripts/evaluate_uci224_exact_interval_extension.py --out results/exact_NEW
  python scripts/evaluate_phase3_delay100_availability.py --out results/delay100_NEW
The six split seeds are sensitivities on one dataset, not six independent domains.
The audit extension is outcome-informed, not a new confirmatory trial. It does
not establish human decision benefit. Phase 2 DOCX and submitted ZIP hashes are
preserved in PRESERVATION.json. New package is prepared, not uploaded.

Reproduce the ELEC2 second-stream checks and the seed-held-out forecast:
  python scripts/run_elec2_temporal_validation.py --out results/elec2_NEW
  python scripts/evaluate_elec2_temporal_validation.py --run results/elec2_NEW
  python scripts/run_elec2_delayed_recalibration.py --run results/elec2_NEW --out results/recal_NEW
  python scripts/evaluate_elec2_delayed_recalibration.py --run results/elec2_NEW --recalibration results/recal_NEW
  python scripts/run_future_harm_forecast.py --out results/forecast_NEW
  python scripts/evaluate_future_harm_forecast.py results/forecast_NEW
  python scripts/report_future_harm_forecast.py --run results/forecast_NEW --out reports/forecast_report_NEW
The forecast rerun takes about seven minutes on the tested laptop. It uses
30 development and 30 disjoint evaluation seeds. Its strong result is limited
to the same seven known synthetic laws, not unseen or natural streams.
'''
    with zipfile.ZipFile(OUT,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for n,p in files.items():z.write(p,n)
        z.writestr('REPRODUCE.md',instructions)
        z.writestr('MEMBER_SHA256.json',json.dumps(inventory,indent=2))
        z.writestr('PRESERVATION.json',json.dumps(protected,indent=2))
    with zipfile.ZipFile(OUT) as z:
        assert z.testzip() is None
        for n,r in inventory.items():assert hashlib.sha256(z.read(n)).hexdigest()==r['sha256'],n
        with tempfile.TemporaryDirectory(prefix='phase3_final_extract_') as tmp:
            z.extractall(tmp);work=Path(tmp);env=os.environ.copy()
            env['PYTHONPATH']=str(work)+os.pathsep+str(work/'src');env['PYTHONNOUSERSITE']='1'
            runs=[]
            for name,cmd in [
                ('tests',[str(PY),'-m','pytest','tests/phase3/test_decision_audit_extension.py','tests/phase3/test_large_seed_robustness.py','tests/phase3/test_dependence_detector.py','tests/phase3/test_uci224_exact_interval_extension.py','tests/phase3/test_uci224_split_model_sensitivity.py','tests/phase3/test_elec2_temporal_validation.py','tests/phase3/test_elec2_delayed_recalibration.py','tests/phase3/test_future_harm_forecast.py','-q']),
                ('audit',[str(PY),'scripts/strengthen_phase3_decisions.py','--out','results/audit_EXTRACTED']),
                ('horizon',[str(PY),'scripts/evaluate_phase3_horizon_extension.py','--out','results/horizon_EXTRACTED']),
                ('dependence',[str(PY),'scripts/evaluate_phase3_dependence_extension.py','--run','results/phase3/dependence_extension_20261116_final_v2']),
                ('exact_interval',[str(PY),'scripts/evaluate_uci224_exact_interval_extension.py','--out','results/exact_interval_EXTRACTED']),
                ('delay100',[str(PY),'scripts/evaluate_phase3_delay100_availability.py','--out','results/delay100_EXTRACTED']),
                ('split_model_runner',[str(PY),'scripts/run_uci224_split_model_sensitivity.py','--out','results/uci224_sensitivity_EXTRACTED']),
                ('split_model_evaluator',[str(PY),'scripts/evaluate_uci224_split_model_sensitivity.py','--run','results/uci224_sensitivity_EXTRACTED'])]:
                r=subprocess.run(cmd,cwd=work,env=env,text=True,capture_output=True,timeout=900)
                runs.append(dict(stage=name,command=cmd,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
                if r.returncode:raise RuntimeError(json.dumps(runs,indent=2))
            forecast_out='results/future_forecast_EXTRACTED'
            r=subprocess.run([str(PY),'scripts/run_future_harm_forecast.py','--out',forecast_out],cwd=work,env=env,text=True,capture_output=True,timeout=3600)
            runs.append(dict(stage='future_forecast_runner',command='python scripts/run_future_harm_forecast.py --out '+forecast_out,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
            if r.returncode:raise RuntimeError(json.dumps(runs,indent=2))
            r=subprocess.run([str(PY),'scripts/evaluate_future_harm_forecast.py',forecast_out],cwd=work,env=env,text=True,capture_output=True,timeout=300)
            runs.append(dict(stage='future_forecast_evaluator',command='python scripts/evaluate_future_harm_forecast.py '+forecast_out,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
            if r.returncode:raise RuntimeError(json.dumps(runs,indent=2))
            report_out='reports/future_forecast_report_EXTRACTED'
            r=subprocess.run([str(PY),'scripts/report_future_harm_forecast.py','--run',forecast_out,'--out',report_out],cwd=work,env=env,text=True,capture_output=True,timeout=300)
            runs.append(dict(stage='future_forecast_report',command='python scripts/report_future_harm_forecast.py --run '+forecast_out+' --out '+report_out,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
            if r.returncode:raise RuntimeError(json.dumps(runs,indent=2))
            checks=[]
            for old,new in [('results/phase3/decision_strengthening_v2','results/audit_EXTRACTED'),
                            ('results/phase3/horizon_metadata_strengthening_v2','results/horizon_EXTRACTED')]:
                for p in (work/old).glob('*.csv'):
                    pd.testing.assert_frame_equal(pd.read_csv(p),pd.read_csv(work/new/p.name),check_exact=True)
                    checks.append(str(p.relative_to(work)))
            for p in (work/'results/phase3/dependence_extension_20261116_final_v2').glob('validated_*.csv'):
                checks.append(str(p.relative_to(work))+' (clean-extraction evaluator passed)')
            comparisons=[
                ('results/phase3/external_uci224_exact_interval_20261116_v1/cohort_interval_comparison.csv','results/exact_interval_EXTRACTED/cohort_interval_comparison.csv'),
                ('results/phase3/delay100_availability_20261116_v1/release_schedule.csv','results/delay100_EXTRACTED/release_schedule.csv'),
                ('results/phase3/uci224_split_model_sensitivity_20261116_v2/evaluation/per_cohort_metrics.csv','results/uci224_sensitivity_EXTRACTED/evaluation/per_cohort_metrics.csv'),
                ('results/phase3/uci224_split_model_sensitivity_20261116_v2/evaluation/per_class_metrics.csv','results/uci224_sensitivity_EXTRACTED/evaluation/per_class_metrics.csv'),
                ('results/phase3/uci224_split_model_sensitivity_20261116_v2/evaluation/common_cohort_split_sensitivity.csv','results/uci224_sensitivity_EXTRACTED/evaluation/common_cohort_split_sensitivity.csv')]
            for saved,fresh in comparisons:
                pd.testing.assert_frame_equal(pd.read_csv(work/saved),pd.read_csv(work/fresh),check_exact=True)
                checks.append(saved+' == clean-extraction rerun')
            for saved,fresh in [
              ('results/phase3/future_harm_forecast_20261116_v1/evaluator_truth/future_targets.csv',forecast_out+'/evaluator_truth/future_targets.csv'),
              ('results/phase3/future_harm_forecast_20261116_v1/predictions/heldout_probabilities.csv',forecast_out+'/predictions/heldout_probabilities.csv'),
              ('results/phase3/future_harm_forecast_20261116_v1/predictions/online_features.csv',forecast_out+'/predictions/online_features.csv')]:
                pd.testing.assert_frame_equal(pd.read_csv(work/saved),pd.read_csv(work/fresh),check_exact=True)
                checks.append(saved+' == clean-extraction forecast rerun')
            for name in ['FUTURE_HARM_FORECAST.tex','forecast_metrics.png']:
                saved='reports/phase3/future_harm_forecast_20261116_v10/'+name
                fresh=report_out+'/'+name
                if name.endswith('.tex'):
                    assert (work/saved).read_bytes()==(work/fresh).read_bytes()
                else:
                    from PIL import Image
                    import numpy as np
                    assert np.array_equal(np.asarray(Image.open(work/saved).convert('RGBA')),
                                          np.asarray(Image.open(work/fresh).convert('RGBA')))
                checks.append(saved+' == clean-extraction report regeneration')
    assert all(sha(ROOT/k)==v for k,v in protected.items())
    receipt=dict(status='passed',archive=str(OUT),archive_sha256=sha(OUT),bytes=OUT.stat().st_size,
        members=len(inventory)+3,crc='passed',member_hashes='passed',clean_extraction='passed',
        fresh_commands=runs,exact_result_csv_comparisons=checks,preserved_hashes=protected,uploaded=False)
    OUT.with_suffix('.receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
