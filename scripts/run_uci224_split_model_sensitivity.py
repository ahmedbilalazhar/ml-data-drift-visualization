"""Freeze six-history-split x two-model x two-window UCI sensitivity run."""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from evaluate_uci224_day2 import load_cohorts

SEEDS=tuple(range(420001,420007))
WINDOWS={'batch1':(1,),'batch1_2':(1,2)}
MODELS=('logistic','tree_depth6')

def partitions(y, seed):
    i=np.arange(len(y))
    tr,hold=next(StratifiedShuffleSplit(n_splits=1,test_size=.4,random_state=seed).split(i,y))
    ca0,rf0=next(StratifiedShuffleSplit(n_splits=1,test_size=.5,random_state=seed+1_000_000).split(hold,y[hold]))
    ca,rf=hold[ca0],hold[rf0]
    if len(set(tr)&set(ca)) or len(set(tr)&set(rf)) or len(set(ca)&set(rf)):
        raise AssertionError('split overlap')
    if set(np.r_[tr,ca,rf])!=set(i): raise AssertionError('split does not cover history')
    return tr,ca,rf

def model_for(name):
    if name=='logistic': return LogisticRegression(C=1.0,max_iter=2000,random_state=2242026)
    if name=='tree_depth6': return DecisionTreeClassifier(max_depth=6,min_samples_leaf=25,random_state=2242026)
    raise ValueError(name)

def main(out:Path):
    if out.exists(): raise FileExistsError(f'choose a fresh output directory: {out}')
    out.mkdir(parents=True)
    lockfile=ROOT/'docs/phase3/research_improvement_day2/PROTOCOL_LOCK.json'
    base=json.loads(lockfile.read_text(encoding='utf-8'))
    archive=ROOT/base['dataset']['archive']; source_hash=hashlib.sha256(archive.read_bytes()).hexdigest()
    if source_hash!=base['dataset']['archive_sha256']: raise RuntimeError('source hash differs from protocol')
    cohorts=load_cohorts(); start=time.perf_counter(); online=[]; fitted=[]
    split_manifest={}
    for window,batches in WINDOWS.items():
      hx=np.concatenate([cohorts[b][0] for b in batches]); hy=np.concatenate([cohorts[b][1] for b in batches])
      for seed in SEEDS:
       tr,ca,rf=partitions(hy,seed)
       split_manifest[f'{window}:{seed}']={'history_batches':batches,'train_indices':tr.tolist(),
            'calibration_indices':ca.tolist(),'reference_indices':rf.tolist()}
       for model_name in MODELS:
        tic=time.perf_counter(); scaler=StandardScaler().fit(hx[tr]) if model_name=='logistic' else None
        tx=scaler.transform(hx[tr]) if scaler is not None else hx[tr]
        model=model_for(model_name).fit(tx,hy[tr]);
        cx=scaler.transform(hx[ca]) if scaler is not None else hx[ca]
        cp=model.predict_proba(cx); cpred=model.classes_[cp.argmax(axis=1)]
        cc=(cpred==hy[ca]).astype(int); cconf=cp.max(axis=1)
        if len(np.unique(cc))<2: raise RuntimeError(f'one-class correctness calibration: {window}/{seed}/{model_name}')
        calibrator=LogisticRegression(C=1.0,max_iter=2000,random_state=2242026).fit(cconf.reshape(-1,1),cc)
        rx=scaler.transform(hx[rf]) if scaler is not None else hx[rf]
        rp=model.predict_proba(rx); rpred=model.classes_[rp.argmax(axis=1)]
        ref_accuracy=float(np.mean(rpred==hy[rf]))
        for batch in range(min(batches)+len(batches),11):
          if batch not in cohorts: continue
          # For pooled history, monitor begins at batch 3; batch 2 is history.
          if window=='batch1_2' and batch<3: continue
          x,y=cohorts[batch]; sx=scaler.transform(x) if scaler is not None else x
          proba=model.predict_proba(sx); pred=model.classes_[proba.argmax(axis=1)]; conf=proba.max(axis=1)
          q=calibrator.predict_proba(conf.reshape(-1,1))[:,1]
          for j in range(len(y)):
            online.append((window,seed,model_name,batch,j,int(pred[j]),float(conf[j]),float(q[j])))
        fitted.append({'history_window':window,'split_seed':seed,'model':model_name,
          'train_n':len(tr),'calibration_n':len(ca),'reference_n':len(rf),
          'reference_accuracy':ref_accuracy,'fit_seconds':time.perf_counter()-tic,
          'classifier_classes':model.classes_.tolist()})
    frame=pd.DataFrame(online,columns=['history_window','split_seed','model','batch','row_index','prediction','confidence','calibrated_q'])
    if frame.duplicated(['history_window','split_seed','model','batch','row_index']).any(): raise AssertionError('duplicate scored row')
    if frame.columns.intersection({'correct','accuracy','harm','label','y'}).size: raise AssertionError('online cache leaks truth')
    frame.to_csv(out/'online_predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(fitted).to_csv(out/'fit_records.csv',index=False)
    lock={'status':'online_complete_evaluator_pending','protocol':'UCI224_SPLIT_MODEL_SENSITIVITY_20261116',
      'source_sha256':source_hash,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'seeds':SEEDS,'windows':{k:list(v) for k,v in WINDOWS.items()},'models':MODELS,
      'per_window_model_split':6,'online_rows':len(frame),'online_truth_columns_absent':True,
      'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'sklearn':sklearn.__version__},
      'elapsed_seconds':time.perf_counter()-start,'split_indices':split_manifest}
    (out/'protocol_lock.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in lock.items() if k!='split_indices'},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
