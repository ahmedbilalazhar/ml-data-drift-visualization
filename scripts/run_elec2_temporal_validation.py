"""Feature-only, fixed-model IDM-Viz run on the temporally ordered ELEC2 stream."""
from __future__ import annotations
import argparse,hashlib,json,platform,sys,time
from pathlib import Path
import numpy as np,pandas as pd,scipy,sklearn
from scipy.stats import ks_2samp
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from drift_monitoring.phase3.core import holm,impact
SOURCE=ROOT/'data/external/openml_151/electricity.arff'

def load_ele2(path=SOURCE):
 raw=Path(path).read_bytes(); digest=hashlib.sha256(raw).hexdigest()
 if digest!='2d86fbc74c69a5c04bcde0119bce280dd6d8efa0115987edc75b77ccf024d7f2': raise ValueError('ELEC2 source checksum mismatch')
 lines=raw.decode('ascii').splitlines(); start=next((i+1 for i,s in enumerate(lines) if s.strip().lower()=='@data'),None)
 if start is None: raise ValueError('ARFF has no @data section')
 rows=[s.strip() for s in lines[start:] if s.strip() and not s.lstrip().startswith('%')]
 if len(rows)!=45312: raise ValueError(f'expected 45,312 source rows, got {len(rows)}')
 vals=np.empty((len(rows),8),dtype=np.float64); labels=np.empty(len(rows),dtype=np.int8)
 for i,line in enumerate(rows):
  cols=line.split(',')
  if len(cols)!=9: raise ValueError(f'malformed row {i}')
  vals[i]=[float(x) for x in cols[:8]]
  if cols[8]=='UP': labels[i]=1
  elif cols[8]=='DOWN': labels[i]=0
  else: raise ValueError(f'unknown target {cols[8]}')
 if not np.isfinite(vals).all(): raise ValueError('non-finite data')
 # Restore coarse order; the source is not date-monotone and 11 dates have weekday ties.
 group_first={}
 for i,(date_value,weekday) in enumerate(zip(vals[:,0],vals[:,1])):
  group_first.setdefault((date_value,weekday),i)
 tie_order=np.asarray([group_first[(d,w)] for d,w in zip(vals[:,0],vals[:,1])])
 order=np.lexsort((np.arange(len(rows)),vals[:,2],tie_order,vals[:,0]))
 ordered=vals[order]; target=labels[order]
 if np.any(np.diff(ordered[:,0])<0): raise AssertionError('date order failed')
 same_group=(ordered[1:,0]==ordered[:-1,0]) & (ordered[1:,1]==ordered[:-1,1])
 if np.any(np.diff(ordered[:,2])[same_group]<0): raise AssertionError('within-date/weekday period order failed')
 # Exclude normalized date (time index) and contemporaneous NSW price (target-proximal).
 x=ordered[:,[1,2,4,5,6,7]]
 return x,target,ordered[:,0],ordered[:,2],order,digest

def run(out:Path):
 if out.exists(): raise FileExistsError(f'choose a fresh output directory: {out}')
 started=time.perf_counter();x,y,date,period,ids,digest=load_ele2();out.mkdir(parents=True)
 # Fixed contiguous historical window, then chronological monitoring.
 train_end,cal_end,ref_end=10800,14462,18124
 if ref_end>=len(y):raise AssertionError('history extends beyond source')
 scaler=StandardScaler().fit(x[:train_end]);xt=scaler.transform(x[:train_end]);
 model=LogisticRegression(C=1.0,max_iter=1000,random_state=2242026).fit(xt,y[:train_end])
 pca=PCA(n_components=3,svd_solver='full').fit(xt)
 cluster=MiniBatchKMeans(n_clusters=3,n_init=3,batch_size=256,random_state=2242026).fit(xt)
 calp=model.predict_proba(scaler.transform(x[train_end:cal_end]));calpred=model.classes_[calp.argmax(axis=1)]
 calq=(calpred==y[train_end:cal_end]).astype(np.int8);calconf=calp.max(axis=1)
 calibrator=IsotonicRegression(out_of_bounds='clip').fit(calconf,calq)
 refx=x[cal_end:ref_end];refy=y[cal_end:ref_end];refp=model.predict_proba(scaler.transform(refx))
 refcorrect=int(np.sum(model.classes_[refp.argmax(axis=1)]==refy));refn=len(refy);refacc=refcorrect/refn
 ref_input=refx[-1000:];ref_latent=pca.transform(scaler.transform(ref_input))
 np.savez_compressed(out/'evaluator_truth.npz',row_id=ids[ref_end:],y=y[ref_end:])
 records=[]; batch_size=500
 for b,start in enumerate(range(ref_end,len(y),batch_size),start=1):
  stop=min(start+batch_size,len(y)); bx=x[start:stop]; bid=ids[start:stop]
  bxz=scaler.transform(bx);prob=model.predict_proba(bxz);pred=model.classes_[prob.argmax(axis=1)]
  conf=prob.max(axis=1);q=np.clip(calibrator.predict(conf),0,1)
  latent=pca.transform(bxz);pm,_=impact(q,refcorrect,refn)
  pvalues=[ks_2samp(ref_input[:,j],bx[:,j],method='auto').pvalue for j in range(x.shape[1])]
  pvalues += [ks_2samp(ref_latent[:,j],latent[:,j],method='auto').pvalue for j in range(3)]
  adjusted=holm(pvalues);labels=cluster.predict(bxz);occupancy=np.bincount(labels,minlength=3)
  np.savez_compressed(out/f'batch_{b:03}.npz',row_id=bid,prediction=pred,probability=prob,confidence=conf,calibrated_q=q)
  r={'batch':b,'n':len(bx),'first_source_row':int(bid[0]),'last_source_row':int(bid[-1]),
    'date_start':float(date[start]),'date_end':float(date[stop-1]),'period_start':float(period[start]),'period_end':float(period[stop-1]),
    'input_alarm':bool(np.any(adjusted<.05)),'alarm_count':int(np.sum(adjusted<.05)),
    'estimate':pm['estimate'],'interval_lower':pm['lower'],'interval_upper':pm['upper'],'interval_width':pm['upper']-pm['lower'],
    'concern_score':pm['concern'],'concern_flag':pm['concern_flag'],'point_flag':pm['point_flag'],
    'historical_reference_accuracy':refacc,'confidence_mean':float(conf.mean()),'mean_q':float(q.mean()),
    'latent_rejections':int(np.sum(adjusted[6:]<.05)),'feature_pvalues':json.dumps(pvalues),
    'feature_holm_pvalues':json.dumps(adjusted.tolist()),'cluster_counts':json.dumps(occupancy.tolist()),
    'prediction_seconds':None}
  for j in range(6):r[f'x{j}_ks']=float(ks_2samp(ref_input[:,j],bx[:,j],method='auto').statistic)
  records.append(r)
 online=pd.DataFrame(records)
 if set(online)&{'accuracy','correct','harm','y','label','macro_f1'}:raise AssertionError('online outputs leak truth')
 online.to_csv(out/'online_batches.csv',index=False)
 meta={'status':'online_complete_evaluator_pending','dataset':'OpenML electricity 151 v1 / ELEC2',
  'source_sha256':digest,'runner_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  'rows':len(y),'rows_sorted_by':['normalized_date','first_source_occurrence_of_date_weekday_group','period','source_row_index'],
  'source_file_order_was_not_date_monotone':True,'normalized_date_values_with_multiple_weekday_groups':11,
  'feature_names':['day','period','nswdemand','vicprice','vicdemand','transfer'],
  'excluded':['normalized_date (ordering only)','nswprice (target-proximal)'],
  'split':{'train':[0,train_end],'calibration':[train_end,cal_end],'reference':[cal_end,ref_end],'monitor':[ref_end,len(y)],
           'reference_input_rows':1000,'batch_size':500},'monitor_batches':len(records),
  'delays':[0,5,20],'online_truth_columns_absent':True,
  'environment':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
  'elapsed_seconds':time.perf_counter()-started}
 (out/'protocol_lock.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
 print(json.dumps({k:v for k,v in meta.items() if k not in ('environment',)},indent=2))
 return meta

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();run(args.out)
