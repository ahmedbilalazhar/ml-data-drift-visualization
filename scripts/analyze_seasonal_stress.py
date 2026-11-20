from pathlib import Path
import io,json,zipfile,hashlib,argparse
import numpy as np,pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'results/phase3/seasonal_stress_evaluation_20261118_v1'
OUT=RUN/'analysis'

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def cell_data(run_dir, run, cell):
 item=run['completed_cells'][cell]; p=run_dir/item['artifact']
 if digest(p)!=item['sha256']: raise ValueError('artifact checksum mismatch: '+cell)
 with zipfile.ZipFile(p) as z:
  online=pd.read_csv(io.BytesIO(z.read('online/batch_scores.csv')))
  delays=pd.read_csv(io.BytesIO(z.read('run/delayed_verification.csv')))
  t=np.load(io.BytesIO(z.read('evaluator/truth.npz')))
  labels=t['labels'].copy(); expected=t['expected_label_probability'].copy(); pred=np.load(io.BytesIO(z.read('online/prediction_cache.npz')))['prediction'].copy()
  truth=json.loads(z.read('evaluator/truth_metadata.json'))
 return online,delays,labels,pred,expected,truth

def boot(x,seed=8112,reps=10000):
 x=np.asarray(x,float); rng=np.random.default_rng(seed)
 a=rng.integers(0,len(x),(reps,len(x))); m=x[a].mean(axis=1)
 return [float(x.mean()),float(np.quantile(m,.025)),float(np.quantile(m,.975))]

def main():
 global RUN, OUT
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--run',type=Path,default=RUN,help='completed seasonal run directory')
 parser.add_argument('--primary',type=Path,default=ROOT/'results/phase3/strengthening_v3_evaluation_d8',help='matched stationary primary run')
 parser.add_argument('--output',type=Path,default=None,help='analysis output directory; defaults to RUN/analysis')
 args=parser.parse_args(); RUN=args.run.resolve(); OUT=(args.output or RUN/'analysis').resolve(); primary_dir=args.primary.resolve()
 OUT.mkdir(parents=True,exist_ok=True)
 lock=json.loads((RUN/'protocol_lock.json').read_text())
 if lock.get('state')!='complete' or len(lock['completed_cells'])!=90 or set(lock['completed_cells'])!=set(lock['expected_cells']): raise ValueError('seasonal study not complete')
 records=[]; daily=[]
 for task in ('linear_probabilistic','xor_nonlinear','five_region'):
  for seed in map(int, __import__('json').loads((ROOT/'configs/phase3_strengthening_v3.json').read_text())['seed_banks']['evaluation']):
   sc=f'{task}|S1_seasonal_nuisance|{seed}|d8'; bc=f'{task}|M0_stationary|{seed}|d8'
   on,delay,y,pred,prob,truth=cell_data(RUN,lock,sc); primary_lock=json.loads((primary_dir/'protocol_lock.json').read_text());base,bdelay,by,bpred,bprob,btruth=cell_data(primary_dir,primary_lock,bc)
   if not np.array_equal(y,by) or not np.array_equal(prob,bprob): raise ValueError(f'paired outcome truth mismatch {sc}')
   if not np.array_equal(on.batch.to_numpy(),np.arange(1,97)): raise ValueError('batch order mismatch')
   seasonal_acc=np.array([(pred[i*500:(i+1)*500]==y[i*500:(i+1)*500]).mean() for i in range(96)])
   base_acc=np.array([(bpred[i*500:(i+1)*500]==by[i*500:(i+1)*500]).mean() for i in range(96)])
   rec={'task':task,'seed':seed,'accuracy_seasonal':float(seasonal_acc.mean()),'accuracy_stationary':float(base_acc.mean()),'accuracy_delta':float((seasonal_acc-base_acc).mean()),'x7_alarm_rate':float(on.x7_flag.astype(bool).mean()),'x7_stationary_rate':float(base.x7_flag.astype(bool).mean()),'x7_alarm_delta':float(on.x7_flag.astype(bool).mean()-base.x7_flag.astype(bool).mean()),'input_alarm_rate':float(on.input_alarm.astype(bool).mean()),'input_stationary_rate':float(base.input_alarm.astype(bool).mean()),'input_alarm_delta':float(on.input_alarm.astype(bool).mean()-base.input_alarm.astype(bool).mean()),'pca_alarm_rate':float(on.latent_rejection.astype(bool).mean()),'correlation_alarm_rate':float(on.correlation_alarm.astype(bool).mean()),'max_abs_true_probability_delta':float(np.max(np.abs(prob-bprob)))}
   records.append(rec)
   for b in range(96): daily.append({'task':task,'seed':seed,'batch':b+1,'seasonal_accuracy':seasonal_acc[b],'stationary_accuracy':base_acc[b],'accuracy_delta':seasonal_acc[b]-base_acc[b],'x7_alarm':int(bool(on.x7_flag.iloc[b])),'input_alarm':int(bool(on.input_alarm.iloc[b])),'stationary_input_alarm':int(bool(base.input_alarm.iloc[b])),'seasonal_shift':float(.6*np.sin(2*np.pi*b/24))})
 seed_df=pd.DataFrame(records); temporal=pd.DataFrame(daily)
 seed_df.to_csv(OUT/'seasonal_paired_per_seed.csv',index=False); temporal.to_csv(OUT/'seasonal_temporal_per_seed.csv',index=False)
 summaries=[]
 for task,g in seed_df.groupby('task'):
  for metric in ('accuracy_delta','x7_alarm_delta','input_alarm_delta','x7_alarm_rate','x7_stationary_rate','input_alarm_rate','input_stationary_rate','pca_alarm_rate','correlation_alarm_rate'):
   m,lo,hi=boot(g[metric].to_numpy(),seed=9001+len(summaries))
   summaries.append({'task':task,'metric':metric,'mean':m,'ci95_low':lo,'ci95_high':hi,'seeds':len(g)})
 summ=pd.DataFrame(summaries); summ.to_csv(OUT/'seasonal_summary.csv',index=False)
 # Figures: emphasize paired outcome contrast and detector timeline.
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titleweight':'bold','axes.spines.top':False,'axes.spines.right':False})
 colors={'linear_probabilistic':'#167D8D','xor_nonlinear':'#E07A5F','five_region':'#7666A8'}
 fig,ax=plt.subplots(figsize=(8.4,4.6),layout='constrained')
 for j,(task,g) in enumerate(seed_df.groupby('task',sort=False)):
  mean,lo,hi=boot(g.accuracy_delta.to_numpy(),seed=520+j)
  ax.errorbar(mean,j,xerr=[[mean-lo],[hi-mean]],fmt='o',ms=8,lw=2.3,capsize=4,color=colors[task],label=task.replace('_',' '))
 ax.axvline(0,color='#46515C',lw=1.2,ls='--');ax.set_yticks(range(3),[x.replace('_',' ') for x in seed_df.task.drop_duplicates()]);ax.set_xlabel('Paired accuracy difference (seasonal − stationary)');ax.set_title('Outcome impact of nuisance seasonality',loc='left',pad=12);ax.grid(axis='x',alpha=.2);fig.savefig(OUT/'fig_seasonal_paired_accuracy.svg');fig.savefig(OUT/'fig_seasonal_paired_accuracy.png',dpi=240);plt.close(fig)
 fig,axes=plt.subplots(2,1,figsize=(9.2,6.2),sharex=True,layout='constrained')
 x=np.arange(1,97)
 for task,g in temporal.groupby('task',sort=False):
  s=g.groupby('batch').agg(x7=('x7_alarm','mean'),input=('input_alarm','mean'),base=('stationary_input_alarm','mean'),shift=('seasonal_shift','mean'),acc=('accuracy_delta','mean'))
  axes[0].plot(x,s.x7,color=colors[task],lw=1.8,label=task.replace('_',' '))
  axes[1].plot(x,s.acc,color=colors[task],lw=1.8,label=task.replace('_',' '))
 axes[0].set_ylabel('Feature 7 alarm rate');axes[0].set_title('Seasonal timing and diagnostic response',loc='left');axes[0].legend(frameon=False,ncol=3);axes[0].grid(alpha=.2)
 axes[1].axhline(0,color='#46515C',lw=1,ls='--');axes[1].set_ylabel('Paired batch accuracy Δ');axes[1].set_xlabel('Batch (seasonal period = 24 batches)');axes[1].grid(alpha=.2)
 for a in axes:
  for k in (1,25,49,73):a.axvline(k,color='#ABB5BF',lw=.7,ls=':')
 fig.savefig(OUT/'fig_seasonal_timeline.svg');fig.savefig(OUT/'fig_seasonal_timeline.png',dpi=240);plt.close(fig)
 manifest={'study':'S1 seasonal nuisance stress; separate from primary matrix','run_dir':str(RUN.relative_to(ROOT)),'run_state':lock['state'],'cells':len(lock['completed_cells']),'tasks':3,'seeds_per_task':30,'batches':96,'batch_size':500,'seed_unit':'paired evaluation seed within task','bootstrap_replicates':10000,'bootstrap_unit':'seed within task','seasonal_feature':'index 7, additive amplitude 0.6, period 24 batches, begins at batch 1','outcome_law':'paired labels and rowwise expected label probabilities checked identical against M0','run_lock_sha256':digest(RUN/'protocol_lock.json'),'analysis_script_sha256':digest(Path(__file__)),'artifacts_verified':True,'files':{p.name:digest(p) for p in OUT.iterdir() if p.is_file()}}
 (OUT/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
 print(summ.to_string(index=False));print('analysis:',OUT)
if __name__=='__main__':main()
