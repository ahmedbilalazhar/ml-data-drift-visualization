"""Exploratory persistence forecasts and corrected external reference metadata."""
from pathlib import Path
import argparse,json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def main(out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    data=pd.read_csv(ROOT/'results/phase3/final_v1/evaluator_batches.csv')
    rows=[];forecasts=[]
    for (seed,condition),g in data.groupby(['seed','condition']):
        g=g.sort_values('batch')
        for horizon in (0,1,5):
            now=g.iloc[:24-horizon];future=g.iloc[horizon:]
            for method,col in [('calibrated_persistence','cal_estimate'),('raw_persistence','raw_estimate'),('ATC_persistence','atc_estimate'),('historical_constant','constant_estimate')]:
                estimated=now[col].to_numpy();actual=future.accuracy.to_numpy()
                forecasts.extend(dict(seed=int(seed),condition=condition,issued_at=int(b),target_batch=int(b+horizon),
                    horizon=horizon,method=method,estimate=float(q)) for b,q in zip(now.batch,estimated))
                rows.append(dict(seed=seed,condition=condition,horizon=horizon,method=method,
                    cohorts=len(now),mae=float(np.abs(estimated-actual).mean()),bias=float((estimated-actual).mean())))
    d=pd.DataFrame(rows);d.to_csv(out/'horizon_per_seed.csv',index=False)
    pd.DataFrame(forecasts).to_csv(out/'online_persistence_forecasts.csv',index=False)
    d.groupby(['condition','horizon','method'])[['mae','bias','cohorts']].mean().reset_index().to_csv(out/'horizon_summary.csv',index=False)
    # Corrections live beside, rather than overwrite, the preserved external study.
    source=ROOT/'results/phase3/external_uci224_day2_v2'
    lock=json.loads((source/'protocol_lock.json').read_text())
    idx=lock['historical_partition_row_indices']['batch1_reference']
    from scripts.evaluate_uci224_day2 import load_cohorts
    _,labels=load_cohorts()[1]
    metadata=[]
    for population,y in [('source_batch1',labels),('reference_partition',labels[idx])]:
        counts=np.bincount(y,minlength=7)
        for k in range(1,7):metadata.append(dict(population=population,**{'class':k},count=int(counts[k]),denominator=len(y),proportion=float(counts[k]/len(y))))
    m=pd.DataFrame(metadata)
    assert np.allclose(m.groupby('population').proportion.sum(),1)
    m.to_csv(out/'corrected_external_history_class_composition.csv',index=False)
    (out/'protocol_and_limits.json').write_text(json.dumps(dict(status='complete',outcome_informed=True,
        forecast_rule='At t persist the source-calibrated/raw/ATC estimate to target t+h; no future features or labels used',
        horizons=[0,1,5],forecast_intervals_claimed=False,
        limits='Simple persistence benchmarks, not learned or validated general future-risk forecasting. Different horizons have different valid target cohorts. Compare methods within horizon.',
        external_fix='Corrects historical source/reference class-count denominators in separate metadata; all original outcomes unchanged'),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
