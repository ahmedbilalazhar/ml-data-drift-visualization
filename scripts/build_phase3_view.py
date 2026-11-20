"""Self-contained offline replay UI. Truth is always filtered by release clock."""
from pathlib import Path
import argparse
import json
import math
import pandas as pd


def clean(value):
    if isinstance(value,float) and not math.isfinite(value): return None
    if isinstance(value,dict): return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list): return [clean(v) for v in value]
    return value


def build(run,out):
    online=pd.read_csv(run/'online_batches.csv')
    verified=pd.read_csv(run/'released_verification.csv')
    # No evaluator CSV is loaded. Release records retain their actual available_at.
    cols=['seed','condition','batch','cohort','n','cal_estimate','cal_lower','cal_upper','cal_concern',
          'cal_concern_flag','input_alarm','min_adjusted_p','reference_accuracy']
    cols += [f'x{j}_{suffix}' for j in range(8) for suffix in ['ks','rank','flag']]
    cols += [f'cluster{j}_{suffix}' for j in range(3) for suffix in ['n','occupancy','q']]
    data=clean(dict(online=online[cols].to_dict('records'),
        releases=verified[['cohort','delay','available_at','accuracy','harm','macro_f1']].to_dict('records')))
    template=Path(__file__).with_name('phase3_view_template.html').read_text(encoding='utf-8')
    out.write_text(template.replace('/*__DATA__*/',json.dumps(data,separators=(',',':'),allow_nan=False)),encoding='utf-8')
    print(f'{out}: {out.stat().st_size:,} bytes')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();build(a.run,a.out)
