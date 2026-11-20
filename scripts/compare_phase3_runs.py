"""Compare independent runs, excluding clocks, elapsed timings and binary ZIP metadata."""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd


def compare(first,second,out):
    a=json.loads((first/'protocol_lock.json').read_text());b=json.loads((second/'protocol_lock.json').read_text())
    assert a['status']==b['status']=='complete'
    for key in ['seeds','original_manifest_sha256','original_documents_sha256']:
        assert a[key]==b[key],key
    # Reporting/audit scripts may be added after numerical lock; every originally
    # locked source file (including the entire numerical pipeline/tests) must match.
    for name,digest in a['source_sha256'].items():
        assert b['source_sha256'].get(name)==digest,name
    # Scientific CSVs, including all per-instance releases through their counts/ID hashes.
    files=['online_batches.csv','released_verification.csv','clock_events.csv','fit_records.csv','warnings.csv',
           'evaluator_batches.csv','impact_per_seed.csv','impact_summary.csv','harm_counts.csv',
           'detection_per_seed.csv','detection_summary.csv','localization_batches.csv',
           'localization_per_seed.csv','localization_summary.csv','overall_per_seed.csv','overall_summary.csv',
           'paired_seed_differences.csv','paired_bootstrap.csv','calibration_error_per_seed.csv',
           'score_loss_correlations.csv','delay_availability.csv']
    cells=0
    for name in files:
        left=pd.read_csv(first/name);right=pd.read_csv(second/name)
        excluded=[c for c in left if c.endswith('_seconds')]
        left=left.drop(columns=excluded);right=right.drop(columns=excluded)
        if 'metric' in left:
            keep=~left.metric.str.endswith('_seconds')
            left=left[keep].reset_index(drop=True);right=right[keep].reset_index(drop=True)
        pd.testing.assert_frame_equal(left,right,check_exact=True)
        cells+=left.size
    arrays=0;items=0;archives=0
    for directory in ['predictions','evaluator_truth']:
        names=sorted(p.name for p in (first/directory).glob('*.npz'))
        assert names==sorted(p.name for p in (second/directory).glob('*.npz'))
        for name in names:
            with np.load(first/directory/name) as left,np.load(second/directory/name) as right:
                assert left.files==right.files
                for key in left.files:
                    np.testing.assert_array_equal(left[key],right[key])
                    arrays+=1;items+=left[key].size
            archives+=1
    record=dict(status='passed',first=str(first.resolve()),second=str(second.resolve()),
                final_seeds=len(a['seeds']),csv_files_compared=len(files),csv_cells_compared=cells,
                npz_archives_compared=archives,arrays_compared=arrays,array_values_compared=items,
                additional_postprocessing_sources=sorted(set(b['source_sha256'])-set(a['source_sha256'])),
                comparison='exact scientific CSV/array equality; elapsed timings and archive metadata excluded')
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(record,indent=2))
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--first',type=Path,required=True)
    p.add_argument('--second',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();compare(a.first,a.second,a.out)
