"""Create versioned compact/full archives with explicit checksum inventories."""
from pathlib import Path
import argparse
from datetime import datetime,timezone,timedelta
import hashlib
import importlib.metadata
import json
from zipfile import ZipFile,ZIP_DEFLATED,ZIP_STORED

ROOT=Path(__file__).resolve().parents[1]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def package(version):
    run=ROOT/'results/phase3/final_v1';report=ROOT/'reports/phase3/final_v1'
    validation=ROOT/'reports/phase3/validation'
    lock=json.loads((run/'protocol_lock.json').read_text())
    assert lock['status']=='complete' and lock['seeds']==list(range(100,130))
    assert json.loads((run/'validation.json').read_text())['status']=='passed'
    assert json.loads((validation/'reproduction_comparison.json').read_text())['status']=='passed'
    browser=json.loads((validation/'browser_validation.json').read_text())
    assert browser['success'] and browser['data']['result']['status']=='passed'
    for name,digest in lock['original_documents_sha256'].items():
        assert sha(ROOT/name)==digest,f'original document changed: {name}'
    for name,digest in lock['source_sha256'].items():
        assert sha(ROOT/name)==digest,f'numerical source changed after final lock: {name}'
    target=ROOT/'output/phase3';target.mkdir(parents=True,exist_ok=True)
    prefix=f'IDM_Viz_Phase3_{version}'
    files={}
    def add(p,name=None):
        files[name or p.relative_to(ROOT).as_posix()]=p
    for directory in ['src','tests']:
        for p in (ROOT/directory).rglob('*.py'):add(p)
    for p in (ROOT/'scripts').glob('*phase3*'):
        if p.is_file():add(p)
    for p in (ROOT/'docs/phase3').glob('*'):
        if p.is_file():add(p)
    for name in lock['original_documents_sha256']:add(ROOT/name)
    for p in report.rglob('*'):
        if p.is_file() and 'qa' not in p.relative_to(report).parts:add(p)
    for p in validation.glob('*'):
        if p.is_file():add(p)
    for p in run.glob('*'):
        if p.is_file():add(p)
    for name in ['pyproject.toml','requirements-phase3.txt']:add(ROOT/name)
    add(ROOT/'docs/phase3/REPRODUCTION.md','README.md')
    created=datetime.now(timezone.utc)
    base=dict(version=version,created_utc=created.isoformat(),
        created_karachi=created.astimezone(timezone(timedelta(hours=5))).isoformat(),
        submission_status='prepared_not_uploaded',methodological_basis='submitted Phase 2 DOCX',
        legacy_results_used=False,final_lock_sha256=sha(run/'protocol_lock.json'),
        preservation='Archives refuse overwrite. Preserve v1 and use a new version for improvements.')
    receipts=[]
    for kind in ['submission','full_reproducibility']:
        payload=files.copy()
        if kind=='full_reproducibility':
            for directory in ['predictions','evaluator_truth']:
                for p in (run/directory).glob('*'):payload[p.relative_to(ROOT).as_posix()]=p
        archive=target/f'{prefix}_{kind}.zip'
        if archive.exists():raise FileExistsError(archive)
        inventory={**base,'archive_kind':kind,'instance_caches_included':kind=='full_reproducibility',
            'files':{name:dict(sha256=sha(p),bytes=p.stat().st_size) for name,p in sorted(payload.items())}}
        with ZipFile(archive,'x',compression=ZIP_DEFLATED,compresslevel=6) as z:
            for name,p in sorted(payload.items()):
                z.write(p,f'{prefix}/{name}',compress_type=ZIP_STORED if p.suffix=='.npz' else ZIP_DEFLATED)
            z.writestr(f'{prefix}/PACKAGE_INVENTORY.json',json.dumps(inventory,indent=2))
        # Read every member and check its actual bytes, not just ZIP CRC headers.
        with ZipFile(archive) as z:
            assert z.testzip() is None
            for name,item in inventory['files'].items():
                blob=z.read(f'{prefix}/{name}')
                assert hashlib.sha256(blob).hexdigest()==item['sha256'],name
        receipt=dict(path=str(archive.resolve()),sha256=sha(archive),bytes=archive.stat().st_size,
                     payload_files=len(payload),all_member_sha256_checks='passed',kind=kind)
        receipts.append(receipt)
        print(json.dumps(receipt,indent=2),flush=True)
    receipt_path=target/f'{prefix}_receipt.json'
    receipt_path.write_text(json.dumps({**base,'archives':receipts},indent=2))
    print(receipt_path)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--version',required=True)
    a=p.parse_args();package(a.version)
