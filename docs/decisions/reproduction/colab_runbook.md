# Colab runbook — fire-and-forget heavy execution (roadmap Sec. 13.6)

PRECONDITION: G4 freeze signed (`experiments/protocols/freeze_g4_draft.md`).
Laptop never runs this; pilot evidence already covers laptop-side claims.

## 1. Prepare (first cell)
```python
!pip install -q scikit-learn pandas scipy matplotlib pyyaml pyarrow
from google.colab import drive; drive.mount('/content/drive')
%cd /content/drive/MyDrive/drift-monitoring   # your checkout copy
!git log --oneline -1; !python -c "import sklearn,scipy,pandas; print('deps ok')"
!df -h /content/drive | tail -1               # free space for artifacts
```

## 2. Verify (never skip)
```python
!python scripts/audit_datasets.py              # checksums must pass
!head -3 experiments/registries/runs.csv       # history intact, append-only
```

## 3. Execute — ONE mode per session, sequential jobs only
```python
!python scripts/run_full_inventory.py --mode short --seeds 0 1 2 3 4
# next session: !python scripts/run_full_inventory.py --mode stationary
```
Reruns skip completed evidence files automatically. Failures append to the
registry with reason — never delete rows, never edit thresholds mid-run.

## 4. Checkpoint & close (every session)
```python
!cat results/aggregates/budget.csv             # vs 25%-contingency envelope
# Drive syncs automatically; verify read-back:
!ls results/runs/_full/*.evidence.csv | wc -l
```
Update the registry count in your weekly note; note remaining jobs.

## 5. Recover
Fresh session → mount → cd → rerun the SAME command. Resume continues from the
recorded next window; afterwards check for duplicate/missing window_ids:
```python
!python -c "import pandas as pd,glob
d=[pd.read_csv(f) for f in glob.glob('results/runs/_full/*.evidence.csv')]
print('dup windows:', sum(x.window_id.duplicated().sum() for x in d))"
```

## Interruption drill (week 4 requirement)
Run `--mode short --seeds 0 --templates S0`, kill the runtime mid-job, reconnect,
rerun — expect `skipped-complete` or clean continuation, zero duplicate windows.
