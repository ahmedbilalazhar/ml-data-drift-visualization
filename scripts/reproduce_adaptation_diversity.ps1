$ErrorActionPreference = "Stop"
$python = "C:\Users\hp\anaconda3\python.exe"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$run = "results/phase3/matched_adaptation_diversity_reproduction_$stamp"
$report = "reports/phase3/matched_adaptation_diversity_reproduction_$stamp"
& $python scripts/run_matched_adaptation_diversity.py `
  --config configs/phase3_adaptation_diversity_20261118.json `
  --output $run
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python scripts/report_matched_adaptation_diversity.py --run $run --out $report
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
