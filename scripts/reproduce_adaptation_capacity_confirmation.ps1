$ErrorActionPreference = "Stop"
$python = "C:\Users\hp\anaconda3\python.exe"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$run = "results/phase3/adaptation_capacity_confirmation_reproduction_$stamp"
$report = "reports/phase3/adaptation_capacity_confirmation_reproduction_$stamp"
& $python scripts/run_adaptation_capacity_control.py `
  --config configs/phase3_adaptation_capacity_control_confirmation_20261118.json `
  --output $run
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $python scripts/report_adaptation_capacity_confirmation.py --run $run --out $report
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
