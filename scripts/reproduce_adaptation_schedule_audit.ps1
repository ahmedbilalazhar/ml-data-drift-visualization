param(
  [string]$RunDir,
  [string]$ReportDir,
  [Nullable[int]]$SeedLimit
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
if (-not $RunDir) { $RunDir = "results/phase3/adaptation_schedule_audit_reproduction_$stamp" }
if (-not $ReportDir) { $ReportDir = "reports/phase3/adaptation_schedule_audit_reproduction_$stamp" }
$runnerArgs = @('scripts/run_adaptation_schedule_audit.py', '--output', $RunDir)
if ($null -ne $SeedLimit) { $runnerArgs += @('--seed-limit', "$SeedLimit") }
python @runnerArgs
if ($LASTEXITCODE -ne 0) { throw "Experiment runner failed with exit code $LASTEXITCODE" }
python scripts/report_adaptation_schedule_audit.py --run $RunDir --out $ReportDir
if ($LASTEXITCODE -ne 0) { throw "Report builder failed with exit code $LASTEXITCODE" }
$tex = Join-Path $ReportDir 'adaptation_schedule_audit.tex'
tectonic -X compile $tex --outdir $ReportDir --keep-logs
if ($LASTEXITCODE -ne 0) { throw "TeX compilation failed with exit code $LASTEXITCODE" }
Write-Host "Run: $RunDir"
Write-Host "Report: $ReportDir"
