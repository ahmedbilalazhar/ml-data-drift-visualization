param(
  [int]$Replicates = 1000,
  [string]$RunDir,
  [string]$ReportDir
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
if (-not $RunDir) { $RunDir = "results/phase3/noaa_selective_label_replay_reproduction_$stamp" }
if (-not $ReportDir) { $ReportDir = "reports/phase3/noaa_selective_label_replay_reproduction_$stamp" }
python scripts/run_noaa_selective_label_replay.py --replicates $Replicates --output $RunDir
if ($LASTEXITCODE -ne 0) { throw "Sampling replay failed with exit code $LASTEXITCODE" }
python scripts/summarize_noaa_selective_label_replay.py --run $RunDir --output $ReportDir
if ($LASTEXITCODE -ne 0) { throw "Report compilation failed with exit code $LASTEXITCODE" }
Write-Host "Run: $RunDir"
Write-Host "Report: $ReportDir"
