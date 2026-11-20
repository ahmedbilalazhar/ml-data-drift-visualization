$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $root

$python = if ($env:IDMVIZ_PYTHON) { $env:IDMVIZ_PYTHON } else { 'C:\Users\hp\anaconda3\python.exe' }
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found: $python. Set IDMVIZ_PYTHON to the project environment."
}

& $python scripts/check_dde_seed_collisions.py
if ($LASTEXITCODE -ne 0) { throw "Seed collision check failed with exit code $LASTEXITCODE" }
& $python scripts/run_dde_subset_audit.py --workers 2
if ($LASTEXITCODE -ne 0) { throw "DDE-style audit runner failed with exit code $LASTEXITCODE" }
& $python scripts/report_dde_subset_audit.py
if ($LASTEXITCODE -ne 0) { throw "DDE-style audit report failed with exit code $LASTEXITCODE" }
