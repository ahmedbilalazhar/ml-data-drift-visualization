param(
    [string]$RunDir = 'results/phase3/large_seed_robustness_reproduction',
    [string]$ReportDir = 'reports/phase3/large_seed_robustness_reproduction',
    [string]$ScientificPython = 'C:/Users/hp/anaconda3/python.exe',
    [string]$ArtifactPython = 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe'
)
$ErrorActionPreference='Stop'
$Root=Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $Root
try {
    if ((Test-Path -LiteralPath $RunDir) -or (Test-Path -LiteralPath $ReportDir)) {
        throw 'Choose unused output paths; existing runs are preserved.'
    }
    & $ScientificPython scripts/run_phase3_large_seed.py --mode final --out $RunDir
    if ($LASTEXITCODE -ne 0) { throw 'Large-seed run failed.' }
    & $ScientificPython scripts/evaluate_phase3_large_seed.py --run $RunDir
    if ($LASTEXITCODE -ne 0) { throw 'Large-seed evaluation failed.' }
    $SampleDir=Join-Path $ReportDir 'sample_reproduction'
    & $ScientificPython scripts/compare_phase3_large_seed_sample.py --run $RunDir --out $SampleDir
    if ($LASTEXITCODE -ne 0) { throw 'Independent sample comparison failed.' }
    & $ScientificPython scripts/report_phase3_large_seed.py --run $RunDir --out $ReportDir --reproduction (Join-Path $SampleDir 'sample_reproduction_comparison.json')
    if ($LASTEXITCODE -ne 0) { throw 'Report generation failed.' }
    & $ArtifactPython scripts/render_phase3_v2_pdf.py (Join-Path $ReportDir '23I-2581_23I-2604_Phase3_Robustness_Report.md')
    if ($LASTEXITCODE -ne 0) { throw 'PDF rendering failed.' }
} finally { Pop-Location }
