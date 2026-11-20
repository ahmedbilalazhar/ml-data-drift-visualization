param(
    [string]$RunDir = 'results/phase3/reproduced_final',
    [string]$ReportDir = 'reports/phase3/reproduced_final',
    [string]$ScientificPython = 'C:/Users/hp/anaconda3/python.exe',
    [string]$ArtifactPython = 'C:/Program Files/WindowsApps/OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt/dependencies/python/python.exe'
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if ((Test-Path -LiteralPath $RunDir) -or (Test-Path -LiteralPath $ReportDir)) {
        throw 'Choose fresh RunDir and ReportDir paths; existing snapshots are never overwritten.'
    }
    & $ScientificPython -m pytest tests -q
    if ($LASTEXITCODE -ne 0) { throw 'Tests failed' }
    & $ScientificPython scripts/run_phase3.py --mode final --out $RunDir
    if ($LASTEXITCODE -ne 0) { throw 'Final runner failed' }
    & $ScientificPython scripts/evaluate_phase3.py --run $RunDir
    if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed' }
    & $ScientificPython scripts/report_phase3.py --run $RunDir --out $ReportDir
    if ($LASTEXITCODE -ne 0) { throw 'Figure/report generation failed' }
    & $ScientificPython scripts/build_phase3_view.py --run $RunDir --out (Join-Path $ReportDir 'evidence_view.html')
    if ($LASTEXITCODE -ne 0) { throw 'Evidence view generation failed' }
    & $ArtifactPython scripts/render_phase3_pdf.py (Join-Path $ReportDir 'IDM_Viz_Phase3_Report.md')
    if ($LASTEXITCODE -ne 0) { throw 'PDF rendering failed' }
} finally {
    Pop-Location
}
