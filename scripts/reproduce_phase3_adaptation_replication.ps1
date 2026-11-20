param(
    [string]$RunDir = 'results/phase3/adaptation_seed_replication_reproduction',
    [string]$ReportDir = 'reports/phase3/adaptation_seed_replication_reproduction',
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
    & $ScientificPython scripts/run_phase3_adaptation_replication.py --mode final --out $RunDir
    if ($LASTEXITCODE -ne 0) { throw 'Adaptation replication failed.' }
    & $ScientificPython scripts/evaluate_phase3_adaptation_replication.py --run $RunDir
    if ($LASTEXITCODE -ne 0) { throw 'Adaptation evaluation failed.' }
    & $ScientificPython scripts/report_phase3_adaptation_replication.py --run $RunDir --out $ReportDir
    if ($LASTEXITCODE -ne 0) { throw 'Adaptation report generation failed.' }
    & $ArtifactPython scripts/render_phase3_v2_pdf.py (Join-Path $ReportDir '23I-2581_23I-2604_Phase3_Adaptation_Replication.md')
    if ($LASTEXITCODE -ne 0) { throw 'PDF rendering failed.' }
} finally { Pop-Location }
