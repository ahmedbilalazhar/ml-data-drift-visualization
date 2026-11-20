param(
    [string]$RunDir = 'results/phase3/adaptation_reproduction',
    [string]$ReportDir = 'reports/phase3/adaptation_reproduction'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$Python = 'C:\Users\hp\anaconda3\python.exe'
$ArtifactPython = 'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe'
$Markdown = Join-Path $ReportDir '23I-2581_23I-2604_Phase3_Adaptation_Report.md'

& $Python scripts/run_phase3_adaptation.py --mode final --out $RunDir
if ($LASTEXITCODE -ne 0) { throw 'Adaptation runner failed.' }
& $Python scripts/evaluate_phase3_adaptation.py --run $RunDir
if ($LASTEXITCODE -ne 0) { throw 'Adaptation evaluation failed.' }
& $Python scripts/compare_phase3_adaptation.py --first results/phase3/adaptation_final --second $RunDir --out (Join-Path $ReportDir 'reproduction_comparison.json')
if ($LASTEXITCODE -ne 0) { throw 'Adaptation reproduction comparison failed.' }
& $Python scripts/report_phase3_adaptation.py --run $RunDir --out $ReportDir
if ($LASTEXITCODE -ne 0) { throw 'Adaptation report failed.' }
& $ArtifactPython scripts/render_phase3_v2_pdf.py $Markdown
if ($LASTEXITCODE -ne 0) { throw 'Adaptation PDF rendering failed.' }
