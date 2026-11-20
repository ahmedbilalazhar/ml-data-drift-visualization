param(
    [string]$RunDir = 'results/phase3/v2_reproduction',
    [string]$ReportDir = 'reports/phase3/v2_reproduction'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $Root
$ScientificPython = 'C:\Users\hp\anaconda3\python.exe'
$ArtifactPython = 'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\dependencies\python\python.exe'
$PdfMarkdown = Join-Path $ReportDir '23I-2581_23I-2604_Phase3_Report.md'
$ViewPath = Join-Path $ReportDir '23I-2581_23I-2604_Phase3_Evidence_Replay.html'

& $ScientificPython scripts/run_phase3_v2.py --mode final --out $RunDir
if ($LASTEXITCODE -ne 0) { throw 'Final v2 runner failed.' }
& $ScientificPython scripts/evaluate_phase3_v2.py --run $RunDir
if ($LASTEXITCODE -ne 0) { throw 'V2 evaluation failed.' }
& $ScientificPython scripts/build_phase3_v2_view.py --run $RunDir --out $ViewPath
if ($LASTEXITCODE -ne 0) { throw 'V2 evidence replay build failed.' }
& $ScientificPython scripts/report_phase3_v2.py --run $RunDir --out $ReportDir
if ($LASTEXITCODE -ne 0) { throw 'V2 report build failed.' }
& $ArtifactPython scripts/render_phase3_v2_pdf.py $PdfMarkdown
if ($LASTEXITCODE -ne 0) { throw 'V2 PDF rendering failed.' }
& $ScientificPython scripts/compare_phase3_v2.py --first results/phase3/v2_final --second $RunDir --out (Join-Path $ReportDir 'reproduction_comparison.json')
if ($LASTEXITCODE -ne 0) { throw 'V2 reproduction comparison failed.' }
