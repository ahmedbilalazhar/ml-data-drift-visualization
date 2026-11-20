$ErrorActionPreference = 'Stop'
$pythonExe = 'C:\Users\hp\anaconda3\python.exe'
$rawDir = 'data\external\noaa_gsod_weather_1949_1999_v1\raw'
$reference = 'data\external\noaa_gsod_weather_1949_1999_v1\reference\scikit_multiflow_weather.csv'
$dataDir = 'data\processed\weather_reproduction_v2'
$runDir = 'results\phase3\weather_reproduction_v2'
$reportDir = 'reports\phase3\weather_reproduction_v2'
foreach ($path in @($dataDir, $runDir, $reportDir)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite existing reproduction output: $path" }
}
& $pythonExe scripts\prepare_noaa_weather.py --raw-dir $rawDir --out-dir $dataDir --reference $reference
if ($LASTEXITCODE -ne 0) { throw 'Weather preprocessing failed.' }
& $pythonExe scripts\run_noaa_weather_transfer.py --data "$dataDir\weather_ordered.csv" --manifest "$dataDir\weather_manifest.json" --output $runDir
if ($LASTEXITCODE -ne 0) { throw 'Weather evaluation failed.' }
& $pythonExe scripts\report_noaa_weather.py --run $runDir --data "$dataDir\weather_ordered.csv" --manifest "$dataDir\weather_manifest.json" --output $reportDir
if ($LASTEXITCODE -ne 0) { throw 'Weather report generation failed.' }
tectonic "$reportDir\REPORT.tex"
if ($LASTEXITCODE -ne 0) { throw 'Tectonic report compilation failed.' }
