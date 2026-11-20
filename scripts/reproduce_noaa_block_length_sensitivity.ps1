$ErrorActionPreference = 'Stop'
$Python = 'C:\Users\hp\anaconda3\python.exe'

& $Python scripts\run_noaa_reference_calibration_uncertainty.py `
  --block-length 7 `
  --output results\phase3\noaa_reference_calibration_uncertainty_block7_20261117_v1
if ($LASTEXITCODE -ne 0) { throw '7-day block bootstrap failed' }

& $Python scripts\run_noaa_reference_calibration_uncertainty.py
if ($LASTEXITCODE -ne 0) { throw '30-day block bootstrap failed' }

& $Python scripts\run_noaa_reference_calibration_uncertainty.py `
  --block-length 90 `
  --output results\phase3\noaa_reference_calibration_uncertainty_block90_20261117_v1
if ($LASTEXITCODE -ne 0) { throw '90-day block bootstrap failed' }

& $Python scripts\summarize_noaa_block_length_sensitivity.py
if ($LASTEXITCODE -ne 0) { throw 'Block-length report generation failed' }
