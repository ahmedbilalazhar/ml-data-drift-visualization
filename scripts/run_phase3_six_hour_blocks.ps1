param(
    [string]$OutPrefix = "results/phase3/extended_shift_stress_20261120",
    [int]$FirstSeed = 973300,
    [int]$SeedsPerBlock = 1200,
    [int]$BlockCount = 10,
    [int]$WallSeconds = 21600
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root
$python = Join-Path $env:USERPROFILE "anaconda3/python.exe"
$clock = [Diagnostics.Stopwatch]::StartNew()
for ($block = 0; $block -lt $BlockCount; $block++) {
    $remaining = $WallSeconds - [int]$clock.Elapsed.TotalSeconds
    if ($remaining -le 0) { break }
    $seedStart = $FirstSeed + $block * $SeedsPerBlock
    $out = "{0}_block{1:D2}_seeds{2}_{3}" -f $OutPrefix, ($block + 1), $seedStart, ($seedStart + $SeedsPerBlock - 1)
    $count = $SeedsPerBlock
    if ($block -eq $BlockCount - 1 -and $remaining -lt 2400) {
        $count = [math]::Min($SeedsPerBlock, [math]::Max(1, [int](($remaining - 120) / 1.7)))
    }
    & $python scripts/run_phase3_extended_shift_stress.py --workers 2 --seconds $remaining --eval-seeds $count --seed-start $seedStart --out $out
    if ($LASTEXITCODE -ne 0) { throw "Block $($block + 1) stopped. Its protocol lock and partial checkpoint remain in $out." }
    [pscustomobject]@{Block=$block+1; SeedStart=$seedStart; Seeds=$count; ElapsedMinutes=[math]::Round($clock.Elapsed.TotalMinutes,1); Output=$out}
}
