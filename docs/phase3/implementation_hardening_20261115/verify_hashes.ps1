$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..\..')).Path
$manifest = Join-Path $PSScriptRoot 'HASHES.sha256'
$entries = Get-Content -LiteralPath $manifest | Where-Object { $_ -match '^[0-9a-fA-F]{64}\s{2,}.+$' }
$failures = [System.Collections.Generic.List[string]]::new()
foreach ($entry in $entries) {
    $parts = $entry -split '\s{2,}', 2
    $expected = $parts[0].ToLowerInvariant()
    $relative = $parts[1] -replace '/', '\'
    $path = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        $failures.Add("MISSING $relative")
        continue
    }
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $expected) { $failures.Add("MISMATCH $relative expected=$expected actual=$actual") }
}
if ($failures.Count) {
    $failures | ForEach-Object { Write-Error $_ }
    throw "Hash verification failed ($($failures.Count) of $($entries.Count))."
}
Write-Output "Verified $($entries.Count) SHA-256 entries."
