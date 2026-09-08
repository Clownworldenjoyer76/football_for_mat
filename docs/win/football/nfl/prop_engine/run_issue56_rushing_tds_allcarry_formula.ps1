$ErrorActionPreference = "Stop"

function Find-RepoRoot {
    $p = (Get-Location).Path
    while ($true) {
        if (Test-Path -LiteralPath (Join-Path $p ".git")) { return $p }
        $parent = Split-Path -Parent $p
        if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $p) {
            throw "Could not locate repository root (.git)."
        }
        $p = $parent
    }
}

$repo = Find-RepoRoot
Set-Location -LiteralPath $repo
$prop = Join-Path $repo "docs\win\football\nfl\prop_engine"

$pyCmd = Get-Command py -ErrorAction SilentlyContinue
if ($null -ne $pyCmd) {
    $python = $pyCmd.Source
} else {
    $python = (Get-Command python -ErrorAction Stop).Source
}

$script = Join-Path $prop "diagnose_issue56_rushing_tds_allcarry_formula.py"

Write-Host "PREFLIGHT: verify all-carry formula inputs before evaluation"
& $python $script --preflight-only
if ($LASTEXITCODE -ne 0) {
    throw "Preflight failed with exit code $LASTEXITCODE. No diagnostic evaluation was started."
}

Write-Host "DIAGNOSTIC: evaluate target-aligned all-carry formula candidates"
& $python $script
if ($LASTEXITCODE -ne 0) {
    throw "All-carry rushing TD diagnostic failed with exit code $LASTEXITCODE"
}
