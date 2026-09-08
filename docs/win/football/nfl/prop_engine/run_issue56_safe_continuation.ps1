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

function Run-Python {
    param(
        [Parameter(Mandatory=$true)][string]$Script,
        [string[]]$Arguments = @()
    )
    & $python $Script @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE : $Script $($Arguments -join ' ')"
    }
}

Write-Host "PREFLIGHT 01: normalize and verify complete Issue 24 source/seed contract"
Run-Python -Script (Join-Path $prop "fix_issue56_issue24_static_contract.py")

Write-Host "PREFLIGHT 02: run full Issue 24 validator BEFORE any expensive work"
Run-Python -Script (Join-Path $prop "validate_issue24.py")

$directFeatureValidator = Join-Path $prop "validate_issue56_sacks_direct_feature_fix.py"
if (-not (Test-Path -LiteralPath $directFeatureValidator)) {
    throw "Missing sacks direct-feature validator: $directFeatureValidator"
}

Write-Host "PREFLIGHT 03: verify rebuilt sacks direct feature/model artifacts"
Run-Python -Script $directFeatureValidator -Arguments @("--post-rebuild")

$priorSacksValidator = Join-Path $prop "validate_issue56_sacks_formula_fix.py"
if (Test-Path -LiteralPath $priorSacksValidator) {
    Write-Host "PREFLIGHT 04: verify corrected top-level sacks denominator source"
    Run-Python -Script $priorSacksValidator
}

Write-Host "CHECK 01: rerun 2024-only architecture selection"
Run-Python -Script (Join-Path $prop "scripts\train\select_model_architecture.py")

if (Test-Path -LiteralPath $priorSacksValidator) {
    Write-Host "CHECK 02: verify regenerated sacks selected-model dependency"
    Run-Python -Script $priorSacksValidator -Arguments @("--post-selection")
}

Write-Host "CHECK 03: verify direct feature/model contract after selection"
Run-Python -Script $directFeatureValidator -Arguments @("--post-rebuild")

Write-Host "CHECK 04: regenerate uncertainty calibration using 2024 validation only"
Run-Python -Script (Join-Path $prop "scripts\train\calibrate_uncertainty.py")

$approval = Join-Path $prop "evaluate_production_approval.py"
if (-not (Test-Path -LiteralPath $approval)) {
    throw "Missing Issue 56 approval evaluator: $approval"
}

Write-Host "CHECK 05: production-path approval evaluation"
Run-Python -Script $approval
