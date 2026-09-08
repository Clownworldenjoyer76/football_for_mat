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

Write-Host "CHECK 01: reconcile Issue 24 deterministic seed contract"
& $python (Join-Path $prop "apply_issue56_issue24_seed_contract_fix.py")
$seedExit = $LASTEXITCODE

if ($seedExit -eq 10) {
    Write-Host "CHECK 02: metadata/source reconciliation requires deterministic direct-model retraining"
    Run-Python -Script (Join-Path $prop "scripts\train\train_direct_models.py")
} elseif ($seedExit -eq 0) {
    Write-Host "CHECK 02: existing direct-model seed metadata already matches the contract"
} else {
    throw "Seed-contract diagnostic failed with exit code $seedExit"
}

Write-Host "CHECK 03: Issue 24 acceptance"
Run-Python -Script (Join-Path $prop "validate_issue24.py")

Write-Host "CHECK 04: rerun 2024-only architecture selection"
Run-Python -Script (Join-Path $prop "scripts\train\select_model_architecture.py")

$priorSacksValidator = Join-Path $prop "validate_issue56_sacks_formula_fix.py"
if (Test-Path -LiteralPath $priorSacksValidator) {
    Write-Host "CHECK 05: preserve corrected top-level sacks component denominator"
    Run-Python -Script $priorSacksValidator -Arguments @("--post-selection")
}

$directFeatureValidator = Join-Path $prop "validate_issue56_sacks_direct_feature_fix.py"
if (-not (Test-Path -LiteralPath $directFeatureValidator)) {
    throw "Missing sacks direct-feature validator: $directFeatureValidator"
}
Write-Host "CHECK 06: validate rebuilt sacks direct feature/model artifacts"
Run-Python -Script $directFeatureValidator -Arguments @("--post-rebuild")

Write-Host "CHECK 07: regenerate 2024-only uncertainty calibration"
Run-Python -Script (Join-Path $prop "scripts\train\calibrate_uncertainty.py")

$approval = Join-Path $prop "evaluate_production_approval.py"
if (-not (Test-Path -LiteralPath $approval)) {
    throw "Missing Issue 56 approval evaluator: $approval"
}
Write-Host "CHECK 08: production-path approval evaluation"
Run-Python -Script $approval
