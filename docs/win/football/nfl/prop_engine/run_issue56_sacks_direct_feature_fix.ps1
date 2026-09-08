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

function Invoke-Python {
    param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Args)
    & $script:PythonExe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE : $($Args -join ' ')"
    }
}

$repo = Find-RepoRoot
Set-Location -LiteralPath $repo

$prop = Join-Path $repo "docs\win\football\nfl\prop_engine"
$pyCmd = Get-Command py -ErrorAction SilentlyContinue
if ($null -ne $pyCmd) {
    $script:PythonExe = $pyCmd.Source
} else {
    $pythonCmd = Get-Command python -ErrorAction Stop
    $script:PythonExe = $pythonCmd.Source
}

Write-Host "CHECK 01: patch confirmed direct sacks matchup denominator bug"
Invoke-Python (Join-Path $prop "apply_issue56_sacks_direct_feature_fix.py")

Write-Host "CHECK 02: validate corrected source contracts"
Invoke-Python (Join-Path $prop "validate_issue56_sacks_direct_feature_fix.py")

Write-Host "CHECK 03: market exclusion preflight"
$marketAudit = Join-Path $prop "scripts\validate\audit_market_exclusion.py"
if (Test-Path -LiteralPath $marketAudit) {
    Invoke-Python $marketAudit "--preflight"
}

Write-Host "CHECK 04: rebuild canonical historical features"
Invoke-Python (Join-Path $prop "scripts\build\build_historical_features.py")

Write-Host "CHECK 05: validate Issue 17 historical feature reconstruction"
Invoke-Python (Join-Path $prop "validate_issue17.py")

$issue19 = Join-Path $prop "validate_issue19.py"
if (Test-Path -LiteralPath $issue19) {
    Write-Host "CHECK 06: validate target feature contracts"
    Invoke-Python $issue19
}

Write-Host "CHECK 07: retrain deterministic direct models through 2024"
Invoke-Python (Join-Path $prop "scripts\train\train_direct_models.py")

$issue24 = Join-Path $prop "validate_issue24.py"
if (Test-Path -LiteralPath $issue24) {
    Write-Host "CHECK 08: validate direct-model training contract"
    Invoke-Python $issue24
}

Write-Host "CHECK 09: rerun 2024-only architecture selection"
Invoke-Python (Join-Path $prop "scripts\train\select_model_architecture.py")

$priorSacksValidator = Join-Path $prop "validate_issue56_sacks_formula_fix.py"
if (Test-Path -LiteralPath $priorSacksValidator) {
    Write-Host "CHECK 10: preserve prior sacks component denominator fix"
    Invoke-Python $priorSacksValidator "--post-selection"
}

Write-Host "CHECK 11: validate rebuilt direct sacks feature/model artifacts"
Invoke-Python (Join-Path $prop "validate_issue56_sacks_direct_feature_fix.py") "--post-rebuild"

Write-Host "CHECK 12: regenerate uncertainty calibration from 2024 validation only"
Invoke-Python (Join-Path $prop "scripts\train\calibrate_uncertainty.py")

$approval = Join-Path $prop "evaluate_production_approval.py"
if (-not (Test-Path -LiteralPath $approval)) {
    throw "Missing required local Issue 56 evaluator: $approval"
}

Write-Host "CHECK 13: evaluate production-path approval"
Invoke-Python $approval
