$ErrorActionPreference = "Stop"

$prop = Split-Path -Parent $MyInvocation.MyCommand.Path
$cursor = $prop
while ($true) {
    if (Test-Path -LiteralPath (Join-Path $cursor ".git") -PathType Container) {
        $repo = $cursor
        break
    }
    $parent = Split-Path -Parent $cursor
    if ([string]::IsNullOrWhiteSpace($parent) -or $parent -eq $cursor) {
        throw "Could not locate repository root containing .git"
    }
    $cursor = $parent
}

$pyCmd = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $pyCmd = "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $pyCmd = "python"
} else {
    throw "Python launcher not found (py/python)"
}

Set-Location -LiteralPath $repo

$base = "docs/win/football/nfl/prop_engine"

& $pyCmd "$base/apply_issue56_sacks_formula_fix.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pyCmd "$base/validate_issue56_sacks_formula_fix.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (Test-Path -LiteralPath "$base/validate_issue56_calibration_point_fix.py") {
    & $pyCmd "$base/validate_issue56_calibration_point_fix.py"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $pyCmd "$base/scripts/train/select_model_architecture.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pyCmd "$base/validate_issue56_sacks_formula_fix.py" --post-selection
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $pyCmd "$base/scripts/train/calibrate_uncertainty.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not (Test-Path -LiteralPath "$base/evaluate_production_approval.py")) {
    throw "Missing $base/evaluate_production_approval.py"
}

& $pyCmd "$base/evaluate_production_approval.py"
exit $LASTEXITCODE
