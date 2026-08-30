[CmdletBinding()]
param(
    [string]$Python = "python",
    [ValidateSet("val", "test-dev")]
    [string]$Split = "val",
    [string]$Checkpoint = "",
    [string]$ValRoot = "/root/autodl-tmp/data/VisDrone2019-MOT-val",
    [string]$TestDevRoot = "/root/autodl-tmp/data/VisDrone2019-MOT-test-dev",
    [string]$ReIDWeights = "",
    [int]$Device = 0,
    [int]$ReIDBatchSize = 128,
    [switch]$Fuse
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if ([string]::IsNullOrWhiteSpace($Checkpoint)) {
    $Checkpoint = Join-Path $RepoRoot "model\best_ckpt.pth.tar"
}
if ([string]::IsNullOrWhiteSpace($ReIDWeights)) {
    $ReIDWeights = Join-Path $RepoRoot "pretrained\veriwild_bot_R50-ibn.pth"
}

$PrepareArgs = @(
    "tools\prepare_visdrone.py",
    "--val-root", $ValRoot,
    "--test-dev-root", $TestDevRoot,
    "--output-root", (Join-Path $RepoRoot "datasets\visdrone"),
    "--splits", $Split
)
& $Python @PrepareArgs
if ($LASTEXITCODE -ne 0) {
    throw "VisDrone data preparation failed with exit code $LASTEXITCODE"
}

$EvalArgs = @(
    "tools\eval_damot.py",
    "-c", $Checkpoint,
    "--split", $Split,
    "--val-root", $ValRoot,
    "--test-dev-root", $TestDevRoot,
    "--reid-weights", $ReIDWeights,
    "--device", $Device,
    "--reid-batch-size", $ReIDBatchSize,
    "--fp16"
)
if ($Fuse) {
    $EvalArgs += "--fuse"
}

& $Python @EvalArgs
if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed with exit code $LASTEXITCODE"
}
