[CmdletBinding()]
param(
    [string]$Python = "python",
    [string]$TrainRoot = "/root/autodl-tmp/data/VisDrone2019-MOT-train",
    [string]$ValRoot = "/root/autodl-tmp/data/VisDrone2019-MOT-val",
    [string]$TestDevRoot = "/root/autodl-tmp/data/VisDrone2019-MOT-test-dev",
    [string]$DetectorPretrained = "",
    [int]$BatchSize = 1,
    [int]$Devices = 1,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot

if ([string]::IsNullOrWhiteSpace($DetectorPretrained)) {
    $DetectorPretrained = Join-Path $RepoRoot "pretrained\yolox_x.pth"
}

& $Python "tools\prepare_visdrone.py" `
    --train-root $TrainRoot `
    --val-root $ValRoot `
    --test-dev-root $TestDevRoot `
    --output-root (Join-Path $RepoRoot "datasets\visdrone") `
    --splits train val test-dev
if ($LASTEXITCODE -ne 0) {
    throw "VisDrone data preparation failed with exit code $LASTEXITCODE"
}

$TrainArgs = @(
    "tools\train.py",
    "-f", "exps\example\mot\damot_visdrone.py",
    "-d", $Devices,
    "-b", $BatchSize,
    "--fp16"
)
if ($Resume) {
    $TrainArgs += "--resume"
} else {
    if (-not (Test-Path -LiteralPath $DetectorPretrained)) {
        throw "Detector pretrained checkpoint not found: $DetectorPretrained"
    }
    $TrainArgs += @("-c", $DetectorPretrained)
}

& $Python @TrainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Training failed with exit code $LASTEXITCODE"
}
