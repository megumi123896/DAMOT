[CmdletBinding()]
param(
    [string]$YOLOXUrl = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth",
    [string]$VERIWildUrl = "https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veriwild_bot_R50-ibn.pth"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$PretrainedRoot = Join-Path $RepoRoot "pretrained"
New-Item -ItemType Directory -Path $PretrainedRoot -Force | Out-Null

function Get-OfficialWeight {
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    if (Test-Path -LiteralPath $Destination) {
        Write-Host "Already downloaded: $Destination"
        return
    }
    $Partial = "$Destination.part"
    Write-Host "Downloading: $Url"
    & curl.exe -L --fail --retry 5 -C - -o $Partial $Url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed with exit code $LASTEXITCODE. Re-run this script to resume $Partial"
    }
    Move-Item -LiteralPath $Partial -Destination $Destination
    Write-Host "Saved: $Destination"
}

Get-OfficialWeight `
    -Url $YOLOXUrl `
    -Destination (Join-Path $PretrainedRoot "yolox_x.pth")
Get-OfficialWeight `
    -Url $VERIWildUrl `
    -Destination (Join-Path $PretrainedRoot "veriwild_bot_R50-ibn.pth")
