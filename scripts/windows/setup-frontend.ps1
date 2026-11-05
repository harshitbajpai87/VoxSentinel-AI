# VoxSentinel frontend setup (Windows, run once from the repo root)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location (Join-Path (Split-Path $root -Parent) "frontend")
npm install
Write-Host "Frontend deps installed."
