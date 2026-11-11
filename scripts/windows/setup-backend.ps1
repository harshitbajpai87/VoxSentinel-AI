# VoxSentinel backend setup (Windows, run once from the repo root)
# Creates the virtualenv and installs pinned requirements.
param([switch]$FullAudio)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$repo = Split-Path $root -Parent
Set-Location (Join-Path $repo "backend")

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed" }
& ".venv\Scripts\python.exe" -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw "Backend dependency installation failed" }
if ($FullAudio) {
    & ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "Full audio dependency installation failed" }
} else {
    Write-Host "Core detector installed. Add -FullAudio to install ECAPA/Whisper (large download)."
}

Write-Host ""
Write-Host "Backend deps installed. Optional (enables M4A/AAC/WebM/Opus uploads):"
Write-Host "  winget install Gyan.FFmpeg"
Write-Host ""
Write-Host "Model artifacts (not committed):"
Write-Host "  - models/detector.joblib      -> train via scripts/train_detector.py (docs/DATASET_GUIDE.md)"
Write-Host "  - models/speaker_ecapa/      -> downloaded automatically on first /speaker-verify"
Write-Host "  - models/speaker_thresholds.json -> regenerate via scripts/evaluate_speaker.py"
Write-Host "Without detector.joblib the API stays honest: /health model_ready=false, /analyze -> 503."
