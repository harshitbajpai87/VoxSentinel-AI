# VoxSentinel backend setup (Windows, run once from the repo root)
# Creates the virtualenv and installs pinned requirements.
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$repo = Split-Path $root -Parent
Set-Location (Join-Path $repo "backend")

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install --upgrade pip
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt -r requirements-dev.txt

Write-Host ""
Write-Host "Backend deps installed. Optional (enables M4A/AAC/WebM/Opus uploads):"
Write-Host "  winget install Gyan.FFmpeg"
Write-Host ""
Write-Host "Model artifacts (not committed):"
Write-Host "  - models/detector.joblib      -> train via scripts/train_detector.py (docs/DATASET_GUIDE.md)"
Write-Host "  - models/speaker_ecapa/      -> downloaded automatically on first /speaker-verify"
Write-Host "  - models/speaker_thresholds.json -> regenerate via scripts/evaluate_speaker.py"
Write-Host "Without detector.joblib the API stays honest: /health model_ready=false, /analyze -> 503."
