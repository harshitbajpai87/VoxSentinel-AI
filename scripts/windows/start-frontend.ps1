# Start the frontend dev server on http://localhost:5173
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location (Join-Path (Split-Path $root -Parent) "frontend")
npm run dev
