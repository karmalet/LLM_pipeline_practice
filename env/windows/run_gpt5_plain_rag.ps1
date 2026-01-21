Param([switch]$langsmith)

# Always run from repo root
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

# 1) Doctor check (GPT-5 plain)
& "$PSScriptRoot\doctor.ps1" -mode "plain"

Write-Host ""
Write-Host "=============================="
Write-Host "Running GPT-5 Plain-RAG" ($(if ($langsmith) { "(LangSmith Tracking)" } else { "(Local Save)" }))
Write-Host "=============================="

# 2) Activate venv
$activate = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activate) {
  . $activate
} else {
  Write-Host "ERROR: .venv not found. Run: env\windows\setup_py311.ps1 -profile gpt5" -ForegroundColor Red
  exit 1
}

# 3) Select script
$script = if ($langsmith) {
  ".\src\2_GPT5_Plain_RAG_langsmith.py"
} else {
  ".\src\2_GPT5_Plain_RAG.py"
}

if (-not (Test-Path $script)) {
  Write-Host "ERROR: script not found: $script" -ForegroundColor Red
  exit 1
}

Write-Host "Script:" (Resolve-Path $script)
Write-Host "=============================="
Write-Host ""

# 4) Run
python $script

