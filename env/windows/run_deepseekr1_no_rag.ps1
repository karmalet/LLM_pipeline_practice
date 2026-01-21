# env/windows/run_deepseekr1_no_rag.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# --- ensure we are in repo root ---
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

# --- activate venv if exists ---
$activate = ".\.venv\Scripts\Activate.ps1"
if (Test-Path $activate) {
  . $activate
} else {
  Write-Host "ERROR: .venv not found. Run: env\windows\setup_py311.ps1 -profile deepseek" -ForegroundColor Red
  exit 1
}

# (선택) 디버그: 지금 어떤 python을 쓰는지 확인
Write-Host "INFO: python =" (Get-Command python).Source

# 0) doctor 체크 (deepseek no-rag 모드)
& "$PSScriptRoot\doctor.ps1" -mode "deepseek-no-rag"

# 1) 실행
$ScriptPath = Join-Path $PSScriptRoot "..\..\src\3_DeepSeekR1_No_RAG.py"
$ScriptPath = (Resolve-Path $ScriptPath).Path

Write-Host ""
Write-Host "=============================="
Write-Host "Running DeepSeek-R1 No-RAG (Ollama)"
Write-Host "Script: $ScriptPath"
Write-Host "=============================="
Write-Host ""

python $ScriptPath
