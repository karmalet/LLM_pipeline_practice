# env/windows/run_deepseekr1_no_rag.ps1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

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
