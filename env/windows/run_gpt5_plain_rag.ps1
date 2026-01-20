# env/windows/run_gpt5_plain_rag.ps1
$ErrorActionPreference = "Stop"

# 1) 사전 점검 (Plain-RAG 모드)
& "$PSScriptRoot\doctor.ps1" -mode "plain"

# 2) venv 활성화
& "$PSScriptRoot\..\..\.venv\Scripts\Activate.ps1"

# 3) 실행
# - 'py' 런처가 없는 PC에서도 동작하도록 venv의 python을 사용
$scriptPath = Join-Path $PSScriptRoot "..\..\src\2_GPT5_Plain_RAG_localsave.py"

Write-Host ""
Write-Host "=============================="
Write-Host "Running Plain-RAG (GPT-5)"
Write-Host "Script: $scriptPath"
Write-Host "=============================="
Write-Host ""

python $scriptPath
