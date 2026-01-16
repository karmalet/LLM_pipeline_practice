# env/windows/run_deepseekr1_plain_rag.ps1
$ErrorActionPreference = "Stop"

# 1) doctor 점검(DeepSeek Plain-RAG)
& "$PSScriptRoot\doctor.ps1" -mode "deepseek-plain"

# 2) venv 활성화
& "$PSScriptRoot\..\..\.venv\Scripts\Activate.ps1"

# 3) 실행
$scriptPath = Join-Path $PSScriptRoot "..\..\src\3_DeepSeekR1_Plain_RAG.py"

Write-Host ""
Write-Host "=============================="
Write-Host "Running DeepSeek-R1 Plain-RAG (Ollama)"
Write-Host "Script: $scriptPath"
Write-Host "=============================="
Write-Host ""

python $scriptPath
