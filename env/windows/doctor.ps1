$ErrorActionPreference = "Stop"

if (!(Test-Path ".\.venv")) { throw "No venv found. Run env\windows\setup_py311.ps1 first." }
if (!(Test-Path ".\.env")) { throw "No .env found. Copy .env.example to .env and fill keys." }
if (!(Test-Path ".\prompts\no-rag-final.yaml")) { throw "Missing prompts\rag.yaml" }
if (!(Test-Path ".\src\1_GPT5_No_RAG.py")) { throw "Missing src\1_GPT5_Plain_RAG.py" }

Write-Host "OK: file structure"

# 키 존재 여부(문자열 길이만 체크)
$envText = Get-Content .\.env -Raw
if ($envText -notmatch "OPENAI_API_KEY=") { throw "OPENAI_API_KEY not set in .env" }

Write-Host "OK: env keys present (format check only)"
Write-Host "Doctor check complete."
