# Windows 11 + Python 3.11
# 실행: PowerShell에서  .\env\windows\setup_py311.ps1
$ErrorActionPreference = "Stop"

Write-Host "[1/6] Check Python 3.11"
py -3.11 --version

Write-Host "[2/6] Create venv (.venv)"
py -3.11 -m venv .venv

Write-Host "[3/6] Activate venv"
.\.venv\Scripts\Activate.ps1

Write-Host "[4/6] Upgrade pip"
python -m pip install --upgrade pip wheel setuptools

Write-Host "[5/6] Install dependencies (minimal + constraints)"
pip install -c env\constraints.txt -r env\requirements-min-gpt5.txt

Write-Host "[6/6] Quick import test"
python -c "import faiss, tiktoken; import langchain, langchain_openai; print('OK: basic imports')"

Write-Host "DONE. Next: copy .env.example -> .env, then run env\windows\run_gpt5_plain_rag.ps1"
