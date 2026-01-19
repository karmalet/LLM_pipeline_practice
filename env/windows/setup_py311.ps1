# Windows 11 + Python 3.11
# 실행: PowerShell에서  .\env\windows\setup_py311.ps1
param(
    [ValidateSet("gpt5","deepseek")]
    [string]$profile = "gpt5"
)
$ErrorActionPreference = "Stop"

Write-Host "[1/6] Check Python 3.11"
$pyver = python --version 2>&1
if ($pyver -notmatch "Python 3\.11") {
    Write-Host "ERROR: Python 3.11 is required."
    Write-Host "Current version: $pyver"
    Write-Host "Please install Python 3.11 and ensure it is added to PATH."
    exit 1
}
Write-Host "OK: $pyver"
Write-Host ""

Write-Host "[2/6] Create venv (.venv)"
if (!(Test-Path ".venv")) {
    python -m venv .venv
    Write-Host "OK: .venv created"
} else {
    Write-Host "OK: .venv already exists"
}
Write-Host ""

Write-Host "[3/6] Activate venv"
.\.venv\Scripts\Activate.ps1
Write-Host "OK: venv activated"
Write-Host ""

Write-Host "[4/6] Upgrade pip / wheel / setuptools"
python -m pip install --upgrade pip wheel setuptools
Write-Host "OK: pip toolchain upgraded"
Write-Host ""

Write-Host "[5/6] Install dependencies (profile = $profile)"

if ($profile -eq "gpt5") {
    $req = "env\requirements-min-gpt5.txt"
}
elseif ($profile -eq "deepseek") {
    $req = ".\env\requirements-min-deepseek.txt"
}

if (Test-Path ".\env\constraints.txt") {
    pip install -c env\constraints.txt -r $req
} else {
    pip install -r $req
}
Write-Host "OK: dependencies installed"
Write-Host ""

Write-Host "[6/6] Quick import test"

if ($profile -eq "gpt5") {
    python -c "import faiss, tiktoken; import langchain, langchain_openai; print('OK: GPT-5 imports')"
}
elseif ($profile -eq "deepseek") {
    python -c "import faiss; import langchain, langchain_ollama; print('OK: DeepSeek imports')"
}

Write-Host ""
Write-Host "DONE."
Write-Host "Next steps:"
if ($profile -eq "gpt5") {
    Write-Host "  1) copy .env.example -> .env"
    Write-Host "  2) set OPENAI_API_KEY"
    Write-Host "  3) run: env\windows\run_gpt5_plain_rag.ps1"
} else {
    Write-Host "  1) ensure Ollama is installed. install: https://www.ollama.com/ -> download"
	Write-Host "  2-1) run baseline model: env\windows\run_deepseekr1_No_rag.ps1"
    Write-Host "  2-2) run RAG model: env\windows\run_deepseekr1_plain_rag.ps1"
}

