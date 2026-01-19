# env/windows/doctor.ps1
param(
  [ValidateSet("no-rag","plain","deepseek-no-rag","deepseek-plain")]
  [string]$mode = "no-rag"
)

$ErrorActionPreference = "Stop"

function Fail($msg) { throw $msg }

# ----------------------------
# 0) 프로젝트 루트 기준 경로
# ----------------------------
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
$isDeepSeek = ($mode -like "deepseek-*")

# ----------------------------
# 1) venv 존재 확인
# ----------------------------
$venvPath = Join-Path $repoRoot ".venv"
if (!(Test-Path $venvPath)) {
  if ($isDeepSeek) {
    Fail "No venv found. Run: env\windows\setup_py311.ps1 (DeepSeek profile if you use it)."
  } else {
    Fail "No venv found. Run env\windows\setup_py311.ps1 first."
  }
}

# ----------------------------
# 2) 기본 파일/폴더 구조 확인
# ----------------------------
$srcDir      = Join-Path $repoRoot "src"
$envFile     = Join-Path $repoRoot ".env"
$envExample  = Join-Path $repoRoot ".env.example"
$promptsDir  = Join-Path $repoRoot "prompts"
$dataDir     = Join-Path $repoRoot "data"

if (!(Test-Path $srcDir))      { Fail "Missing folder: src/" }
if (!(Test-Path $promptsDir))  { Fail "Missing folder: prompts/" }
if (!(Test-Path $dataDir))     { Fail "Missing folder: data/" }

# .env는 GPT-5 모드에서는 필수, DeepSeek 모드에서는 선택(있으면 로드, 없어도 실행 가능)
if (!(Test-Path $envFile)) {
  if ($isDeepSeek) {
    if (Test-Path $envExample) {
      Write-Host "WARN: .env not found. (DeepSeek runs can still work) If needed, copy .env.example -> .env."
    } else {
      Write-Host "WARN: .env and .env.example not found. (DeepSeek runs can still work) Add template if you want."
    }
  } else {
    if (Test-Path $envExample) {
      Fail "Missing .env. Copy .env.example to .env and set OPENAI_API_KEY."
    } else {
      Fail "Missing .env and .env.example. Please add environment template."
    }
  }
}

Write-Host "OK: file structure"

# ----------------------------
# 3) .env 내용(키) 형식 확인
#    - DeepSeek 모드에서는 OPENAI_API_KEY 강제하지 않음
# ----------------------------
if (Test-Path $envFile) {
  $envText = Get-Content $envFile -Raw

  if (!$isDeepSeek) {
    if ($envText -notmatch "OPENAI_API_KEY=") {
      Fail "OPENAI_API_KEY not set in .env"
    }

    # (선택) LlamaParse 사용 시
    if ($mode -eq "plain") {
      if ($envText -notmatch "LLAMA_CLOUD_API_KEY=") {
        Write-Host "WARN: LLAMA_CLOUD_API_KEY not found in .env (if you use LlamaParse, set it)."
      }
    }

    Write-Host "OK: env keys present (format check only)"
  } else {
    # DeepSeek 모드: .env가 있으면 읽기만(강제 키 없음)
    Write-Host "OK: env file present (DeepSeek mode: key check skipped)"
  }
} else {
  if ($isDeepSeek) {
    Write-Host "OK: env file not present (DeepSeek mode: allowed)"
  }
}

# ----------------------------
# 4) 모드별 실행 파일 및 필수 리소스 확인
# ----------------------------
if ($mode -eq "no-rag") {
  $noRagScript = Join-Path $repoRoot "src\1_GPT5_No_RAG.py"
  $noRagPrompt = Join-Path $repoRoot "prompts\no-rag-final.yaml"

  if (!(Test-Path $noRagScript)) { Fail "Missing script: src\1_GPT5_No_RAG.py" }
  if (!(Test-Path $noRagPrompt)) { Fail "Missing prompt: prompts\no-rag-final.yaml" }

  Write-Host "OK: No-RAG assets"
}

if ($mode -eq "plain") {
  $plainScript = Join-Path $repoRoot "src\2_GPT5_Plain_RAG.py"
  $plainPrompt = Join-Path $repoRoot "prompts\rag-final.yaml"

  if (!(Test-Path $plainScript)) { Fail "Missing script: src\2_GPT5_Plain_RAG.py" }
  if (!(Test-Path $plainPrompt)) { Fail "Missing prompt: prompts\rag-final.yaml" }

  $biblioDir1 = Join-Path $repoRoot "Bibliography"
  $biblioDir2 = Join-Path $repoRoot "bibliography"
  if (!(Test-Path $biblioDir1) -and !(Test-Path $biblioDir2)) {
    Write-Host "WARN: Bibliography/ folder not found. If your Plain-RAG script loads local papers, add Bibliography/."
  }

  $picklePath = Join-Path $repoRoot "llama_parsed_docs.pickle"
  if (!(Test-Path $picklePath)) {
    Write-Host "INFO: llama_parsed_docs.pickle not found (will be created on first run)."
  }

  $cacheDir = Join-Path $repoRoot "cache_for_rag_openAIembeddings"
  if (!(Test-Path $cacheDir)) {
    Write-Host "INFO: cache_for_rag_openAIembeddings/ not found (will be created on first run)."
  }

  Write-Host "OK: Plain-RAG assets"
}

# ----------------------------
# 5) DeepSeek 모드 공통 체크 (Ollama + 모델 pull)
# ----------------------------
if ($isDeepSeek) {

  # (A) 실행 스크립트 존재
  if ($mode -eq "deepseek-plain") {
    $script = Join-Path $repoRoot "src\3_DeepSeekR1_Plain_RAG.py"
  } else {
    $script = Join-Path $repoRoot "src\3_DeepSeekR1_No_RAG.py"
  }
  if (!(Test-Path $script)) { Fail "Missing script: $($script.Replace($repoRoot.Path + '\','src\'))" }

  # (B) 프롬프트 존재 (DeepSeek No-RAG/Plain-RAG 모두 사용)
  $prompt = Join-Path $repoRoot "prompts\no-rag-final.yaml"
  if (!(Test-Path $prompt)) { Fail "Missing prompt: prompts\no-rag-final.yaml" }

  # (C) Ollama 설치 확인
  $ollama = Get-Command ollama -ErrorAction SilentlyContinue
  if ($null -eq $ollama) {
    Fail "Ollama not found. Install Ollama for Windows, then retry."
  }
  Write-Host "OK: ollama found"

  # --- 자동 pull 옵션 (기본 ON) ---
  # 원치 않으면 PowerShell에서: $env:AUTO_PULL_OLLAMA="0"
  $autoPull = $env:AUTO_PULL_OLLAMA
  if ([string]::IsNullOrWhiteSpace($autoPull)) { $autoPull = "1" }

  function Ensure-OllamaModel([string]$name) {
    $list = & ollama list 2>$null
    if ($list -notmatch [regex]::Escape($name)) {
      if ($autoPull -eq "1") {
        Write-Host "INFO: $name not found. Pulling now..."
        & ollama pull $name
        if ($LASTEXITCODE -ne 0) {
          Fail "Failed to pull model: $name. Retry manually: ollama pull $name"
        }
        Write-Host "OK: pulled $name"
      } else {
        Write-Host "WARN: $name not found. Run:  ollama pull $name"
      }
    } else {
      Write-Host "OK: $name exists"
    }
  }

  # DeepSeek 본체 모델은 공통 필수
  Ensure-OllamaModel "deepseek-r1:14b"

  # nomic-embed-text는 Plain-RAG에서만 필요
  if ($mode -eq "deepseek-plain") {
    Ensure-OllamaModel "nomic-embed-text"
  }

  # 결과 폴더 안내
  $resultDir = Join-Path $repoRoot "Result"
  $msglogDir = Join-Path $repoRoot "Msglog"
  if (!(Test-Path $resultDir)) { Write-Host "INFO: Result/ will be created on first run." }
  if (!(Test-Path $msglogDir)) { Write-Host "INFO: Msglog/ will be created on first run." }

  if ($mode -eq "deepseek-plain") {
    Write-Host "OK: DeepSeek Plain-RAG assets"
  } else {
    Write-Host "OK: DeepSeek No-RAG assets"
  }
}

Write-Host "Doctor check complete."
