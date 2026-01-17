# env/windows/doctor.ps1
param(
  [ValidateSet("no-rag","plain","deepseek-plain")]
  [string]$mode = "no-rag"
)

$ErrorActionPreference = "Stop"

function Fail($msg) {
  throw $msg
}

# ----------------------------
# 0) 프로젝트 루트 기준 경로
# ----------------------------
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

# ----------------------------
# 1) venv 존재 확인
# ----------------------------
$venvPath = Join-Path $repoRoot ".venv"
if (!(Test-Path $venvPath)) {
  Fail "No venv found. Run env\windows\setup_py311.ps1 first."
}

# ----------------------------
# 2) 기본 파일/폴더 구조 확인
# ----------------------------
$srcDir = Join-Path $repoRoot "src"
$envFile = Join-Path $repoRoot ".env"
$envExample = Join-Path $repoRoot ".env.example"
$promptsDir = Join-Path $repoRoot "prompts"
$dataDir = Join-Path $repoRoot "data"

if (!(Test-Path $srcDir))      { Fail "Missing folder: src/" }
if (!(Test-Path $promptsDir))  { Fail "Missing folder: prompts/" }
if (!(Test-Path $dataDir))     { Fail "Missing folder: data/" }

if (!(Test-Path $envFile)) {
  if (Test-Path $envExample) {
    Fail "Missing .env. Copy .env.example to .env and set OPENAI_API_KEY."
  } else {
    Fail "Missing .env and .env.example. Please add environment template."
  }
}

Write-Host "OK: file structure"

# ----------------------------
# 3) .env 내용(키) 형식 확인
#    - 값 검증(유효키 여부)까지는 하지 않음
# ----------------------------
$envText = Get-Content $envFile -Raw

if ($envText -notmatch "OPENAI_API_KEY=") {
  Fail "OPENAI_API_KEY not set in .env"
}

# (선택) LlamaParse 사용 시
# - 코드에서 LlamaParse를 사용한다면, 키가 없을 때 더 친절한 안내를 주는 것이 좋음
if ($mode -eq "plain") {
  if ($envText -notmatch "LLAMA_CLOUD_API_KEY=") {
    Write-Host "WARN: LLAMA_CLOUD_API_KEY not found in .env (if you use LlamaParse, set it)."
  }
}

Write-Host "OK: env keys present (format check only)"

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

  # 문헌 폴더는 프로젝트마다 이름이 다를 수 있어 “강제 실패” 대신 경고로 처리
  $biblioDir1 = Join-Path $repoRoot "Bibliography"
  $biblioDir2 = Join-Path $repoRoot "bibliography"
  if (!(Test-Path $biblioDir1) -and !(Test-Path $biblioDir2)) {
    Write-Host "WARN: Bibliography/ folder not found. If your Plain-RAG script loads local papers, add Bibliography/."
  }

  # 인덱스/캐시 파일은 '없어도' 첫 실행 시 생성되므로 경고만
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

# doctor.ps1 내부에 추가 (param ValidateSet에 deepseek-plain도 포함)
# param([ValidateSet("no-rag","plain","deepseek-plain")] [string]$mode="no-rag")
if ($mode -eq "deepseek-plain") {

  $repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")

  # venv 존재
  $venvPath = Join-Path $repoRoot ".venv"
  if (!(Test-Path $venvPath)) {
    throw "No venv found. Run: env\windows\setup_py311.ps1 -profile deepseek"
  }

  # 실행 스크립트 존재
  $script = Join-Path $repoRoot "src\3_DeepSeekR1_Plain_RAG.py"
  if (!(Test-Path $script)) { throw "Missing script: src\3_DeepSeekR1_Plain_RAG.py" }

  # 프롬프트 존재
  $prompt = Join-Path $repoRoot "prompts\no-rag-final.yaml"
  if (!(Test-Path $prompt)) { throw "Missing prompt: prompts\no-rag-final.yaml" }

  # Ollama 설치 확인
  $ollama = Get-Command ollama -ErrorAction SilentlyContinue
  if ($null -eq $ollama) {
    throw "Ollama not found. Install Ollama for Windows, then retry."
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
          throw "Failed to pull model: $name. Retry manually: ollama pull $name"
        }
        Write-Host "OK: pulled $name"
      } else {
        Write-Host "WARN: $name not found. Run: ollama pull $name"
      }
    } else {
      Write-Host "OK: $name exists"
    }
  }

  Ensure-OllamaModel "deepseek-r1:14b"
  Ensure-OllamaModel "nomic-embed-text"

  # 결과 폴더 안내
  $resultDir = Join-Path $repoRoot "Result"
  $msglogDir = Join-Path $repoRoot "Msglog"
  if (!(Test-Path $resultDir)) { Write-Host "INFO: Result/ will be created on first run." }
  if (!(Test-Path $msglogDir)) { Write-Host "INFO: Msglog/ will be created on first run." }

  Write-Host "OK: DeepSeek Plain-RAG assets"
}

Write-Host "Doctor check complete."
