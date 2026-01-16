# env/windows/doctor.ps1
param(
  [ValidateSet("no-rag","plain")]
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

Write-Host "Doctor check complete."
