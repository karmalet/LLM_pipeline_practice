@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ==================================================
REM  LLM Pipeline Practice - One Click Runner
REM  (Windows 11 / Python 3.11)
REM ==================================================

title LLM Pipeline Practice - One Click Runner

REM --- Resolve repo root (this BAT is assumed in oneclick\) ---
pushd "%~dp0" >nul

:MENU
cls
echo ==================================================
echo   LLM Pipeline - One Click Runner
echo   (Windows 11 / Python 3.11)
echo ==================================================
echo.
echo [ Environment Setup ]
echo   1. Install CLOUD environment (GPT-5)
echo   2. Install LOCAL environment (DeepSeek)
echo.
echo [ Run Experiments ]
echo   3. GPT-5 No-RAG
echo   4. GPT-5 Plain-RAG (Local Save)
echo   5. GPT-5 Plain-RAG (LangSmith Tracking)
echo   6. DeepSeek-R1 No-RAG
echo   7. DeepSeek-R1 Plain-RAG
echo.
echo   0. Exit
echo.
set /p choice=Select an option (0-6): 

if "%choice%"=="0" goto END
if "%choice%"=="1" goto SETUP_GPT5
if "%choice%"=="2" goto SETUP_DEEPSEEK
if "%choice%"=="3" goto RUN_GPT5_NO_RAG
if "%choice%"=="4" goto RUN_GPT5_PLAIN_RAG
if "%choice%"=="5" goto RUN_GPT5_PLAIN_RAG_LANGSMITH
if "%choice%"=="6" goto RUN_DEEPSEEK_NO_RAG
if "%choice%"=="7" goto RUN_DEEPSEEK_PLAIN_RAG

echo.
echo Invalid choice. Please try again.
pause
goto MENU

REM ==================================================
REM  Setup
REM ==================================================
:SETUP_GPT5
echo.
echo [*] Setting up GPT-5 environment (py3.11)
powershell -NoProfile -ExecutionPolicy Bypass -File "env\windows\setup_py311.ps1" -profile gpt5
echo.
pause
goto MENU

:SETUP_DEEPSEEK
echo.
echo [*] Setting up DeepSeek environment (py3.11)
powershell -NoProfile -ExecutionPolicy Bypass -File "env\windows\setup_py311.ps1" -profile deepseek
echo.
pause
goto MENU


REM ==================================================
REM  Preflight (auto-fix missing deps)
REM  - If .venv missing OR required imports fail:
REM    run setup script automatically.
REM ==================================================

REM --- Preflight for GPT-5 pipelines ---
:ENSURE_ENV_GPT5
set "_NEED_SETUP=0"

call :CHECK_VENV || set "_NEED_SETUP=1"
if "%_NEED_SETUP%"=="0" (
  call :CHECK_IMPORTS_GPT5 || set "_NEED_SETUP=1"
)

if "%_NEED_SETUP%"=="1" (
  echo "[!] Environment not ready. Running GPT-5 setup once..."
  powershell -NoProfile -ExecutionPolicy Bypass -File "env\windows\setup_py311.ps1" -profile gpt5
)

REM setup 후 최종 검증(1회)
call :CHECK_VENV || (
  echo [X] .venv still missing after setup.
  exit /b 1
)
call :CHECK_IMPORTS_GPT5 || (
  echo [X] Setup finished but imports still fail. Please run setup manually or open an issue.
  exit /b 1
)

exit /b 0

REM --- Preflight for DeepSeek pipelines ---
:ENSURE_ENV_DEEPSEEK
set "_NEED_SETUP=0"

call :CHECK_VENV || set "_NEED_SETUP=1"
if "%_NEED_SETUP%"=="0" (
  call :CHECK_IMPORTS_DEEPSEEK || set "_NEED_SETUP=1"
)

if "%_NEED_SETUP%"=="1" (
  echo "[!] Environment not ready. Running DeepSeek setup once..."
  powershell -NoProfile -ExecutionPolicy Bypass -File "env\windows\setup_py311.ps1" -profile deepseek
)

call :CHECK_VENV || (
  echo [X] .venv still missing after setup.
  exit /b 1
)
call :CHECK_IMPORTS_DEEPSEEK || (
  echo [X] Setup finished but imports still fail. Please run setup manually or open an issue.
  exit /b 1
)

exit /b 0

:CHECK_VENV
if exist ".venv\Scripts\python.exe" (
  exit /b 0
) else (
  exit /b 1
)

REM GPT-5 runs need (at least) dotenv + langchain_openai
:CHECK_IMPORTS_GPT5
".venv\Scripts\python.exe" -c "import dotenv; import langchain; import langchain_openai" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0

REM DeepSeek runs need (at least) dotenv + langchain_ollama + ollama client
:CHECK_IMPORTS_DEEPSEEK
".venv\Scripts\python.exe" -c "import dotenv; import langchain; import langchain_ollama; import ollama" >nul 2>&1
if errorlevel 1 exit /b 1
exit /b 0


REM ==================================================
REM  Run Experiments
REM ==================================================
:RUN_GPT5_NO_RAG
echo.
echo [*] Running GPT-5 No-RAG experiment
call :ENSURE_ENV_GPT5 || (pause & goto MENU)
powershell -NoProfile -ExecutionPolicy Bypass -File "env\windows\run_gpt5_no_rag.ps1"
echo.
pause
goto MENU

:RUN_GPT5_PLAIN_RAG
echo.
echo [*] Running GPT-5 Plain-RAG experiment
call :ENSURE_ENV_GPT5 || (pause & goto MENU)
powershell -NoProfile -ExecutionPolicy Bypass -File "env\windows\run_gpt5_plain_rag.ps1"
echo.
pause
goto MENU

:RUN_GPT5_PLAIN_RAG_LANGSMITH
echo.
echo [*] Running GPT-5 Plain-RAG (LangSmith Tracking) experiment
call :ENSURE_ENV_GPT5 || (pause & goto MENU)
powershell -NoProfile -ExecutionPolicy Bypass -File "env\windows\run_gpt5_plain_rag.ps1" -langsmith
echo.
pause
goto MENU

:RUN_DEEPSEEK_NO_RAG
echo.
echo [*] Running DeepSeek-R1 No-RAG experiment
call :ENSURE_ENV_DEEPSEEK || (pause & goto MENU)
powershell -NoProfile -ExecutionPolicy Bypass -File "env\windows\run_deepseekr1_no_rag.ps1"
echo.
pause
goto MENU

:RUN_DEEPSEEK_PLAIN_RAG
echo.
echo [*] Running DeepSeek-R1 Plain-RAG experiment
call :ENSURE_ENV_DEEPSEEK || (pause & goto MENU)
powershell -NoProfile -ExecutionPolicy Bypass -File "env\windows\run_deepseekr1_plain_rag.ps1"
echo.
pause
goto MENU


:END
popd >nul
endlocal
exit /b 0
