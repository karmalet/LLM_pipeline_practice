@echo off
chcp 65001 > nul
setlocal

echo ==================================================
echo   LLM Pipeline Practice - One Click Runner
echo   (Windows 11 / Python 3.11)
echo ==================================================
echo.
echo [ Environment Setup ]
echo   1. GPT-5 Plain-RAG environment
echo   2. DeepSeek-R1 Plain-RAG environment
echo.
echo [ Run Experiments ]
echo   3. GPT-5 No-RAG
echo   4. GPT-5 Plain-RAG
echo   5. DeepSeek-R1 No-RAG
echo   6. DeepSeek-R1 Plain-RAG
echo.
echo   0. Exit
echo.

set /p choice=Select an option (0-6): 

if "%choice%"=="1" goto SETUP_GPT5
if "%choice%"=="2" goto SETUP_DEEPSEEK
if "%choice%"=="3" goto RUN_GPT5_NORAG
if "%choice%"=="4" goto RUN_GPT5_PLAINRAG
if "%choice%"=="5" goto RUN_DEEPSEEK_NORAG
if "%choice%"=="6" goto RUN_DEEPSEEK_PLAINRAG
if "%choice%"=="0" goto END

echo.
echo Invalid selection. Please try again.
pause
goto END

:SETUP_GPT5
echo.
echo [*] Setting up GPT-5 Plain-RAG environment...
powershell -ExecutionPolicy Bypass -File "..\env\windows\setup_py311.ps1" -profile gpt5
pause
goto END

:SETUP_DEEPSEEK
echo.
echo [*] Setting up DeepSeek-R1 Plain-RAG environment...
powershell -ExecutionPolicy Bypass -File "..\env\windows\setup_py311.ps1" -profile deepseek
pause
goto END

:RUN_GPT5_NORAG
echo.
echo [*] Running GPT-5 No-RAG experiment...
powershell -ExecutionPolicy Bypass -File "..\env\windows\run_gpt5_no_rag.ps1"
pause
goto END

:RUN_GPT5_PLAINRAG
echo.
echo [*] Running GPT-5 Plain-RAG experiment...
powershell -ExecutionPolicy Bypass -File "..\env\windows\run_gpt5_plain_rag.ps1"
pause
goto END

:RUN_DEEPSEEK_NORAG
echo.
echo [*] Running DeepSeek-R1 No-RAG experiment...
powershell -ExecutionPolicy Bypass -File "..\env\windows\run_deepseek_no_rag.ps1"
pause
goto END

:RUN_DEEPSEEK_PLAINRAG
echo.
echo [*] Running DeepSeek-R1 Plain-RAG experiment...
powershell -ExecutionPolicy Bypass -File "..\env\windows\run_deepseek_plain_rag.ps1"
pause
goto END

:END
echo.
echo Done.
endlocal
