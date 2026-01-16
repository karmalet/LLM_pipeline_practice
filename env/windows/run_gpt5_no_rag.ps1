$ErrorActionPreference = "Stop"

.\env\windows\doctor.ps1
.\.venv\Scripts\Activate.ps1

# src 경로에서 실행(상대경로 깨짐 방지)
python .\src\1_GPT5_No_RAG.py
