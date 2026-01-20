# One-click 실행(Windows 11) 안내

본 저장소는 컴퓨터 비전공자(언어학 연구자)도 실험을 재현할 수 있도록,
명령어 입력을 최소화한 “원클릭 실행 메뉴(배치 파일)”를 제공한다.

---

## 1. 준비물(최소 요건)

- Windows 11
- Python 3.11 (설치 권장)
- PowerShell 사용 가능(Windows 기본 탑재)
- (DeepSeek 실험 시) Ollama 설치 권장

> 실험에 필요한 Python 패키지는 메뉴에서 자동 설치된다.

---

## 2. 실행 순서(가장 쉬운 루트)

1) 저장소 다운로드(ZIP) 후 압축 해제  
2) 저장소 루트에 `.env.example`을 복사하여 `.env` 생성  
3) `.env` 파일에 필요한 키 값을 입력  
4) `oneclick/00_MENU.bat` 더블클릭  
5) 메뉴에서 번호 선택

---

## 3. 메뉴 기능 설명

### (A) 환경 설치/갱신
- [1] GPT-5 Plain-RAG 환경 설치/갱신
- [2] DeepSeek-R1 Plain-RAG 환경 설치/갱신

> 처음 실행하는 PC라면 반드시 (A)를 먼저 수행한다.

### (B) 실험 실행
- [3] GPT-5 No-RAG 실행
- [4] GPT-5 Plain-RAG 실행
- [5] DeepSeek-R1 No-RAG 실행
- [6] DeepSeek-R1 Plain-RAG 실행

---

## 4. DeepSeek 실험(Ollama) 관련 안내

DeepSeek-R1 실험은 로컬 실행 기반이며, 다음이 필요할 수 있다.

- Ollama 설치
- 모델 다운로드:
  - `deepseek-r1:14b`
  - `nomic-embed-text` (Plain-RAG에서 임베딩용)

본 저장소는 실행 전 점검(doctor) 단계에서
모델이 없으면 자동으로 경고 메시지를 표시한다.

---

## 5. 자주 발생하는 문제(FAQ)

### Q1. “ExecutionPolicy” 때문에 ps1 실행이 막혀요.
- 메뉴 배치 파일은 기본적으로 `-ExecutionPolicy Bypass`로 실행한다.
- 그래도 막히면 Windows 보안 정책(회사 PC 등) 때문일 수 있다.

### Q2. `.env`를 만들었는데 키가 없다고 나와요.
- `.env.example` → `.env`로 “복사”가 되었는지 확인한다.
- OpenAI / LangSmith 키 값이 공백이 아닌지 확인한다.

### Q3. DeepSeek가 “모델이 없다”고 나와요.
- doctor 메시지대로 `ollama pull ...`을 진행한다.
- 최초 1회 다운로드 시간이 걸릴 수 있다.

