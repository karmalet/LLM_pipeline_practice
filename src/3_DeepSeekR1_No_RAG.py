# DeepSeekR1_No_RAG.py
from datetime import datetime
import os, re, json, pickle
from dotenv import load_dotenv

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnableConfig

from langchain_ollama import ChatOllama

# -----------------------------
# 0) 실험 설정
# -----------------------------
MODEL = "deepseek-r1:14b"
INPUT = "data/Art_Nat_20250509.json"
DATA_name = re.findall(r"/(.+)\.json", INPUT)[0]

# 로그/결과 저장
MSGLOG_PATH = "Msglog/DeepSeekR1-NoRAG-decision_msgs.jsonl"
RAW_DIR = "Msglog/raw_deepseek_norag"
RESULT_TXT = f"Result/DeepSeekR1-NoRAG-{DATA_name}.txt"
PKL_OUT = "DeepSeekR1-NoRAG-preds_golds.pkl"


# -----------------------------
# 1) 유틸: LangSmith 완전 비활성화
# -----------------------------
def _disable_langsmith_env():
    for k in [
        "LANGSMITH_API_KEY",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_PROJECT",
        "LANGCHAIN_ENDPOINT",
        "LANGSMITH_TRACING",
    ]:
        os.environ.pop(k, None)


# -----------------------------
# 2) 유틸: JSON 파싱(DeepSeek 출력에서 label/reason만 추출)
# -----------------------------
def parse_json_response(text: str):
    """
    DeepSeek 출력에서 JSON 객체만 파싱.
    - ```json ... ``` 처리
    - 여러 JSON이 있을 경우 마지막 객체 우선
    """
    try:
        code_block_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
        else:
            json_matches = re.findall(r"\{.*?\}", text, re.DOTALL)
            if not json_matches:
                raise ValueError("No JSON object found.")
            json_str = json_matches[-1]

        data = json.loads(json_str.strip())
        label = data.get("label", "R")
        reason = data.get("reason", "")
        return label, reason
    except Exception as e:
        return "R", f"JSON parse error: {e}"


# -----------------------------
# 3) 유틸: NLI 질문 포맷
# -----------------------------
def format_nli_prompt(predicate, text, hypothesis, options):
    return f"""
请根据以下输入内容判断：
- 前提（text）: {text}
- 假设（hypothesis）: {hypothesis}
- 判断依据: {predicate}
- 选项（option）: {options}
""".strip()


# -----------------------------
# 4) 유틸: JSONL 로그 누적 저장
# -----------------------------
def _json_default(o):
    try:
        return dict(o)
    except Exception:
        return str(o)

def dump_decision_msg(path, d_id, question, msg, final_label, reasoning_txt, cfg=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "d_id": d_id,
        "question": question,
        "final_label": final_label,
        "reasoning_txt": reasoning_txt,
        "decision_msg": {
            "type": type(msg).__name__,
            "content": getattr(msg, "content", None),
            "response_metadata": getattr(msg, "response_metadata", None),
            "additional_kwargs": getattr(msg, "additional_kwargs", None),
        },
        "run_config": cfg or {},
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")


# -----------------------------
# 5) 유틸: 파싱 실패 시 1회 재시도
# -----------------------------
def invoke_with_retry(chain, question: str, cfg: RunnableConfig):
    # 1차 호출
    msg = chain.invoke(question, config=cfg)
    raw = getattr(msg, "content", "") or str(msg)
    label, reason = parse_json_response(raw)

    # 파싱 실패(= JSON parse error)면 1회 재요청
    if (label == "R") and ("JSON parse error" in (reason or "")):
        repair = (
            question
            + "\n\n[格式修正] 上一次输出不符合要求。"
              "请严格只输出一个JSON对象，不要任何其他文字："
              "{\"label\":\"T|F|U|R\",\"reason\":\"...\"}"
        )
        msg2 = chain.invoke(repair, config=cfg)
        raw2 = getattr(msg2, "content", "") or str(msg2)
        label2, reason2 = parse_json_response(raw2)
        return msg2, raw2, label2, reason2

    return msg, raw, label, reason


# -----------------------------
# 6) (No-RAG) 체인: 프롬프트 → DeepSeek 추론
# -----------------------------
def build_no_rag_chain():
    """
    - RAG 없음(검색/컨텍스트 주입 제거)
    - prompts/no-rag-final.yaml 로드
    - extra로 JSON-only 출력 강제
    """
    prompt = load_prompt("prompts/no-rag-final.yaml", encoding="utf-8")
    print("프롬프트 로딩 완료")

    llm = ChatOllama(model=MODEL, temperature=1)

    chain = (
        prompt.partial(
            extra=(
                "请先在心里进行逐步推理（不要把推理过程写出来），"
                "然后只输出一个JSON对象（不要Markdown，不要代码块，不要多余文字）。\n"
                "JSON格式必须完全一致：\n"
                "{\"label\":\"T|F|U|R\",\"reason\":\"1~3句简短依据\"}\n"
                "如果无法确定，请输出U；只有在明确需要拒答的安全原因时才输出R。\n"
            )
        )
        | llm
    )
    return chain


# -----------------------------
# 7) main
# -----------------------------
if __name__ == "__main__":
    load_dotenv(dotenv_path="./.env")
    _disable_langsmith_env()

    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs("Result", exist_ok=True)
    os.makedirs("Msglog", exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

    # 1) No-RAG 체인
    DECISION_CHAIN = build_no_rag_chain()

    preds, golds = [], []
    results = []

    for q in data:
        d_id = q["d_id"]
        question = format_nli_prompt(
            predicate=q["predicate"],
            text=q["text"],
            hypothesis=q["hypothesis"],
            options={"T": "真", "F": "假", "U": "不能确定", "R": "模型拒绝回答"},
        )

        cfg = RunnableConfig(
            run_name=f"{d_id}",
            tags=[f"d_id:{d_id}", "NO_RAG", DATA_name],
            metadata={"d_id": d_id, "type": q.get("type"), "predicate": q.get("predicate")},
        )

        # 2) 모델 호출(+파싱 실패 시 1회 재시도)
        decision_msg, raw, final_label, reasoning_txt = invoke_with_retry(
            DECISION_CHAIN, question, cfg
        )

        # 3) raw 저장
        with open(os.path.join(RAW_DIR, f"{d_id}.raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw)

        # 4) JSONL 기록
        dump_decision_msg(
            path=MSGLOG_PATH,
            d_id=d_id,
            question=question,
            msg=decision_msg,
            final_label=final_label,
            reasoning_txt=reasoning_txt,
            cfg=dict(cfg),
        )

        print(f"*** {d_id} ***")
        print("final:", final_label, "| gold:", q["answer"])

        results.append({"d_id": d_id, "predicate": q["predicate"], "answer": final_label})

        if q["answer"] in {"T", "F", "U", "R"} and final_label in {"T", "F", "U", "R"}:
            preds.append(final_label)
            golds.append(q["answer"])

    # 평가
    print("\nClassification Report:")
    rep = classification_report(golds, preds, zero_division=0)
    print(rep)

    labels = ["T", "F", "U", "R"]
    cm = confusion_matrix(golds, preds, labels=labels)
    cm_df = pd.DataFrame(
        cm,
        index=pd.Index(labels, name="True label (행)"),
        columns=pd.Index(labels, name="Predicted label (열)"),
    )
    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    # 저장
    with open(PKL_OUT, "wb") as f:
        pickle.dump({"preds": preds, "golds": golds}, f)

    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write(f"{MODEL}\n\n")
        f.write("Classification Report:\n")
        f.write(rep)
        f.write("\nConfusion Matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n")
