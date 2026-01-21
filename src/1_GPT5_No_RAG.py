# src/1_GPT5_No_RAG.py
from __future__ import annotations

import os
import re
import json
import pickle
from datetime import datetime

import nest_asyncio
from dotenv import load_dotenv

from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_openai import ChatOpenAI

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np


# ============================================================
# Config
# ============================================================
MODEL = "gpt-5"

INPUT = "data/Art_Nat_20250509.json"
DATA_name = re.findall(r"/(.+)\.json", INPUT)[0] if "/" in INPUT else re.findall(r"\\(.+)\.json", INPUT)[0]

# ===== Local trace logging =====
MSGLOG_DIR = "Msglog"
RAW_DIR = os.path.join(MSGLOG_DIR, "raw_gpt5_norag")
MSGLOG_PATH = os.path.join(MSGLOG_DIR, "GPT5-NoRAG-decision_msgs.jsonl")

RESULT_DIR = "Result"
SUBMIT_DIR = "submit"
RESULT_TXT = os.path.join(RESULT_DIR, f"GPT5-NoRAG-{DATA_name}.txt")
PKL_OUT = os.path.join(RESULT_DIR, f"GPT5-NoRAG-{DATA_name}-preds_golds.pkl")


def ensure_dirs():
    os.makedirs(MSGLOG_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(SUBMIT_DIR, exist_ok=True)


def dump_decision_msg(
    path: str,
    d_id: str,
    question: str,
    msg,  # AIMessage
    final_label: str,
    reasoning_txt: str,
    cfg: dict,
):
    """JSONL (한 줄 = 한 문항). DeepSeek 쪽 로그 구조와 최대한 유사하게 저장."""
    # AIMessage를 가능한 한 원형에 가깝게 덤프
    try:
        msg_dump = msg.model_dump()  # pydantic v2
    except Exception:
        try:
            msg_dump = msg.dict()
        except Exception:
            msg_dump = {"type": type(msg).__name__, "content": getattr(msg, "content", "")}

    rec = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "d_id": d_id,
        "question": question,
        "final_label": final_label,
        "reasoning_txt": (reasoning_txt or ""),
        "retrieved_docs": [],  # No-RAG이므로 빈 리스트 고정
        "decision_msg": msg_dump,
        "run_config": cfg,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# Tool-choice schema (No-RAG와 동일 패턴)
# ============================================================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "submit_judgment",
            "description": "输出最终判定及其简短理由",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "enum": ["T", "F", "U", "R"],
                        "description": "T=真, F=假, U=不能确定, R=模型拒绝回答",
                    },
                    "reason": {
                        "type": "string",
                        "description": "先用自然语言(不超过1000字)给出简洁而清晰的推理过程，尽量精炼(step-by-step reasoning)。",
                    },
                },
                "required": ["label", "reason"],
            },
        },
    }
]


def format_nli_prompt(predicate, text, hypothesis, options):
    return f"""
请根据以下输入内容判断：
- 前提（text）: {text}
- 假设（hypothesis）: {hypothesis}
- 判断依据: {predicate}
- 选项（option）: {options}
""".strip()


def parse_tool_result(ai_msg):
    """GPT-5 tool_choice='required' 결과에서 label/reason을 안정적으로 추출"""
    calls = getattr(ai_msg, "tool_calls", None)
    if not calls:
        raise ValueError("Empty tool_calls: model did not call submit_judgment")

    first = calls[0]
    args = first.get("args", {}) or {}

    label = (args.get("label", "") or "").strip()
    reason = (args.get("reason", "") or "").strip()

    if label not in {"T", "F", "U", "R"}:
        raise ValueError(f"Invalid label from tool args: {label}")
    if not reason:
        reason = "(no reason returned)"
    return label, reason


def NoRagChain():
    prompt = load_prompt("prompts/no-rag-final.yaml", encoding="utf-8")
    print("프롬프트 로딩 완료")

    llm = ChatOpenAI(
        model_name=MODEL,
        temperature=1,              # GPT-5는 실험 일관성 위해 고정
        parallel_tool_calls=False,  # tool_call 1회 강제에 유리
    )
    tool_llm = llm.bind_tools(TOOLS, tool_choice="required")

    chain = (
        {"question": RunnablePassthrough()}
        | prompt.partial(
            extra=(
                "先用自然语言(不超过1000字)给出简洁而清晰的推理过程，尽量精炼(step-by-step reasoning)。\n"
                "随后调用一次函数 submit_judgment，并在其参数中填入 {label, reason}。\n"
                "不要调用其它函数；不要省略 reason。"
            )
        )
        | tool_llm
    )
    return chain


# TP,FP,FN,TN 별 인덱스 뽑기 (원본 No-RAG 유틸 유지)
def get_confusion_cell_indices(y_true, y_pred, labels=("T", "F", "U", "R"), one_based=True):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    add = 1 if one_based else 0

    cell_idx = {t: {p: None for p in labels} for t in labels}
    for t in labels:
        for p in labels:
            idx = np.where((y_true == t) & (y_pred == p))[0] + add
            cell_idx[t][p] = idx
    return cell_idx


if __name__ == "__main__":
    ensure_dirs()
    load_dotenv(dotenv_path="./.env")
    nest_asyncio.apply()

    # LangSmith 사용 금지(안전하게 off)
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = ""
    os.environ["LANGCHAIN_PROJECT"] = ""

    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)

    chain = NoRagChain()

    results = []
    preds, golds = [], []

    for q in data:
        d_id = q["d_id"]

        question = format_nli_prompt(
            predicate=q["predicate"],
            text=q["text"],
            hypothesis=q["hypothesis"],
            options={"T": "真", "F": "假", "U": "不能确定", "R": "模型拒绝回答"},
        )

        cfg = RunnableConfig(
            run_name=str(d_id),
            tags=[f"d_id:{d_id}", "NoRAG", DATA_name],
            metadata={"d_id": d_id, "type": q.get("type"), "predicate": q.get("predicate")},
        )

        ai_msg = chain.invoke(question, config=cfg)
        final_label, reasoning_txt = parse_tool_result(ai_msg)

        # ===== (1) raw 저장: 모델 응답 원형 =====
        try:
            raw = json.dumps(ai_msg.model_dump(), ensure_ascii=False, indent=2)
        except Exception:
            raw = str(ai_msg)

        with open(os.path.join(RAW_DIR, f"{d_id}.raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw)

        # ===== (2) JSONL 누적 저장 =====
        dump_decision_msg(
            path=MSGLOG_PATH,
            d_id=d_id,
            question=question,
            msg=ai_msg,
            final_label=final_label,
            reasoning_txt=reasoning_txt,
            cfg=dict(cfg),
        )

        print(f"*** {d_id} ***")
        print("final:", final_label, "| gold:", q.get("answer", "(no gold)"))

        results.append({"d_id": d_id, "predicate": q["predicate"], "answer": final_label, "reason": reasoning_txt})

        gold = q.get("answer")
        if gold in {"T", "F", "U", "R"} and final_label in {"T", "F", "U", "R"}:
            preds.append(final_label)
            golds.append(gold)

    # submit 저장
    out_path = os.path.join(SUBMIT_DIR, f"submit_{MODEL}-NoRAG-{DATA_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nOK: wrote {out_path}")

    # 평가 + 파일 저장
    rep = ""
    cm_df = None
    if golds:
        rep = classification_report(golds, preds, digits=4, zero_division=0)
        cm = confusion_matrix(golds, preds, labels=["T", "F", "U", "R"])
        cm_df = pd.DataFrame(
            cm,
            index=pd.Index(["T", "F", "U", "R"], name="True"),
            columns=pd.Index(["T", "F", "U", "R"], name="Pred"),
        )

        with open(PKL_OUT, "wb") as f:
            pickle.dump({"preds": preds, "golds": golds}, f)

        with open(RESULT_TXT, "w", encoding="utf-8") as f:
            f.write(f"{MODEL} No-RAG\n\n")
            f.write("Classification Report:\n")
            f.write(rep + "\n\n")
            f.write("Confusion Matrix:\n")
            f.write(cm_df.to_string() + "\n")

        print("\n[Classification Report]\n", rep)
        print("\n[Confusion Matrix]\n", cm_df.to_string())
    else:
        print("\nINFO: gold 라벨이 없어 평가를 생략했습니다.")
