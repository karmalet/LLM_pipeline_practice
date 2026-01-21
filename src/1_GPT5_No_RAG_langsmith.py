from datetime import datetime
from typing import Optional
from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

import numpy as np
import pickle

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import os, re, json, pickle, nest_asyncio, tiktoken
from dotenv import load_dotenv

MODEL = "gpt-5"
INPUT = "data/Art_Nat_20250509.json"
DATA_name = re.findall(r"/(.+)\.json", INPUT)[0]


# === Tool-choice labels for deterministic parsing ===
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
                        "description": "T=真, F=假, U=不能确定, R=模型拒绝回答"
                    },
                    "reason": {
                        "type": "string",
                        "description": "先用自然语言(不超过1000字)给出简洁而清晰的推理过程，尽量精炼(step-by-step reasoning)。"
                    }
                },
                "required": ["label", "reason"]
            }
        }
    }
]


def format_nli_prompt(predicate, text, hypothesis, options):
    q = f"""
请根据以下输入内容判断：
- 前提（text）: {text}
- 假设（hypothesis）: {hypothesis}
- 判断依据: {predicate}
- 选项（option）: {options}
"""
    return q


def NoRagChain():
        # 프롬프트 로드
        prompt = load_prompt("prompts/no-rag-final.yaml", encoding="utf-8")
        print("프롬프트 로딩 완료")

        # Ollama 모델 지정
        # llm = ChatOpenAI(model_name=MODEL, temperature=0)
        llm = ChatOpenAI(model_name=MODEL, temperature=1, parallel_tool_calls=False)  # GPT-5는 temperature 조절 못함.

        #1 정답을 말하는 체인 생성 (tool-choice 강제)
        tool_llm = llm.bind_tools(TOOLS, tool_choice="required")
        DECISION_CHAIN = (
            {"question": RunnablePassthrough(),} 
            | prompt.partial(
                extra=("先用自然语言(不超过1000字)给出简洁而清晰的推理过程，尽量精炼(step-by-step reasoning)。\n"
                       "随后调用一次函数 submit_judgment，并在其参数中填入 {label, reason}。\n"
                       "不要调用其它函数；不要省略 reason。")
            )
            | tool_llm
        )
        return DECISION_CHAIN

def parse_tool_result(ai_msg):
    """
    ai_msg.tool_calls['arg']를 파싱
    """
    try:
        calls = getattr(ai_msg, "tool_calls")
        if calls:
            try:
                first = calls[0]
                label = first['args'].get("label", "레이블 없음")
                reason = first['args'].get("reason", "이유 없음").strip()
                # if label in {"T", "F", "U", "R"}:
                return label, reason

            except Exception as e:
                print(e)
                input('TOLL call 파싱 에러 발생!!!')
        else:
            input('비어있는 TOOL CALL')
    except Exception:
        input('decision_msg가 비어 있음')

# TP,FP,FN,TN 별 인덱스 뽑기
def get_confusion_cell_indices(y_true, y_pred, labels=("T","F","U","R"), one_based=True):
    """
    혼동행렬의 각 (실제→예측) 셀에 들어가는 샘플 인덱스를 반환.
    반환 형식: cell_idx[true_label][pred_label] = np.ndarray([...])

    one_based=True 이면 1부터 시작하는 인덱스(보고용)로 반환합니다.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    add = 1 if one_based else 0

    cell_idx = {t: {p: None for p in labels} for t in labels}
    for t in labels:
        for p in labels:
            idx = np.where((y_true == t) & (y_pred == p))[0] + add
            cell_idx[t][p] = idx
    return cell_idx


def print_confusion_cells(y_true, y_pred, labels=("T","F","U","R"), one_based=True, max_show=50):
    """
    각 셀의 개수와(필요하면) 앞부분 인덱스를 프린트.
    max_show: 너무 길면 앞에서부터 max_show개만 표시.
    """
    cells = get_confusion_cell_indices(y_true, y_pred, labels, one_based)
    for t in labels:
        for p in labels:
            idx = cells[t][p]
            n = len(idx)
            head = np.array2string(idx[:max_show], separator=", ")
            tail = "" if n <= max_show else f" ... (+{n - max_show} more)"
            print(f"{t}→{p}: {n}개 {head}{tail}")
    return cells


if __name__ == "__main__":
    # API 키 정보 로드
    load_dotenv(dotenv_path="./.env")
    nest_asyncio.apply()  # LLAMA parser
    # logging.langsmith(f"{MODEL}-NoRAG-{DATA_name}")

    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)

    results = []

    DECISION_CHAIN = NoRagChain()

    for query_data in data[:1]:
        id = query_data["d_id"]
        question = format_nli_prompt(
            predicate=query_data["predicate"],
            text=query_data["text"],
            hypothesis=query_data["hypothesis"],
            options={"T": "真", "F": "假", "U": "不能确定", "R": "模型拒绝回答"},
        )
        cfg = RunnableConfig(
            run_name=f"{id}",
            tags=[f"d_id:{id}", "NoRAG", DATA_name],
            metadata={
                "d_id": id,
                "type": query_data.get("type"),
                "predicate": query_data.get("predicate"),
            },
        )

        # 1 모델 호출
        decision_msg = DECISION_CHAIN.invoke(question, config=cfg)
        # 간단ver.
        # decision_msg = DECISION_CHAIN.invoke(question)
        print(f'*** {id} ***')
        print(decision_msg)

        # 3 최종 라벨 파싱 (tool choice)
        final_label, reasoning_txt = parse_tool_result(decision_msg)
        # 간단ver.
        # final_label, reasoning_txt = parse_tool_result(decision_msg.content)

        # # 4 LangSmith에 reasoning을 기록 (LLM 재호출 없이)
        # RunnableLambda(lambda x: x).with_config(
        #     run_name=f"{cfg['run_name']}#reasoning",  # 리스트에서 쉽게 구분되도록 suffix
        #     tags=cfg["tags"],  # d_id:... 등 기존 태그 재사용
        #     metadata={**cfg["metadata"], "reasoning_txt": reasoning_txt},
        # ).invoke(None)  # no-op 실행 → LangSmith에 한 줄 남김

        # 모델의 응답 기록
        print(f"{id}_最终答案:", final_label)
        result = {"answer": final_label}
        result["d_id"] = id
        result["predicate"] = query_data["predicate"]
        result["final_label"] = final_label
        result["reasoning_txt"] = reasoning_txt
        results.append(result)

        # Gold 표시
        print(f"{id}_Gold:", query_data["answer"])

    # 정답과 예측 비교
    gold_dict = {query_data["d_id"]: query_data["answer"] for query_data in data}
    comparison = []
    preds, golds = [], []

    for pred in results:
        d_id = pred["d_id"]
        model_ans = pred["answer"]
        gold_ans = gold_dict.get(d_id, "정답 없음")

        comparison.append(
            {
                "d_id": d_id,
                "predicate": pred["predicate"],
                "model_prediction": model_ans,
                "gold_answer": gold_ans,
                "match": model_ans == gold_ans,
            }
        )

        if gold_ans in {"T", "F", "U", "R"} and model_ans in {"T", "F", "U", "R"}:
            preds.append(model_ans)
            golds.append(gold_ans)
        else:
            print(f'gold_answer:{gold_ans}\nmodel_answer:{model_ans}')
            input('문제 발생: gold_ans에 답이 없거나 model_ans에 답이 없음')

    # 평가 출력
    print("\nClassification Report:")
    a = classification_report(golds, preds, zero_division=0)
    print(a)
    labels=["T", "F", "U", "R"]
    print("Confusion Matrix:")
    b = confusion_matrix(golds, preds, labels=labels)
    cm_df = pd.DataFrame(b,
                         index=pd.Index(labels, name="True label (행)"),
                         columns=pd.Index(labels, name="Predicted label (열)")
                         )
    print(cm_df.to_string())

    cells = get_confusion_cell_indices(golds, preds, labels, one_based=True)


