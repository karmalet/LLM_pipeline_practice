# src/2_GPT5_Plain_RAG.py
from __future__ import annotations

import os
import re
import json
import pickle
import nest_asyncio
import tiktoken

from dotenv import load_dotenv

from langchain.embeddings import CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.prompts import load_prompt
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableConfig,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from sklearn.metrics import classification_report, confusion_matrix

from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# ===== (NEW) Local trace logging =====
from datetime import datetime

MSGLOG_DIR = "Msglog"
RESULT_TXT = f"Result/GPT5-PlainRAG-{DATA_name}.txt"
PKL_OUT = "DeepSeekR1-PlainRAG-preds_golds.pkl"
RAW_DIR = os.path.join(MSGLOG_DIR, "raw_gpt5_plainrag")
MSGLOG_PATH = os.path.join(MSGLOG_DIR, "GPT5-PlainRAG-decision_msgs.jsonl")

def ensure_dirs():
    os.makedirs(MSGLOG_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

def dump_decision_msg(
    path: str,
    d_id: str,
    question: str,
    msg,  # AIMessage
    final_label: str,
    reasoning_txt: str,
    cfg: dict,
    retrieved_docs=None,
):
    """DeepSeek 쪽 JSONL 로그 포맷과 최대한 유사하게 저장."""
    if retrieved_docs is None:
        retrieved_docs = []

    # AIMessage를 가능한 한 “원형”에 가깝게 덤프 (tool_calls 포함)
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
        "retrieved_docs": retrieved_docs,
        "decision_msg": msg_dump,
        "run_config": cfg,
    }

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ============================================================
# Config
# ============================================================
MODEL = "gpt-5"

# 예시 입력(JSON). repo에서 실제 파일명에 맞춰 조정하세요.
INPUT = "data/Art_Nat_20250509.json"
DATA_name = re.findall(r"/(.+)\.json", INPUT)[0]

VECTOR_SEARCH_TOP_K = 10


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


# ============================================================
# Utility
# ============================================================
def format_nli_prompt(predicate, text, hypothesis, options):
    return f"""
请根据以下输入内容判断：
- 前提（text）: {text}
- 假设（hypothesis）: {hypothesis}
- 判断依据: {predicate}
- 选项（option）: {options}
""".strip()


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))


def parse_tool_result(ai_msg):
    """
    GPT-5 tool_choice="required" 결과에서 label/reason을 안정적으로 추출
    (No-RAG 스크립트와 동일한 접근)
    """
    try:
        calls = getattr(ai_msg, "tool_calls", None)
        if not calls:
            raise ValueError("Empty tool_calls: model did not call submit_judgment")

        first = calls[0]
        args = first.get("args", {}) or {}

        label = args.get("label", "").strip()
        reason = (args.get("reason", "") or "").strip()

        if label not in {"T", "F", "U", "R"}:
            raise ValueError(f"Invalid label from tool args: {label}")

        if not reason:
            reason = "(no reason returned)"

        return label, reason

    except Exception as e:
        # 실습 사용자(비전공자) 관점에서 에러 원인을 눈에 띄게 출력
        print("\n[ERROR] Tool-call parsing failed.")
        print("  - 원인:", repr(e))
        print("  - ai_msg:", ai_msg)
        raise


# ============================================================
# RAG: 문서 로드 + 벡터화
# ============================================================
def load_docs():
    """
    주의:
    - PDF로 바로 로드하면 각주/예문이 섞이는 문제가 생길 수 있어,
      (이미 사용자 코드에 있던 방식대로) 문헌을 doc/docx로 준비해 LlamaParse로 읽는다.
    """
    parser = LlamaParse(
        result_type="markdown",
        num_workers=8,
        verbose=True,
        language="ch_sim",
    )
    file_extractor = {".doc": parser, ".docx": parser}

    documents = SimpleDirectoryReader(
        input_files=[
            "Bibliography/0_第一届中文叙实性推理评测任务.docx",
            "Bibliography/1_叙实性与事实性.doc",
            "Bibliography/2_反叙实动词宾语真假的语法条件及其概念动因.doc",
            "Bibliography/3_知道_的叙实性及其置信度变异的语法环境.doc",
            "Bibliography/4_忘记类动词的叙实性.docx",
            "Bibliography/5_记得类叙实性.docx",
        ],
        file_extractor=file_extractor,
    ).load_data()

    return [doc.to_langchain_format() for doc in documents]


def format_docs(docs, docid_map):
    """
    검색된 문서를 프롬프트에 넣기 좋은 형태로 직렬화
    """
    out = []
    for doc in docs:
        # doc.id는 LangChain Document의 내부 id (환경에 따라 다를 수 있음)
        # vectorstore.index_to_docstore_id를 역매핑한 docid_map을 사용
        src = docid_map.get(doc.id, str(doc.id))
        out.append(
            f"<document><content>{doc.page_content}</content><source>{src}</source></document>"
        )
    return "\n\n".join(out)


def build_vectorstore():
    """
    - 문헌 파싱 결과(pickle)가 있으면 재사용
    - 없으면 새로 파싱 후 저장
    - OpenAI embedding(text-embedding-3-large) + CacheBackedEmbeddings + FAISS
    """
    pickle_path = "llama_parsed_docs.pickle"

    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            docs = pickle.load(f)
        print(f"OK: loaded {pickle_path} ({len(docs)} docs)")
    else:
        docs = load_docs()
        with open(pickle_path, "wb") as f:
            pickle.dump(docs, f)
        print(f"OK: created {pickle_path} ({len(docs)} docs)")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    print(f"OK: split into {len(splits)} chunks")

    store = LocalFileStore("./cache_for_rag_openAIembeddings/")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",chunk_size=16)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings,
        document_embedding_cache=store,
        namespace=embeddings.model,
    )
    print("OK: embeddings ready (with cache)")

    vectorstore = FAISS.from_documents(documents=splits, embedding=cached_embedder)
    print("OK: FAISS vectorstore built")

    # 문서 id 매핑(출처 표기용)
    # index_to_docstore_id: {index: docstore_id}
    idx2id = vectorstore.index_to_docstore_id
    docid_map = {docstore_id: idx for idx, docstore_id in idx2id.items()}

    return vectorstore, docid_map


# ============================================================
# Plain-RAG chain (tool_choice required)
# ============================================================
def PlainRagChain(vectorstore, docid_map):
    """
    (context = retriever 검색 결과) + (question = NLI prompt)
    -> rag-final 프롬프트
    -> GPT-5 tool_choice="required"로 submit_judgment 강제
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": VECTOR_SEARCH_TOP_K})

    # 프롬프트 파일명은 No-RAG의 no-rag-final.yaml에 대응되도록 rag-final.yaml을 권장
    prompt = load_prompt("./prompts/rag-final.yaml", encoding="utf-8")
    print("OK: prompt loaded (rag-final.yaml)")

    llm = ChatOpenAI(
        model_name=MODEL,
        temperature=1,                # No-RAG와 동일
        parallel_tool_calls=False,    # tool_call 1회 강제에 유리
    )

    tool_llm = llm.bind_tools(TOOLS, tool_choice="required")

    # context 직렬화 함수(문서 출처표시 포함)
    def _fmt_docs(docs):
        return format_docs(docs, docid_map)

    chain = (
        {
            "context": retriever | _fmt_docs,
            "question": RunnablePassthrough(),
        }
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


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    ensure_dirs()
    load_dotenv()
    nest_asyncio.apply()  # LlamaParse (async) 대응
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGCHAIN_API_KEY"] = ""
    os.environ["LANGCHAIN_PROJECT"] = ""

    # 1) 데이터 로드
    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)

    # 2) 벡터스토어 구축(또는 재사용)
    vectorstore, docid_map = build_vectorstore()

    # 3) Plain-RAG 체인 구성
    chain = PlainRagChain(vectorstore, docid_map)

    results = []
    y_true, y_pred = [], []

    # (실습 단계에서는 1개만. 논문 실험 시엔 data 전체로 확장)
    for query_data in data:
        d_id = query_data["d_id"]

        question = format_nli_prompt(
            predicate=query_data["predicate"],
            text=query_data["text"],
            hypothesis=query_data["hypothesis"],
            options={"T": "真", "F": "假", "U": "不能确定", "R": "模型拒绝回答"},
        )

        # LangSmith 등 관측 도구에서 문항 단위로 식별하기 쉽게 config 부여
        config = RunnableConfig(
            run_name=str(d_id),
            tags=["Plain-RAG", "GPT-5"],
            metadata={
                "d_id": d_id,
                "model": MODEL,
                "top_k": VECTOR_SEARCH_TOP_K,
                "dataset": DATA_name,
            },
        )

        ai_msg = chain.invoke(question, config=config)
        pred_label, reason = parse_tool_result(ai_msg)

        print("\n==============================")
        print("d_id:", d_id)
        print("pred:", pred_label)
        print("reason:", reason[:300] + ("..." if len(reason) > 300 else ""))

        # 정답/예측 저장
        gold = query_data.get("answer")
        if gold:
            y_true.append(gold)
            y_pred.append(pred_label)

        results.append(
            {
                "d_id": d_id,
                "predicate": query_data["predicate"],
                "pred": pred_label,
                "reason": reason,
            }
        )
        
        # ===== (NEW) raw + jsonl trace 저장 =====
        # raw는 “모델 응답 원형”을 최대한 남기기 (content + tool_calls)
        try:
            raw = json.dumps(ai_msg.model_dump(), ensure_ascii=False, indent=2)
        except Exception:
            raw = str(ai_msg)

        # 3) raw 저장
        os.makedirs(RAW_DIR, exist_ok=True)

        with open(os.path.join(RAW_DIR, f"{d_id}.raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw)

        # 4) JSONL 기록 (DeepSeek와 동일 개념의 로컬 트레이스)
        dump_decision_msg(
            path=MSGLOG_PATH,
            d_id=d_id,
            question=question,
            msg=ai_msg,
            final_label=pred_label,
            reasoning_txt=reason,   # GPT-5는 tool reason을 reasoning_txt로 저장
            cfg=dict(config),
            retrieved_docs=[],      # GPT-5 Plain-RAG에서도 원하면 retrieved_docs를 여기에 넣을 수 있음
        )


    # 4) 제출/로그 저장
    os.makedirs("submit", exist_ok=True)
    out_path = f"submit/submit_{MODEL}-PlainRAG-{DATA_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nOK: wrote {out_path}")

    # 5) 평가(정답이 있는 경우)
    if y_true:
        print("\n[Classification Report]")
        rep = classification_report(y_true, y_pred, digits=4)
        print(rep)
        print("\n[Confusion Matrix] (labels: T,F,U,R)")
        cm_df = confusion_matrix(y_true, y_pred, labels=["T", "F", "U", "R"])
        print()

    # 파일로 저장
    os.makedirs("Result", exist_ok=True)
    with open(PKL_OUT, "wb") as f:
        pickle.dump({"preds": y_pred, "golds": y_true}, f)

    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write(f"{MODEL}, k = {VECTOR_SEARCH_TOP_K}\n\n")
        f.write("Classification Report:\n")
        f.write(rep)
        f.write("\nConfusion Matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n")
