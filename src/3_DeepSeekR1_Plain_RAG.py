# src/3_DeepSeekR1_Plain_RAG.py
from datetime import datetime
import os, re, json, pickle
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from langchain_core.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough, RunnableConfig
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

# -----------------------------
# 0) 실험 설정
# -----------------------------
MODEL = "deepseek-r1:14b"
INPUT = "data/Art_Nat_20250509.json"
DATA_name = re.findall(r"/(.+)\.json", INPUT)[0]

# RAG 설정
VECTOR_SEARCH_TOP_K = 6
EMBED_MODEL = "nomic-embed-text"  # Ollama embedding model (로컬)
CACHE_DIR = "cache_for_rag_ollama_embeddings"
FAISS_DIR = "faiss_index_deepseek_plainrag"
DOC_PKL = "llama_parsed_docs.pickle"  # (있으면) 재사용
DOC_DIR = "Bibliography"             # (없으면) 경고만 하고 넘어감

# 로그/결과 저장
MSGLOG_PATH = "Msglog/DeepSeekR1-PlainRAG-decision_msgs.jsonl"
RESULT_TXT = f"Result/DeepSeekR1-PlainRAG-{DATA_name}.txt"
PKL_OUT = "DeepSeekR1-PlainRAG-preds_golds.pkl"


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

def dump_decision_msg(path, d_id, question, msg, final_label, reasoning_txt, cfg=None, retrieved=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "d_id": d_id,
        "question": question,
        "final_label": final_label,
        "reasoning_txt": reasoning_txt,
        "retrieved_docs": retrieved or [],
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
# 5) (Plain-RAG) 벡터스토어 구축/재사용
# -----------------------------
def build_vectorstore():
    # 5.1 문서 소스 준비: (가능하면) pickle 재사용
    documents = None
    if os.path.exists(DOC_PKL):
        with open(DOC_PKL, "rb") as f:
            documents = pickle.load(f)
        print(f"OK: loaded {DOC_PKL} ({len(documents)} docs)")
    else:
        # pickle이 없으면, Bibliography/에서 텍스트 파일만 대충 읽는 fallback
        # (비전공자용: "없으면 경고" + "일단 실행은 가능"이 목적)
        print(f"WARN: {DOC_PKL} not found. Fallback to reading files in {DOC_DIR}/")
        documents = []
        if os.path.isdir(DOC_DIR):
            for root, _, files in os.walk(DOC_DIR):
                for fn in files:
                    if fn.lower().endswith((".txt", ".md")):
                        p = os.path.join(root, fn)
                        with open(p, "r", encoding="utf-8", errors="ignore") as rf:
                            documents.append({"page_content": rf.read(), "metadata": {"source": p}})
        print(f"OK: fallback docs = {len(documents)}")

    # 문서 형태를 LangChain Document 유사 구조로 정규화
    # (pickle이 LangChain Document 리스트라면 그대로 사용될 가능성이 높음)
    # 여기서는 page_content 속성만 있으면 OK인 방식으로 처리
    raw_texts = []
    metas = []
    for d in documents:
        txt = getattr(d, "page_content", None) or d.get("page_content", "")
        meta = getattr(d, "metadata", None) or d.get("metadata", {})
        raw_texts.append(txt)
        metas.append(meta)

    # 5.2 split
    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60)
    splits = splitter.create_documents(raw_texts, metadatas=metas)
    print(f"OK: split into {len(splits)} chunks")

    # 5.3 embeddings (로컬) + cache
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    store = LocalFileStore(CACHE_DIR)
    cached_embedder = CacheBackedEmbeddings.from_bytes_store(
        underlying_embeddings=embeddings,
        document_embedding_cache=store,
        namespace=EMBED_MODEL,
    )
    print("OK: embeddings ready (ollama + cache)")

    # 5.4 FAISS: 디스크에 저장/재사용
    if os.path.isdir(FAISS_DIR):
        vectorstore = FAISS.load_local(FAISS_DIR, embeddings=cached_embedder, allow_dangerous_deserialization=True)
        print(f"OK: loaded FAISS index from {FAISS_DIR}/")
    else:
        vectorstore = FAISS.from_documents(splits, cached_embedder)
        os.makedirs(FAISS_DIR, exist_ok=True)
        vectorstore.save_local(FAISS_DIR)
        print(f"OK: built & saved FAISS index to {FAISS_DIR}/")

    return vectorstore


# -----------------------------
# 6) (Plain-RAG) 체인: 검색 → 컨텍스트 주입 → DeepSeek 추론
# -----------------------------
def build_plain_rag_chain(vectorstore):
    prompt = load_prompt("prompts/no-rag-final.yaml", encoding="utf-8")
    print("프롬프트 로딩 완료")

    llm = ChatOllama(model=MODEL, temperature=1)

    def retrieve_context(question: str):
        docs = vectorstore.similarity_search(question, k=VECTOR_SEARCH_TOP_K)
        # (출처 추적을 위한 간단 표기)
        snippets = []
        for i, d in enumerate(docs, 1):
            src = (d.metadata or {}).get("source", "unknown")
            snippets.append(f"[{i}] source={src}\n{d.page_content}")
        context = "\n\n".join(snippets)
        return {"context": context, "docs": docs}

    # JSON 출력 형식 강제(원본 스크립트와 동일한 방식)
    chain = (
        {"question": RunnablePassthrough()}
        | RunnablePassthrough.assign(
            retrieved=lambda x: retrieve_context(x["question"])
        )
        | (lambda x: {
            "question": x["question"],
            "context": x["retrieved"]["context"],
            "docs": x["retrieved"]["docs"],
        })
        | prompt.partial(
            extra=(
                "请用自然语言(不超过1000字)进行逐步推理，"
                "最后只输出一个JSON对象，格式如下：\n"
                "{ \"label\": \"T/F/U/R\", \"reason\": \"...\" }。\n"
                "不要输出其他任何文字。\n\n"
                "【可用参考资料】\n"
                "{context}\n"
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

    # 1) 벡터스토어 구축/재사용
    vectorstore = build_vectorstore()

    # 2) Plain-RAG 체인
    DECISION_CHAIN = build_plain_rag_chain(vectorstore)

    results = []
    preds, golds = [], []
    gold_dict = {q["d_id"]: q["answer"] for q in data}

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
            tags=[f"d_id:{d_id}", "PLAIN_RAG", DATA_name],
            metadata={"d_id": d_id, "type": q.get("type"), "predicate": q.get("predicate")},
        )

        # 3) 모델 호출
        decision_msg = DECISION_CHAIN.invoke({"question": question})

        # 4) 최종 라벨 파싱
        final_label, reasoning_txt = parse_json_response(getattr(decision_msg, "content", ""))

        # 5) 로컬 JSONL 기록(검색 결과 출처 포함)
        # - chain 중간에서 docs를 직접 빼기 어렵기 때문에, 여기서는 최소 기록만 남김
        dump_decision_msg(
            path=MSGLOG_PATH,
            d_id=d_id,
            question=question,
            msg=decision_msg,
            final_label=final_label,
            reasoning_txt=reasoning_txt,
            cfg=dict(cfg),
            retrieved=[],  # 필요 시: retrieve_context를 별도로 재호출해 저장 가능
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
    cm_df = pd.DataFrame(cm,
                         index=pd.Index(labels, name="True label (행)"),
                         columns=pd.Index(labels, name="Predicted label (열)"))
    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    # 저장
    os.makedirs("Result", exist_ok=True)
    os.makedirs("Msglog", exist_ok=True)

    with open(PKL_OUT, "wb") as f:
        pickle.dump({"preds": preds, "golds": golds}, f)

    with open(RESULT_TXT, "w", encoding="utf-8") as f:
        f.write(f"{MODEL}, k = {VECTOR_SEARCH_TOP_K}\n\n")
        f.write("Classification Report:\n")
        f.write(rep)
        f.write("\nConfusion Matrix:\n")
        f.write(cm_df.to_string())
        f.write("\n")
