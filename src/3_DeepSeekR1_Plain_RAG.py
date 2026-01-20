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

from typing import List, Dict, Any

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None


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
RAW_DIR = "Msglog/raw_deepseek_plainrag"


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
import json
import re

LABEL_SET = {"T", "F", "U", "R"}
ZH_MAP = {"真": "T", "假": "F", "不能确定": "U", "模型拒绝回答": "R"}

def parse_json_response(text: str):
    """
    DeepSeek 출력에서 최종 label/reason 추출(견고 버전)
    - <think>...</think> 제거(하지만 reason이 없으면 think를 fallback으로 사용)
    - codeblock(json) 우선 -> 없으면 문자열 내 마지막 JSON 객체를 파싱
    - JSON이 없으면 여러 패턴 시도 (영어/중국어 키워드 모두 지원)
    - 키는 label, answer, 答案 모두 허용
    """
    raw = text or ""

    # 1) think 내용 분리(있으면 reason 후보로 저장)
    think_match = re.search(r"<think>(.*?)</think>", raw, flags=re.DOTALL)
    think_txt = think_match.group(1).strip() if think_match else ""

    # 2) think 블록 제거(파싱 안정화)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL).strip()

    # 3) ```json ... ``` 우선
    code_block = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", cleaned, flags=re.DOTALL)
    if code_block:
        json_str = code_block.group(1).strip()
    else:
        # 4) 문자열 내 JSON object 후보 모두 찾고 마지막 것 사용
        matches = re.findall(r"\{[\s\S]*?\}", cleaned, flags=re.DOTALL)
        if matches:
            json_str = matches[-1].strip()
        else:
            # 5) JSON이 없으면 다양한 패턴 시도
            # 패턴 1: "answer": "T", "answer": T, answer: "T", answer: T
            # 패턴 2: 答案：T, 答案: T (중국어)
            # 패턴 3: reasoning:... answer: T (여러 줄에 걸쳐 있을 수 있음)
            simple_patterns = [
                r'(?:"answer"\s*:\s*"?([TFUR])"?|answer\s*:\s*"?([TFUR])"?)',
                r'(?:答案|答)\s*[：:]\s*"?([TFUR])"?',
                r'\b([TFUR])\s*$',  # 마지막에 단독으로 T, F, U, R이 나오는 경우
            ]
            
            for pattern in simple_patterns:
                simple_match = re.search(pattern, cleaned, flags=re.IGNORECASE | re.MULTILINE)
                if simple_match:
                    # 그룹 중 None이 아닌 첫 번째 값 찾기
                    label = None
                    for group in simple_match.groups():
                        if group:
                            label = group.upper()
                            break
                    
                    if label and label in LABEL_SET:
                        return label, think_txt
            
            # 6) 그래도 안되면 R 반환
            return "R", think_txt

    # 7) JSON 파싱 시도
    try:
        data = json.loads(json_str)
    except Exception as e:
        # JSON 파싱 실패 시 간단한 패턴 재시도
        simple_patterns = [
            r'(?:"answer"\s*:\s*"?([TFUR])"?|answer\s*:\s*"?([TFUR])"?)',
            r'(?:答案|答)\s*[：:]\s*"?([TFUR])"?',
            r'\b([TFUR])\s*$',
        ]
        
        for pattern in simple_patterns:
            simple_match = re.search(pattern, cleaned, flags=re.IGNORECASE | re.MULTILINE)
            if simple_match:
                label = None
                for group in simple_match.groups():
                    if group:
                        label = group.upper()
                        break
                
                if label and label in LABEL_SET:
                    return label, think_txt
        
        return "R", f"JSON parse error: {e}"

    # 8) label 우선, 없으면 answer도 허용
    label = (data.get("label") or data.get("answer") or "").strip()
    label = label.upper()

    # 중국어 값까지 대비
    if label in ZH_MAP:
        label = ZH_MAP[label]

    if label not in LABEL_SET:
        # label이 이상하면 R 처리 + reason 후보를 최대한 남김
        reason = (data.get("reason") or "").strip()
        if not reason:
            reason = think_txt
        return "R", reason

    # 9) reason 없으면 think_txt로 채움
    reason = (data.get("reason") or "").strip()
    if not reason:
        reason = think_txt

    return label, reason

def invoke_with_retry(chain, payload: dict, cfg: RunnableConfig):
    """
    - 1차 호출 후 parse_json_response로 label/reason 추출
    - JSON 파싱 오류가 명시적으로 난 경우에만 1회 '형식修正' 재시도
    """
    # 1) first try
    msg = chain.invoke(payload, config=cfg)
    raw_txt = getattr(msg, "content", "") or str(msg)
    label, reason = parse_json_response(raw_txt)

    # 2) retry only when JSON parsing error
    if label == "R" and (reason or "").startswith("JSON parse error"):
        repair_payload = dict(payload)
        repair_payload["question"] = (
            payload["question"]
            + "\n\n[格式修正] 请严格只输出一个JSON对象，不要任何其他文字："
              "{\"label\":\"T|F|U|R\",\"reason\":\"(2-4句简要理由)\"}"
        )
        msg2 = chain.invoke(repair_payload, config=cfg)
        raw_txt2 = getattr(msg2, "content", "") or str(msg2)
        label2, reason2 = parse_json_response(raw_txt2)
        return msg2, raw_txt2, label2, reason2

    return msg, raw_txt, label, reason

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

# -------Helper function ---------
def _load_docs_via_llamaparse(doc_dir: str):
    """
    2_GPT5_Plain_RAG.py의 load_docs() 패턴을 최대한 그대로 따름:
    - LlamaParse + SimpleDirectoryReader 로 doc/docx를 markdown으로 파싱
    - 성공 시 LangChain Document 리스트 반환
    """
    try:
        from llama_parse import LlamaParse
        from llama_index.core import SimpleDirectoryReader
    except Exception as e:
        raise RuntimeError(f"LlamaParse import failed: {e}")

    # LLAMA_CLOUD_API_KEY가 없으면 여기서 실패할 확률이 높음(환경에 따라 다름)
    parser = LlamaParse(
        result_type="markdown",
        num_workers=8,
        verbose=True,
        language="ch_sim",
    )
    file_extractor = {".doc": parser, ".docx": parser}

    # Bibliography/ 아래의 doc/docx 자동 수집
    input_files = []
    for root, _, files in os.walk(doc_dir):
        for fn in files:
            if fn.lower().endswith((".doc", ".docx")):
                input_files.append(os.path.join(root, fn))

    if not input_files:
        return []

    documents = SimpleDirectoryReader(
        input_files=input_files,
        file_extractor=file_extractor,
    ).load_data()

    # llama_index Document -> LangChain Document 변환
    return [doc.to_langchain_format() for doc in documents]


def _load_docx_locally(doc_dir: str):
    """
    로컬 폴백:
    - python-docx로 .docx만 읽어서 LangChain Document 유사 dict로 반환
    - .doc는 로컬에서 안정적으로 파싱하기 어렵기 때문에(추가 의존성 필요) 스킵/경고
    """
    docs = []

    if DocxDocument is None:
        print("WARN: python-docx not available. Cannot parse .docx locally.")
        return docs

    for root, _, files in os.walk(doc_dir):
        for fn in files:
            lower = fn.lower()
            p = os.path.join(root, fn)

            if lower.endswith(".docx"):
                try:
                    d = DocxDocument(p)
                    text = "\n".join([para.text for para in d.paragraphs if para.text and para.text.strip()])
                    if text.strip():
                        docs.append({"page_content": text, "metadata": {"source": p}})
                except Exception as e:
                    print(f"WARN: failed to parse docx: {p} ({e})")

            elif lower.endswith(".doc"):
                # .doc는 python-docx로 못 읽습니다. (antiword, textract 등 필요)
                print(f"WARN: .doc skipped (need external converter): {p}")

    return docs
# ------------------------------------------------------------------

def build_vectorstore():
    # 5.1 문서 소스 준비: (가능하면) pickle 재사용
    documents = None
    if os.path.exists(DOC_PKL):
        with open(DOC_PKL, "rb") as f:
            documents = pickle.load(f)
        print(f"OK: loaded {DOC_PKL} ({len(documents)} docs)")
    else:
        print(f"WARN: {DOC_PKL} not found. Building docs from {DOC_DIR}/ (doc/docx supported)")

        documents = []

        # (1) 1순위: LlamaParse로 doc/docx 파싱 (2_GPT5_Plain_RAG.py load_docs() 방식 참고)
        #     - 성공 시 품질이 가장 좋음(표/각주/구조가 비교적 잘 풀림)
        try:
            parsed = _load_docs_via_llamaparse(DOC_DIR)
            if parsed:
                documents = parsed
                print(f"OK: loaded via LlamaParse = {len(documents)} docs")
        except Exception as e:
            print(f"WARN: LlamaParse route unavailable. Fallback to local docx parse. ({e})")

        # (2) 2순위: 로컬 docx 파싱
        if not documents:
            if os.path.isdir(DOC_DIR):
                documents = _load_docx_locally(DOC_DIR)
            print(f"OK: local docx docs = {len(documents)}")

        # (3) 마지막: txt/md도 함께 주워담기(있으면 보너스)
        if os.path.isdir(DOC_DIR):
            for root, _, files in os.walk(DOC_DIR):
                for fn in files:
                    if fn.lower().endswith((".txt", ".md")):
                        p = os.path.join(root, fn)
                        try:
                            with open(p, "r", encoding="utf-8", errors="ignore") as rf:
                                txt = rf.read()
                            if txt.strip():
                                documents.append({"page_content": txt, "metadata": {"source": p}})
                        except Exception as e:
                            print(f"WARN: failed to read text file: {p} ({e})")

        print(f"OK: total docs collected = {len(documents)}")

        # 문서가 하나도 없으면, 여기서 멈추고 사용자에게 명확히 안내(FAISS IndexError 방지)
        if not documents:
            raise ValueError(
                f"No documents found in {DOC_DIR}/. "
                "Put at least one .docx (recommended), or configure LlamaParse for .doc/.docx parsing."
            )

        # 다음 실행부터는 재사용 가능하도록 pickle 저장(가능한 형태로)
        try:
            with open(DOC_PKL, "wb") as f:
                pickle.dump(documents, f)
            print(f"OK: created {DOC_PKL} ({len(documents)} docs)")
        except Exception as e:
            print(f"WARN: failed to save {DOC_PKL}: {e}")

    # 문서 형태를 LangChain Document 유사 구조로 정규화
    raw_texts = []
    metas = []
    for d in documents:
        txt = getattr(d, "page_content", None) or (d.get("page_content", "") if isinstance(d, dict) else "")
        meta = getattr(d, "metadata", None) or (d.get("metadata", {}) if isinstance(d, dict) else {})
        if txt and str(txt).strip():
            raw_texts.append(txt)
            metas.append(meta)

    if not raw_texts:
        raise ValueError("Documents loaded but all contents are empty. Check parsing results.")

    # 5.2 split
    splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60)
    splits = splitter.create_documents(raw_texts, metadatas=metas)
    print(f"OK: split into {len(splits)} chunks")

    if not splits:
        raise ValueError("Split produced 0 chunks. Increase text size or verify document parsing.")

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
        vectorstore = FAISS.load_local(
            FAISS_DIR,
            embeddings=cached_embedder,
            allow_dangerous_deserialization=True
        )
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
                "最后只输出一个JSON对象（不要Markdown，不要代码块，不要多余文字）。\n"
                "JSON格式必须完全一致：\n"
                "{\"label\":\"T|F|U|R\",\"reason\":\"1~3句简短依据，可引用参考资料中的关键短语\"}\n"
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

    os.makedirs("Result", exist_ok=True)
    os.makedirs("Msglog", exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)

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
        decision_msg, raw, final_label, reasoning_txt = invoke_with_retry(DECISION_CHAIN, question, cfg)

        # 콘솔에서 원문 일부 확인(너무 길면 앞부분만)
        print("RAW_MODEL_OUTPUT(preview):")
        print(raw[:800].replace("\n", " ") + (" ..." if len(raw) > 800 else ""))

        # 문항별 raw 저장
        with open(os.path.join(RAW_DIR, f"{d_id}.raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw)

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
