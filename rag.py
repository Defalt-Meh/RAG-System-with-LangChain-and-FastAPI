"""
rag.py — Retrieval-Augmented Generation backend

Principles:
1) Initialize once, reuse many (no per-request rebuilds).
2) Two modes:
   - OpenAI mode (OPENAI_API_KEY set): embeddings + LLM generation.
   - Free mode (no key): question-focused extractive answers.
3) Keep deps light in free mode; fail gracefully elsewhere.
4) Always return provenance (snippets + source).
"""

from __future__ import annotations

import os
import re
import glob
import time
import math
from typing import List, Dict, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter

# ----------------------------
# Mode detection (kept simple)
# ----------------------------
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))
if USE_OPENAI:
    try:
        from langchain_openai.embeddings import OpenAIEmbeddings
        from langchain_openai import ChatOpenAI
        try:
            from langchain_community.vectorstores import FAISS
            _FAISS_OK = True
        except Exception:
            _FAISS_OK = False
    except Exception:
        USE_OPENAI = False  # degrade to free mode if imports fail

# ----------------------------
# Singletons / Config
# ----------------------------
# Slightly larger chunks improve retrieval quality for prose datasets
_TEXT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
_DOCS_DIR = os.getenv("DOCS_DIR", "data")
# Optional: limit to specific files; default grabs all .txt/.md
_DOCS_GLOB = os.getenv("DOCS_GLOB", "*.txt,*.md")
_INDEX_READY = False

# OpenAI path
_embeddings = None          # type: ignore
_vectorstore = None         # type: ignore

# Free path
_free_chunks: List[Tuple[str, str]] = []  # (source_path, chunk_text)

# ----------------------------
# Ingestion filters
# ----------------------------
_PROMPTS_HEADER_RE = re.compile(r"^[=\s]*SECTION:\s*PROMPTS.*?$", re.IGNORECASE | re.MULTILINE)
_SECTION_HEADER_RE = re.compile(r"^[=\s]*SECTION:\s*.+?$", re.IGNORECASE | re.MULTILINE)

def _strip_ignored_sections(text: str) -> str:
    """
    Remove UI/testing prompts from retrieval.
    Cuts from 'SECTION: PROMPTS...' to the next 'SECTION:' or EOF.
    Safe no-op if absent.
    """
    m = _PROMPTS_HEADER_RE.search(text)
    if not m:
        return text
    start = m.start()
    tail = text[m.end():]
    n = _SECTION_HEADER_RE.search(tail)
    end = len(text) if not n else m.end() + n.start()
    return (text[:start] + text[end:]).strip()

# ----------------------------
# Utilities
# ----------------------------
def _expand_globs(folder: str, pattern_csv: str) -> List[str]:
    """Allow multiple globs via comma-separated patterns."""
    paths: List[str] = []
    for pat in [p.strip() for p in pattern_csv.split(",") if p.strip()]:
        paths.extend(glob.glob(os.path.join(folder, pat)))
    return sorted(set(paths))  # stable order

def _read_all_text_files(folder: str) -> List[Tuple[str, str]]:
    """
    Read text sources from `folder`. If none, seed a small sample.
    Skips helper files:
      - names starting with '_'
      - names containing 'prompts'
    Returns list of (path, content).
    """
    os.makedirs(folder, exist_ok=True)

    # Prefer explicit file if present (your case: data/my_document.txt)
    preferred = os.path.join(folder, "my_document.txt")
    if os.path.isfile(preferred):
        candidates = [preferred]
    else:
        candidates = _expand_globs(folder, _DOCS_GLOB)

    if not candidates:
        # Seed to avoid “empty index” surprises
        sample_path = os.path.join(folder, "sample_polar_bears.txt")
        sample_text = (
            "Polar bears (Ursus maritimus) rely on sea ice to hunt seals, their primary food source. "
            "They are strong swimmers with large paws and a thick fat layer for insulation. "
            "Climate change reduces sea ice, shrinking hunting seasons and threatening populations."
        )
        with open(sample_path, "w", encoding="utf-8") as f:
            f.write(sample_text)
        candidates = [sample_path]

    docs: List[Tuple[str, str]] = []
    for p in candidates:
        base = os.path.basename(p)
        if base.startswith("_") or "prompts" in base.lower():
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                docs.append((p, f.read()))
        except Exception:
            # Skip unreadable files instead of failing the pipeline
            continue
    return docs

def _simple_tokenize(text: str) -> List[str]:
    """Tiny tokenizer for free mode (no heavy NLP deps)."""
    return re.findall(r"[a-zA-Z0-9']+", text.lower())

def _keyword_overlap_score(query: str, text: str) -> float:
    """Jaccard overlap of unique tokens; cheap baseline for ranking chunks."""
    q = set(_simple_tokenize(query))
    t = set(_simple_tokenize(text))
    return 0.0 if not q or not t else len(q & t) / len(q | t)

# ----------------------------
# Free-mode sentence selection
# ----------------------------
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')

def _split_sentences(text: str) -> List[str]:
    # Cheap sentence splitter; trims and drops tiny fragments
    sents = [s.strip() for s in _SENT_SPLIT_RE.split(text) if len(s.strip()) > 2]
    return sents[:50]  # safety cap

def _score_sentence(query_tokens: set, sent: str) -> float:
    # Keyword overlap + shortness bias (slight)
    toks = set(_simple_tokenize(sent))
    if not toks:
        return 0.0
    inter = len(query_tokens & toks)
    union = len(query_tokens | toks)
    jaccard = inter / union if union else 0.0
    length_penalty = 1.0 / (1.0 + math.log2(1 + len(sent)))
    return jaccard * 0.9 + length_penalty * 0.1

def _compose_answer_free(query: str, hits: List[Dict], max_sentences: int = 3, max_chars: int = 600) -> str:
    """
    Build a concise, question-focused answer from top chunks.
    1) Split top chunks into sentences.
    2) Score by query overlap (+ slight brevity preference).
    3) Return top-N sentences stitched neatly.
    """
    qtoks = set(_simple_tokenize(query))
    candidates: List[Tuple[float, str]] = []
    for h in hits[:5]:  # examine top few chunks
        for s in _split_sentences(h.get("snippet", "")):
            score = _score_sentence(qtoks, s)
            if score > 0:
                candidates.append((score, s))
    if not candidates:
        return "I cannot find this in the documents."

    # Highest first, keep unique-ish sentences
    candidates.sort(key=lambda x: x[0], reverse=True)
    picked: List[str] = []
    seen = set()
    for _, s in candidates:
        key = s.lower()
        if key in seen:
            continue
        picked.append(s)
        seen.add(key)
        if len(picked) >= max_sentences:
            break

    ans = " ".join(picked).strip()
    if len(ans) > max_chars:
        ans = ans[: max_chars - 1].rstrip() + "…"
    return ans or "I cannot find this in the documents."

# ----------------------------
# Initialization
# ----------------------------
def _init_openai_mode(docs: List[Tuple[str, str]]) -> None:
    """
    Build embeddings and a vector store once.
    Falls back to in-memory list if FAISS not available.
    """
    global _embeddings, _vectorstore
    chunks = []
    for path, content in docs:
        content = _strip_ignored_sections(content)
        for d in _TEXT_SPLITTER.create_documents([content], metadatas=[{"source": path}]):
            chunks.append(d)

    _embeddings = OpenAIEmbeddings()

    if 'FAISS' in globals() and _FAISS_OK:
        _vectorstore = FAISS.from_documents(chunks, _embeddings)
    else:
        # Degrade to manual scoring over a list of LangChain Documents
        _vectorstore = chunks  # type: ignore

def _init_free_mode(docs: List[Tuple[str, str]]) -> None:
    """
    Prepare chunks for naive retrieval. Zero external services, zero cost.
    """
    global _free_chunks
    _free_chunks = []
    for path, content in docs:
        content = _strip_ignored_sections(content)
        for d in _TEXT_SPLITTER.create_documents([content], metadatas=[{"source": path}]):
            _free_chunks.append((path, d.page_content))

def _top_k_from_vectorstore(query: str, k: int) -> List[Dict]:
    """
    Retrieve top-k chunks with provenance.
    - FAISS path (if available in OpenAI mode)
    - Otherwise simple keyword overlap over in-memory chunks
    """
    results: List[Dict] = []

    if USE_OPENAI and _vectorstore is not None and 'FAISS' in globals() and _FAISS_OK and hasattr(_vectorstore, "similarity_search"):
        docs = _vectorstore.similarity_search(query, k=k)  # type: ignore
        for d in docs:
            results.append({
                "source": d.metadata.get("source"),
                "snippet": d.page_content[:300],
                "score": None,
            })
        return results

    # Fallback corpus (OpenAI without FAISS, or free mode)
    if USE_OPENAI and isinstance(_vectorstore, list):
        corpus = [(d.metadata.get("source"), d.page_content) for d in _vectorstore]  # type: ignore
    else:
        corpus = list(_free_chunks)

    scored = [(src, txt, _keyword_overlap_score(query, txt)) for (src, txt) in corpus]
    scored.sort(key=lambda x: x[2], reverse=True)
    for src, txt, sc in scored[:k]:
        results.append({"source": src, "snippet": txt[:300], "score": round(sc, 4)})
    return results

def init() -> None:
    """
    Idempotent index build:
      - reads documents (prefers data/my_document.txt if present),
      - builds vector store (OpenAI) or a free-mode chunk list.
    """
    global _INDEX_READY
    if _INDEX_READY:
        return

    docs = _read_all_text_files(_DOCS_DIR)
    if USE_OPENAI:
        _init_openai_mode(docs)
    else:
        _init_free_mode(docs)

    _INDEX_READY = True

# ----------------------------
# Query
# ----------------------------
async def get_rag_response(
    query: str,
    top_k: int = 4,
    trace: bool = False,
    request_id: Optional[str] = None,
) -> Dict:
    """
    Execute a RAG query.

    Returns:
      {
        "answer": str,
        "sources": [{"source": str, "snippet": str, "score": float|None}, ...],
        "latency_ms": int,
        "mode": "free" | "openai",
        "request_id": str | None
      }
    """
    t0 = time.perf_counter()
    if not _INDEX_READY:
        init()

    # 1) Retrieve
    top_k = max(1, min(10, int(top_k)))
    hits = _top_k_from_vectorstore(query, k=top_k)

    # 2) Compose context
    context_text = "\n\n".join([f"[{i+1}] {h['snippet']}" for i, h in enumerate(hits)])

    # 3) Generate
    if USE_OPENAI:
        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        system = (
            "You are a concise assistant. Answer ONLY using the provided context. "
            "If the answer is absent, say 'I cannot find this in the documents.'"
        )
        user = f"Context:\n{context_text}\n\nQuestion: {query}\n"
        msg = await llm.ainvoke([{"role": "system", "content": system},
                                 {"role": "user", "content": user}])
        answer = (msg.content or "").strip()
    else:
        # Free mode: question-focused, sentence-level extraction
        if not hits:
            answer = "I cannot find this in the documents."
        else:
            qlow = query.lower()
            want_more = any(w in qlow for w in ["list", "compare", "difference", "differences", "who", "what is", "what are"])
            answer = _compose_answer_free(query, hits, max_sentences=(4 if want_more else 3), max_chars=600)

    # 4) Trim internal fields unless tracing
    if not trace:
        for h in hits:
            h.pop("score", None)

    return {
        "answer": answer,
        "sources": hits,
        "latency_ms": int((time.perf_counter() - t0) * 1000),
        "mode": "openai" if USE_OPENAI else "free",
        "request_id": request_id,
    }
