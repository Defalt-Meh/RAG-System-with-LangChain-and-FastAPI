"""
Endpoints for the RAG API.

Design goals:
- **Predictable contract**: Pydantic response model with explicit fields.
- **Input hygiene**: length limits and trimming; fast 4xx on bad inputs.
- **Operational clarity**: request_id for traceability; latency reported.
- **Graceful failure**: map common failures to meaningful HTTP status codes.
"""

from __future__ import annotations

import uuid
import time
from typing import Any, List, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from rag import get_rag_response  # Domain entrypoint; keep thin here.


# ---------------------------------------------------------------------------
# Pydantic models (API contract)
# ---------------------------------------------------------------------------

class SourceChunk(BaseModel):
    """
    Minimal provenance payload for each retrieved chunk.

    You can extend this as your RAG pipeline matures, e.g. add
    `doc_id`, `score`, `chunk_id`, or `page_number`.
    """
    source: Optional[str] = Field(None, description="Document identifier or path.")
    snippet: Optional[str] = Field(None, description="Short excerpt of the retrieved text.")
    score: Optional[float] = Field(None, description="Retriever similarity score (higher=closer).")


class QueryResponse(BaseModel):
    """
    Canonical success response for /query.

    Why this shape?
    - `query`: echo for debuggability and reproducibility.
    - `answer`: the generated text (LLM output).
    - `sources`: light provenance to build user trust.
    - `latency_ms`: basic SLO visibility without external tooling.
    - `request_id`: stable handle for correlating logs and bug reports.
    """
    request_id: str = Field(..., description="Server-generated UUID for this request.")
    query: str = Field(..., description="The normalized user query.")
    answer: str = Field(..., description="RAG-generated response.")
    sources: Optional[List[SourceChunk]] = Field(None, description="Top retrieved chunks used as context.")
    latency_ms: int = Field(..., description="End-to-end latency in milliseconds.")

    model_config = {
        "json_schema_extra": {
            "example": {
                "request_id": "c4a8c9d6-1f5a-47a3-8a32-2d0f2fc1b7a1",
                "query": "What do polar bears eat?",
                "answer": "Primarily seals, hunted from sea ice leads; they also scavenge and rarely eat vegetation.",
                "sources": [
                    {
                        "source": "my_document.txt",
                        "snippet": "…relying on sea ice to hunt seals, their primary source of food…",
                        "score": 0.82
                    }
                ],
                "latency_ms": 128
            }
        }
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="", tags=["RAG"])


@router.get(
    "/query/",
    response_model=QueryResponse,
    summary="Query the Retrieval-Augmented Generation (RAG) system",
    responses={
        400: {"description": "Invalid input"},
        422: {"description": "Validation error (query too short/long)"},
        429: {"description": "Upstream rate limiting"},
        500: {"description": "Unexpected server error"},
        503: {"description": "RAG pipeline not ready (e.g., index missing)"},
        504: {"description": "Upstream timeout"},
    },
)
async def query_rag_system(
    query: str = Query(
        ...,
        min_length=3,
        max_length=512,
        description="User question in plain text. Trimmed server-side."
    ),
    top_k: int = Query(
        4,
        ge=1,
        le=10,
        description="How many chunks to retrieve for context (the retriever may ignore if unsupported)."
    ),
    trace: bool = Query(
        False,
        description="If true, RAG may include richer provenance (depends on backend support)."
    ),
):
    """
    Execute a RAG query.

    Pipeline sketch:
    1) **Validate & normalize** input (defensive: never trust the caller).
    2) **Retrieve** top-k semantic matches from the vector index.
    3) **Augment & Generate** an answer using an LLM over retrieved context.
    4) **Return** answer + light provenance + operational metadata.

    Notes for maintainers:
    - This endpoint deliberately knows nothing about embeddings, indexes, or LLMs.
      That separation keeps the boundary clean and tests focused.
    - `top_k` and `trace` are *hints*; if your `get_rag_response` ignores them,
      that is acceptable. As the backend matures, wire them through.
    """
    request_id = str(uuid.uuid4())
    t0 = time.perf_counter()

    # --- Input hygiene: trim whitespace and revalidate length after trimming.
    q = query.strip()
    if len(q) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query too short after trimming; provide at least 3 characters."
        )

    try:
        # --- Delegate to domain layer.
        # Prefer keyword args; fall back gracefully if backend signature is older.
        try:
            result: Any = await get_rag_response(query=q, top_k=top_k, trace=trace, request_id=request_id)
        except TypeError:
            # Backwards compatibility: older `get_rag_response(query)` signature.
            result = await get_rag_response(q)

        # --- Normalize backend outputs to our contract.
        # Accept either a plain string or a dict with richer fields.
        answer: str
        sources: Optional[List[SourceChunk]] = None

        if isinstance(result, str):
            answer = result
        elif isinstance(result, dict):
            # Try common keys; be lenient to reduce coupling.
            answer = (
                result.get("answer")
                or result.get("response")
                or result.get("text")
                or ""
            )
            raw_sources = result.get("sources") or result.get("context") or []
            # Best-effort coercion to SourceChunk list.
            sources = [SourceChunk(**s) if isinstance(s, dict) else SourceChunk(snippet=str(s)) for s in raw_sources]
        else:
            # Last resort: stringify unknown outputs to avoid hard 500s.
            answer = str(result)

        if not answer:
            # Avoid returning empty strings; callers interpret as failure.
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="RAG backend returned no answer (index empty or model unavailable)."
            )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        return QueryResponse(
            request_id=request_id,
            query=q,
            answer=answer,
            sources=sources,
            latency_ms=latency_ms,
        )

    # --- Map common failure modes to clear HTTP semantics.
    except TimeoutError as e:
        raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=f"Upstream timeout: {e}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except HTTPException:
        # Already well-formed; just bubble up.
        raise
    except Exception as e:
        # Do not leak internals to clients; keep details terse but useful.
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG system failure (request_id={request_id})."
        ) from e
