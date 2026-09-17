from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from time import perf_counter

from src.utils.query_logger import (
    log_query,
)
from src.query.query_orchestrator import (
    QueryOrchestrator,
)


# =========================================================
# API models
# =========================================================

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Natural-language CHRIS query",
    )


class QueryResponse(BaseModel):
    route: str
    status: str
    query: str
    answer: str | None = None

    evidence: list[Dict[str, Any]] = Field(
        default_factory=list
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict
    )

# =========================================================
# Application
# =========================================================

app = FastAPI(
    title="CHRIS RAG API",
    description=(
        "Local API for querying CHRIS documents "
        "and structured GWAS datasets."
    ),
    version="0.1.0",
)


# Create only one orchestrator instance.
#
# We do NOT want to recreate retrieval services,
# embedding models, etc. for every HTTP request.
orchestrator = QueryOrchestrator()


# =========================================================
# Endpoints
# =========================================================

@app.get("/health")
def health():
    """
    Lightweight health check.
    """

    return {
        "status": "ok",
        "service": "chris-rag",
    }


@app.post(
    "/query",
    response_model=QueryResponse,
)
def query_chris(
    request: QueryRequest,
):
    """
    Route a natural-language question to either:

    - document RAG
    - GWAS structured query service
    """

    start = perf_counter()

    try:

        result = orchestrator.run(
            request.query
        )

        latency = (
            perf_counter()
            - start
        )

        log_query(
            query=request.query,
            route=result.get(
                "route",
                "unknown",
            ),
            status=result.get(
                "status",
                "unknown",
            ),
            answer=result.get(
                "answer"
            ),
            latency_seconds=latency,
        )

        return result

    except Exception as exc:

        latency = (
            perf_counter()
            - start
        )

        log_query(
            query=request.query,
            route="unknown",
            status="error",
            answer=None,
            latency_seconds=latency,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )