"""API route handlers for /v1/* endpoints.

Wires the subsystem services to HTTP endpoints:
- POST /v1/search
- POST /v1/find_similar
- POST /v1/contents
- POST /v1/answer
- POST /v1/research
- GET  /v1/research/{job_id}
- GET  /v1/research/{job_id}/events
- POST /v1/sessions
- DELETE /v1/sessions/{session_id}
- POST /v1/pipelines
- GET  /v1/pipelines/{pipeline_id}
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.api_gateway.validation import (
    ContentsRequest,
    FindSimilarRequest,
    RequestValidationError,
    ResearchRequest,
    SearchRequest,
    validate_contents_request,
)


# ---------------------------------------------------------------------------
# Pydantic models for Swagger documentation
# ---------------------------------------------------------------------------


class IndexBody(BaseModel):
    """Request body for POST /v1/index."""
    url: str = Field(..., description="Source URL of the document")
    content: str = Field(..., description="Raw HTML or text content to index")


class SearchBody(BaseModel):
    """Request body for POST /v1/search."""
    query: str = Field(..., description="Search query (1-2048 chars)")
    mode: str = Field(default="hybrid", description="Search mode: neural, keyword, or hybrid")
    num_results: int = Field(default=10, description="Max results to return (0-100)")
    filters: str | None = Field(default=None, description="Filter DSL expression")
    pipeline_id: str | None = Field(default=None, description="Pipeline ID to apply")
    min_credibility: float | None = Field(default=None, description="Min credibility threshold [0.0-1.0]")
    max_ai_generated_likelihood: float | None = Field(default=None, description="Max AI-generated threshold [0.0-1.0]")


class FindSimilarBody(BaseModel):
    """Request body for POST /v1/find_similar."""
    url: str = Field(..., description="URL to find similar documents for")
    num_results: int = Field(default=10, description="Max results (0-100)")
    filters: str | None = Field(default=None, description="Filter DSL expression")


class ContentsBody(BaseModel):
    """Request body for POST /v1/contents."""
    document_ids: list[str] = Field(..., description="List of 1-100 document IDs")
    highlights: bool | None = Field(default=None, description="Include highlight spans (requires query)")
    query: str | None = Field(default=None, description="Query for highlight extraction")
    summary: bool | None = Field(default=None, description="Include document summaries")


class AnswerBody(BaseModel):
    """Request body for POST /v1/answer."""
    query: str = Field(..., description="Question to answer")
    stream: bool = Field(default=True, description="Enable streaming (SSE)")
    session_id: str | None = Field(default=None, description="Session ID for context")


class ResearchBody(BaseModel):
    """Request body for POST /v1/research."""
    research_goal: str = Field(..., description="Research goal (1-4096 chars)")
    output_schema: dict[str, Any] | None = Field(default=None, description="JSON Schema for structured output")
    session_id: str | None = Field(default=None, description="Session ID for context")
    max_steps: int | None = Field(default=None, description="Max research steps")
    max_duration_ms: int | None = Field(default=None, description="Max duration in ms")
    max_tool_calls: int | None = Field(default=None, description="Max tool calls")


class SessionBody(BaseModel):
    """Request body for POST /v1/sessions."""
    retention_days: int | None = Field(default=14, description="Days to retain session [1-90]")


class PipelineStepBody(BaseModel):
    """A single pipeline step."""
    name: str = Field(..., description="Step name from registry")
    config: dict[str, Any] | None = Field(default=None, description="Step configuration")
    timeout_ms: int | None = Field(default=2000, description="Step timeout [100-30000]ms")


class PipelineBody(BaseModel):
    """Request body for POST /v1/pipelines."""
    name: str = Field(..., description="Pipeline name")
    steps: list[PipelineStepBody] = Field(..., description="Pipeline steps (1-20)")

router = APIRouter(prefix="/v1")

# ---------------------------------------------------------------------------
# In-memory service singletons (initialized lazily on first request)
# In production these would be injected via dependency injection.
# ---------------------------------------------------------------------------

_services: dict[str, Any] = {}


def _get_services() -> dict[str, Any]:
    """Lazily initialize service singletons."""
    if not _services:
        import os

        from backend.indexer import IndexerService
        from backend.retriever.service import RetrieverService
        from backend.answer_engine.service import AnswerEngine
        from backend.answer_engine.provider import OpenAIProvider, MockLLMProvider, ProviderConfig
        from backend.research_agent.service import ResearchAgentService
        from backend.session_store.service import SessionService
        from backend.pipeline_engine.service import PipelineService
        from backend.api_gateway.contents import ContentsService

        indexer = IndexerService()
        retriever = RetrieverService(indexer=indexer)

        # Use real OpenAI provider if API key is set, otherwise mock
        if os.environ.get("OPENAI_API_KEY"):
            provider = OpenAIProvider(ProviderConfig(model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini")))
        else:
            provider = MockLLMProvider()

        answer_engine = AnswerEngine(provider=provider)
        research_agent = ResearchAgentService(retriever=retriever)
        session_service = SessionService()
        pipeline_service = PipelineService()
        contents_service = ContentsService(document_store=indexer)

        # Kafka ingest producer (optional)
        kafka_producer = None
        try:
            from backend.indexer.kafka_ingest import KafkaIngestProducer
            kafka_producer = KafkaIngestProducer()
        except Exception:
            pass

        _services["indexer"] = indexer
        _services["retriever"] = retriever
        _services["answer_engine"] = answer_engine
        _services["research_agent"] = research_agent
        _services["session_service"] = session_service
        _services["pipeline_service"] = pipeline_service
        _services["contents_service"] = contents_service
        _services["kafka_producer"] = kafka_producer

        # Load persisted documents from PostgreSQL on startup
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.ensure_future(_load_persisted_documents(indexer))
            else:
                loop.run_until_complete(_load_persisted_documents(indexer))
        except Exception:
            pass

    return _services


def _get_tenant_id(request: Request) -> str:
    """Extract tenant_id from request state (set by auth middleware)."""
    return getattr(request.state, "tenant_id", "test-tenant")


async def _load_persisted_documents(indexer) -> None:
    """Load documents from PostgreSQL into the in-memory indexer on startup."""
    import os
    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        return

    try:
        import asyncpg
        conn = await asyncpg.connect(database_url)
        rows = await conn.fetch("SELECT document_id, source_url, version, cleaned_text FROM documents WHERE visible = TRUE")
        await conn.close()

        for row in rows:
            # Inject directly into the indexer's in-memory store
            from backend.indexer.service import DocumentVersion
            from backend.indexer.embeddings import generate_embedding
            from datetime import datetime, timezone

            doc_version = DocumentVersion(
                document_id=row["document_id"],
                version=row["version"],
                content_hash="loaded",
                cleaned_text=row["cleaned_text"],
                source_url=row["source_url"],
                last_seen_at=datetime.now(timezone.utc),
                created_at=datetime.now(timezone.utc),
                visible=True,
            )

            indexer._url_to_doc_id[row["source_url"]] = row["document_id"]
            indexer._documents[row["document_id"]] = [doc_version]

            # Write to in-memory indexes for search
            embedding = generate_embedding(row["cleaned_text"])
            indexer._vector_index.write(row["document_id"], row["version"], embedding)
            indexer._lexical_index.write(row["document_id"], row["version"], row["cleaned_text"])

    except Exception:
        pass


# ---------------------------------------------------------------------------
# POST /v1/index (convenience endpoint for local dev)
# ---------------------------------------------------------------------------


@router.post("/index")
async def index_document(body: IndexBody, request: Request) -> JSONResponse:
    """Index a document (supports async Kafka ingest when available)."""
    if not body.url or not body.content:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "url and content are required"}},
        )

    services = _get_services()
    indexer = services["indexer"]

    # Try Kafka async ingest first
    kafka_producer = services.get("kafka_producer")
    if kafka_producer and kafka_producer.is_available:
        tenant_id = _get_tenant_id(request)
        message_id = await kafka_producer.publish(
            source_url=body.url,
            raw_content=body.content,
            tenant_id=tenant_id,
        )
        return JSONResponse(
            status_code=202,
            content={"message_id": message_id, "status": "queued"},
        )

    # Synchronous indexing
    result = await indexer.index_document(body.content, body.url)

    # Persist to PostgreSQL if available
    try:
        await _persist_document(result.document_id, body.url, result.version, body.content)
    except Exception:
        pass  # DB persistence failure is non-fatal

    return JSONResponse(
        status_code=201,
        content={
            "document_id": result.document_id,
            "version": result.version,
            "is_new": result.is_new,
        },
    )


async def _persist_document(document_id: str, source_url: str, version: int, content: str) -> None:
    """Persist a document to PostgreSQL for durability."""
    import os
    database_url = os.environ.get("DATABASE_URL", "")
    if not database_url:
        return

    try:
        import asyncpg
        conn = await asyncpg.connect(database_url)
        from backend.indexer.cleaner import clean_html
        from backend.indexer.hasher import compute_content_hash

        cleaned = clean_html(content)
        content_hash = compute_content_hash(cleaned)

        await conn.execute("""
            INSERT INTO documents (document_id, source_url, version, content_hash, cleaned_text)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (source_url) DO UPDATE SET
                version = documents.version + 1,
                content_hash = $4,
                cleaned_text = $5,
                updated_at = NOW()
        """, document_id, source_url, version, content_hash, cleaned)
        await conn.close()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# POST /v1/search
# ---------------------------------------------------------------------------


@router.post("/search")
async def search(body: SearchBody, request: Request) -> JSONResponse:
    """Neural, keyword, or hybrid search."""
    # Validate using the strict validator
    try:
        validated = SearchRequest(**body.model_dump())
    except RequestValidationError as e:
        return JSONResponse(status_code=422, content=e.to_response_body())
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": {"code": "validation_error", "message": str(e)}})

    services = _get_services()
    retriever = services["retriever"]
    tenant_id = _get_tenant_id(request)

    from backend.retriever.models import SearchRequest as RetrieverSearchRequest, SearchMode

    search_request = RetrieverSearchRequest(
        query=validated.query,
        mode=SearchMode(validated.mode),
        num_results=validated.num_results,
        filters=validated.filters,
        pipeline_id=validated.pipeline_id,
        min_credibility=validated.min_credibility,
        max_ai_generated_likelihood=validated.max_ai_generated_likelihood,
        tenant_id=tenant_id,
    )

    response = retriever.search(search_request)

    results = [
        {
            "document_id": r.document_id,
            "url": r.url,
            "title": r.title,
            "score": r.score,
            "published_at": r.published_at.isoformat() if r.published_at else None,
            "provenance": {
                "credibility_score": r.provenance.credibility_score,
                "ai_generated_likelihood": r.provenance.ai_generated_likelihood,
            },
            "version": r.version,
        }
        for r in response.results
    ]

    return JSONResponse(
        status_code=200,
        content={"results": results, "total": len(results)},
        headers={"X-Index-Version": str(response.index_version)},
    )


# ---------------------------------------------------------------------------
# POST /v1/find_similar
# ---------------------------------------------------------------------------


@router.post("/find_similar")
async def find_similar(body: FindSimilarBody, request: Request) -> JSONResponse:
    """Find documents similar to a URL."""
    try:
        validated = FindSimilarRequest(**body.model_dump())
    except RequestValidationError as e:
        return JSONResponse(status_code=422, content=e.to_response_body())
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": {"code": "validation_error", "message": str(e)}})

    services = _get_services()
    retriever = services["retriever"]
    tenant_id = _get_tenant_id(request)

    from backend.retriever.models import FindSimilarRequest as RetrieverFindSimilarRequest
    from backend.retriever.service import DocumentNotFoundError

    find_request = RetrieverFindSimilarRequest(
        url=validated.url,
        num_results=validated.num_results,
        filters=validated.filters,
        tenant_id=tenant_id,
    )

    try:
        response = retriever.find_similar(find_request)
    except DocumentNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "document_not_found", "message": "URL not found in index"}},
        )

    results = [
        {
            "document_id": r.document_id,
            "url": r.url,
            "title": r.title,
            "score": r.score,
            "provenance": {
                "credibility_score": r.provenance.credibility_score,
                "ai_generated_likelihood": r.provenance.ai_generated_likelihood,
            },
            "version": r.version,
        }
        for r in response.results
    ]

    return JSONResponse(status_code=200, content={"results": results, "total": len(results)})


# ---------------------------------------------------------------------------
# POST /v1/contents
# ---------------------------------------------------------------------------


@router.post("/contents")
async def contents(body: ContentsBody, request: Request) -> JSONResponse:
    """Batch fetch cleaned text, highlights, summaries."""
    try:
        validated = ContentsRequest(**body.model_dump())
    except RequestValidationError as e:
        return JSONResponse(status_code=422, content=e.to_response_body())
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": {"code": "validation_error", "message": str(e)}})

    # Cross-field validation
    try:
        validate_contents_request(validated)
    except RequestValidationError as e:
        return JSONResponse(status_code=422, content=e.to_response_body())

    services = _get_services()
    contents_service = services["contents_service"]

    response = contents_service.fetch_contents(
        validated.document_ids,
        highlights=validated.highlights,
        query=validated.query,
        summary=validated.summary,
    )

    return JSONResponse(status_code=200, content={"results": contents_service.to_response_dict(response)})


# ---------------------------------------------------------------------------
# POST /v1/answer
# ---------------------------------------------------------------------------


@router.post("/answer")
async def answer(body: AnswerBody, request: Request) -> JSONResponse:
    """Streaming answers with citations.

    Returns SSE stream if Accept: text/event-stream, otherwise JSON.
    """
    if not body.query or not body.query.strip():
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_query", "message": "query is required"}},
        )

    services = _get_services()
    retriever = services["retriever"]
    answer_engine = services["answer_engine"]
    tenant_id = _get_tenant_id(request)

    # First, retrieve relevant documents
    from backend.retriever.models import SearchRequest as RetrieverSearchRequest, SearchMode
    from backend.answer_engine.models import RetrievalResult, TokenEvent, CitationEvent, DoneEvent, ErrorEvent

    search_request = RetrieverSearchRequest(
        query=body.query,
        mode=SearchMode.HYBRID,
        num_results=5,
        tenant_id=tenant_id,
    )
    search_response = retriever.search(search_request)

    # Convert search results to RetrievalResult format
    retrieval_results = []
    indexer = services["indexer"]
    for r in search_response.results:
        doc = indexer.get_latest_version(r.document_id)
        if doc:
            retrieval_results.append(RetrievalResult(
                document_id=r.document_id,
                version=r.version,
                url=r.url,
                title=r.title,
                score=r.score,
                cleaned_text=doc.cleaned_text,
            ))

    # Check if client wants SSE streaming
    accept = request.headers.get("accept", "")
    if "text/event-stream" in accept:
        from backend.api_gateway.sse import SSEEvent, create_sse_response

        async def answer_sse_factory(last_event_id: str | None):
            event_id = 0
            async for event in answer_engine.generate_answer(body.query, retrieval_results, body.session_id):
                event_id += 1
                if isinstance(event, TokenEvent):
                    yield SSEEvent(event="token", data={"text": event.text, "index": event.index}, id=str(event_id))
                elif isinstance(event, CitationEvent):
                    yield SSEEvent(event="citation", data={
                        "document_id": event.document_id, "version": event.version,
                        "answer_start": event.answer_start, "answer_end": event.answer_end,
                        "source_start": event.source_start, "source_end": event.source_end,
                    }, id=str(event_id))
                elif isinstance(event, DoneEvent):
                    yield SSEEvent(event="done", data={"answer": event.answer, "citations": [
                        {"document_id": c.document_id, "version": c.version,
                         "answer_start": c.answer_start, "answer_end": c.answer_end}
                        for c in event.citations
                    ]}, id=str(event_id))
                elif isinstance(event, ErrorEvent):
                    yield SSEEvent(event="error", data={"code": event.code.value, "message": event.message}, id=str(event_id))

        return create_sse_response(request, answer_sse_factory)

    # Non-streaming JSON response: collect all events
    full_answer = ""
    citations = []
    error = None

    async for event in answer_engine.generate_answer(body.query, retrieval_results, body.session_id):
        if isinstance(event, TokenEvent):
            full_answer += event.text
        elif isinstance(event, CitationEvent):
            citations.append({
                "document_id": event.document_id,
                "version": event.version,
                "answer_start": event.answer_start,
                "answer_end": event.answer_end,
                "source_start": event.source_start,
                "source_end": event.source_end,
            })
        elif isinstance(event, DoneEvent):
            full_answer = event.answer
            citations = [
                {"document_id": c.document_id, "version": c.version,
                 "answer_start": c.answer_start, "answer_end": c.answer_end}
                for c in event.citations
            ]
        elif isinstance(event, ErrorEvent):
            error = {"code": event.code.value, "message": event.message}

    if error:
        return JSONResponse(status_code=200, content={"error": error})

    return JSONResponse(
        status_code=200,
        content={"answer": full_answer, "citations": citations},
    )


# ---------------------------------------------------------------------------
# POST /v1/research
# ---------------------------------------------------------------------------


@router.post("/research")
async def create_research(body: ResearchBody, request: Request) -> JSONResponse:
    """Launch a multi-hop research job."""
    try:
        validated = ResearchRequest(**body.model_dump())
    except RequestValidationError as e:
        return JSONResponse(status_code=422, content=e.to_response_body())
    except Exception as e:
        return JSONResponse(status_code=422, content={"error": {"code": "validation_error", "message": str(e)}})

    services = _get_services()
    research_agent = services["research_agent"]
    tenant_id = _get_tenant_id(request)

    from backend.research_agent.models import BudgetConfig
    from backend.research_agent.service import InvalidResearchRequestError

    budgets = BudgetConfig(
        max_steps=validated.max_steps or 32,
        max_duration_ms=validated.max_duration_ms or 300_000,
        max_tool_calls=validated.max_tool_calls or 100,
    )

    try:
        job_id = research_agent.start_job(
            tenant_id=tenant_id,
            research_goal=validated.research_goal,
            output_schema=validated.output_schema,
            session_id=validated.session_id,
            budgets=budgets,
        )
    except InvalidResearchRequestError as e:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_research_request", "message": e.message}},
        )

    return JSONResponse(status_code=201, content={"job_id": job_id})


# ---------------------------------------------------------------------------
# GET /v1/research/{job_id}
# ---------------------------------------------------------------------------


@router.get("/research/{job_id}")
async def get_research(job_id: str, request: Request) -> JSONResponse:
    """Get research job report."""
    services = _get_services()
    research_agent = services["research_agent"]
    tenant_id = _get_tenant_id(request)

    from backend.research_agent.service import JobNotFoundError

    try:
        report = research_agent.get_report(job_id, tenant_id)
    except JobNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "job_not_found", "message": "Research job not found"}},
        )

    return JSONResponse(
        status_code=200,
        content={
            "job_id": report.job_id,
            "text": report.text,
            "structured_payload": report.structured_payload,
            "citations": [
                {
                    "citation_id": c.citation_id,
                    "document_id": c.document_id,
                    "version": c.version,
                    "answer_start": c.answer_start,
                    "answer_end": c.answer_end,
                    "source_start": c.source_start,
                    "source_end": c.source_end,
                }
                for c in report.citations
            ],
            "is_partial": report.is_partial,
        },
    )


# ---------------------------------------------------------------------------
# GET /v1/research/{job_id}/events
# ---------------------------------------------------------------------------


@router.get("/research/{job_id}/events")
async def get_research_events(job_id: str, request: Request) -> JSONResponse:
    """Get research job events (simplified JSON response, SSE in production)."""
    services = _get_services()
    research_agent = services["research_agent"]
    tenant_id = _get_tenant_id(request)

    from backend.research_agent.service import JobNotFoundError

    last_event_id = request.headers.get("Last-Event-ID")
    last_id = int(last_event_id) if last_event_id else None

    try:
        events = research_agent.get_events(job_id, tenant_id, last_event_id=last_id)
    except JobNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "job_not_found", "message": "Research job not found"}},
        )

    return JSONResponse(
        status_code=200,
        content={
            "events": [
                {
                    "event_id": e.event_id,
                    "type": e.type.value,
                    "payload": e.payload,
                    "emitted_at": e.emitted_at.isoformat(),
                }
                for e in events
            ],
        },
    )


# ---------------------------------------------------------------------------
# POST /v1/sessions
# ---------------------------------------------------------------------------


@router.post("/sessions")
async def create_session(body: SessionBody, request: Request) -> JSONResponse:
    """Create a persistent research session."""
    services = _get_services()
    session_service = services["session_service"]
    tenant_id = _get_tenant_id(request)

    from backend.session_store.service import InvalidSessionRequestError

    try:
        session = session_service.create(tenant_id=tenant_id, retention_days=body.retention_days)
    except InvalidSessionRequestError as e:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_session_request", "message": e.message}},
        )

    return JSONResponse(
        status_code=201,
        content={
            "session_id": session.session_id,
            "tenant_id": session.tenant_id,
            "retention_days": session.retention_days,
            "expires_at": session.expires_at.isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# DELETE /v1/sessions/{session_id}
# ---------------------------------------------------------------------------


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request) -> JSONResponse:
    """Delete a session."""
    services = _get_services()
    session_service = services["session_service"]
    tenant_id = _get_tenant_id(request)

    from backend.session_store.service import SessionNotFoundError

    try:
        session_service.delete_session(session_id, tenant_id)
    except SessionNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "session_not_found", "message": "Session not found"}},
        )

    return JSONResponse(status_code=204, content=None)


# ---------------------------------------------------------------------------
# POST /v1/pipelines
# ---------------------------------------------------------------------------


@router.post("/pipelines")
async def create_pipeline(body: PipelineBody, request: Request) -> JSONResponse:
    """Create a retrieval pipeline."""
    if not body.name:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "name is required"}},
        )

    services = _get_services()
    pipeline_service = services["pipeline_service"]
    tenant_id = _get_tenant_id(request)

    from backend.pipeline_engine.service import InvalidPipelineError, UnknownPipelineStepError

    steps = [s.model_dump() for s in body.steps]

    try:
        pipeline = pipeline_service.create_pipeline(tenant_id=tenant_id, name=body.name, steps=steps)
    except InvalidPipelineError as e:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "invalid_pipeline", "message": str(e)}},
        )
    except UnknownPipelineStepError as e:
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "unknown_pipeline_step", "message": str(e)}},
        )

    return JSONResponse(
        status_code=201,
        content={
            "pipeline_id": pipeline.pipeline_id,
            "name": pipeline.name,
            "steps": [
                {"name": s.name, "type": s.type.value, "timeout_ms": s.timeout_ms}
                for s in pipeline.steps
            ],
            "created_at": pipeline.created_at.isoformat(),
        },
    )


# ---------------------------------------------------------------------------
# GET /v1/pipelines/{pipeline_id}
# ---------------------------------------------------------------------------


@router.get("/pipelines/{pipeline_id}")
async def get_pipeline(pipeline_id: str, request: Request) -> JSONResponse:
    """Get a pipeline by ID."""
    services = _get_services()
    pipeline_service = services["pipeline_service"]
    tenant_id = _get_tenant_id(request)

    from backend.pipeline_engine.service import PipelineNotFoundError

    try:
        pipeline = pipeline_service.get_pipeline(pipeline_id, tenant_id)
    except PipelineNotFoundError:
        return JSONResponse(
            status_code=404,
            content={"error": {"code": "pipeline_not_found", "message": "Pipeline not found"}},
        )

    return JSONResponse(
        status_code=200,
        content={
            "pipeline_id": pipeline.pipeline_id,
            "name": pipeline.name,
            "steps": [
                {"name": s.name, "type": s.type.value, "config": s.config, "timeout_ms": s.timeout_ms}
                for s in pipeline.steps
            ],
            "created_at": pipeline.created_at.isoformat(),
        },
    )
