"""FastAPI server for RAG Medical Chatbot"""

import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.utils.config_loader import load_config
from src.rag.pipeline import RAGPipeline

# ─── Config & App ────────────────────────────────────────────────────────────

config = load_config("config/rag_config.yaml")
app = FastAPI(
    title="Medical Chatbot RAG API",
    description="API cho Chatbot Y tế sử dụng RAG pipeline",
    version="1.0.0"
)

# CORS
cors_cfg = config.get("api", {}).get("cors_origins", ["*"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_cfg,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Pipeline ──────────────────────────────────────────────────────────

pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    global pipeline
    if pipeline is None:
        pipeline = RAGPipeline(config)
    return pipeline


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Câu hỏi của người dùng")
    top_k: Optional[int] = Field(5, ge=1, le=20, description="Số tài liệu cần retrieve")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    include_sources: bool = Field(True, description="Bao gồm nguồn trích dẫn")


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    latency_ms: float
    model: str
    top_k: int


class HealthResponse(BaseModel):
    status: str
    components: dict
    stats: dict


class IngestRequest(BaseModel):
    data_path: Optional[str] = "data/processed/data.json"
    rebuild: bool = False
    batch_size: int = Field(100, ge=1, le=1000)


class IngestResponse(BaseModel):
    success: bool
    stats: dict


# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "name": "Medical Chatbot RAG API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Check health of all pipeline components."""
    try:
        pl = get_pipeline()
        pl._lazy_init()
        health_status = pl.check_health()
        return HealthResponse(
            status=health_status["status"],
            components=health_status["components"],
            stats=pl.get_stats(),
        )
    except Exception:
        import traceback as tb_module
        tb_module.print_exc()
        raise HTTPException(status_code=500, detail=tb_module.format_exc())


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Gửi câu hỏi và nhận câu trả lời từ chatbot.
    """
    try:
        pl = get_pipeline()
        pl._lazy_init()

        if request.temperature is not None:
            pl.generator.temperature = request.temperature
        else:
            pl.generator.temperature = pl._generation_config.get("temperature", 0.3)

        response = pl.query(
            question=request.message,
            top_k=request.top_k,
            include_sources=request.include_sources,
        )

        return ChatResponse(
            answer=response.answer,
            sources=[
                {
                    "question": s.get("question", ""),
                    "score": s.get("score", 0),
                    "id": s.get("id", ""),
                }
                for s in response.sources
            ],
            latency_ms=round(response.latency * 1000, 2),
            model=response.generation_config.get("model", ""),
            top_k=response.generation_config.get("top_k", request.top_k),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream", tags=["Chat"])
async def chat_stream(request: ChatRequest):
    """
    Gửi câu hỏi và nhận câu trả lời dạng streaming.
    """
    try:
        pl = get_pipeline()
        pl._lazy_init()

        retrieved_docs = pl.retriever.retrieve(request.message, top_k=request.top_k)

        if not retrieved_docs:
            async def empty():
                yield "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
            return StreamingResponse(empty(), media_type="text/plain")

        context_result = pl.retriever.get_context_with_citations(
            request.message, top_k=request.top_k
        )

        system_prompt = pl._prompt_config.get("system", "")
        user_template = pl._prompt_config.get("user_template", "")
        prompt = user_template.format(
            question=request.message,
            context=context_result["context"]
        )

        def generate():
            full_text = []
            def callback(chunk):
                full_text.append(chunk)
                yield chunk

            pl.generator.generate_streaming(
                prompt=prompt,
                system_prompt=system_prompt,
                callback=callback,
            )

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={"X-Accel-Buffering": "no"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse, tags=["Admin"])
async def ingest(request: IngestRequest):
    """
    Ingest dữ liệu vào vector store.
    """
    try:
        pl = get_pipeline()
        stats = pl.ingest(
            data_path=request.data_path,
            batch_size=request.batch_size,
            rebuild=request.rebuild,
            show_progress=False,
        )
        return IngestResponse(success=True, stats=stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", tags=["System"])
async def stats():
    """Get pipeline statistics."""
    try:
        pl = get_pipeline()
        pl._lazy_init()
        return pl.get_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run():
    host = config.get("api", {}).get("host", "0.0.0.0")
    port = config.get("api", {}).get("port", 8000)
    reload = config.get("api", {}).get("reload", True)

    uvicorn.run(
        "src.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )


if __name__ == "__main__":
    run()
