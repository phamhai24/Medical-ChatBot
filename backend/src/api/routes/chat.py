"""Chat endpoints - ask and stream."""

import time
import uuid
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.api.deps import get_pipeline
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
    ChatStreamRequest,
)
from src.core.logging import logger
from src.core.metrics import track_rag_metrics
from src.core.redis_client import (
    generate_session_id,
    get_chat_history,
    save_chat_message,
)

router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])


@router.post("/ask", response_model=ChatResponse)
def ask(request: ChatRequest):
    """Gửi câu hỏi và nhận câu trả lời từ chatbot."""
    pipeline = get_pipeline()
    pipeline._lazy_init()

    session_id = request.session_id or generate_session_id()

    start_time = time.perf_counter()
    retrieval_time = 0
    generation_time = 0
    status = "success"

    try:
        response = pipeline.query(
            question=request.message,
            top_k=request.top_k,
            include_sources=request.include_sources,
        )
        retrieved_docs = response.retrieved_docs
        retrieval_time = response.retrieval_latency
        generation_time = response.generation_latency

        if not retrieved_docs:
            status = "no_results"
            latency = time.perf_counter() - start_time
            answer = (
                "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong cơ sở dữ liệu "
                "để trả lời câu hỏi này. Vui lòng thử diễn đạt câu hỏi theo cách khác "
                "hoặc liên hệ chuyên gia y tế."
            )
            return ChatResponse(
                answer=answer,
                sources=[],
                latency_ms=round(latency * 1000, 2),
                model=pipeline._generation_config.get("model_name", ""),
                top_k=request.top_k,
                session_id=session_id,
                generated_at=datetime.utcnow(),
            )

        # Track metrics
        latency = response.latency or (time.perf_counter() - start_time)
        track_rag_metrics(
            status=status,
            retrieval_latency=retrieval_time,
            generation_latency=generation_time,
            tokens=len(response.answer.split()),
            sources_count=len(response.sources),
            generator_mode=pipeline._generation_config.get("mode", "local"),
            search_type=pipeline._retrieval_config.get("search_type", "similarity"),
        )

        # Save to session
        save_chat_message(session_id, "user", request.message)
        save_chat_message(session_id, "assistant", response.answer, {
            "sources": len(response.sources),
            "latency_ms": round(latency * 1000, 2),
        })

        return ChatResponse(
            answer=response.answer,
            sources=response.sources,
            latency_ms=round(latency * 1000, 2),
            model=pipeline._generation_config.get("model_name", ""),
            top_k=request.top_k,
            session_id=session_id,
            generated_at=datetime.utcnow(),
        )

    except Exception as e:
        logger.exception(f"Chat error: {e}")
        track_rag_metrics(
            status="error",
            retrieval_latency=retrieval_time,
            generation_latency=generation_time,
            tokens=0,
            sources_count=0,
            generator_mode="local",
            search_type="similarity",
        )
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
def ask_stream(request: ChatStreamRequest):
    """Gửi câu hỏi và nhận câu trả lời dạng streaming."""
    pipeline = get_pipeline()
    pipeline._lazy_init()

    session_id = request.session_id or generate_session_id()
    full_answer = []

    try:
        retrieved_docs = pipeline.retriever.retrieve(request.message, top_k=request.top_k)

        if not retrieved_docs:
            def empty():
                yield "Xin lỗi, tôi không tìm thấy thông tin phù hợp."
            return StreamingResponse(empty(), media_type="text/plain")

        context_result = pipeline._format_context_with_citations(retrieved_docs)
        context = context_result["context"]

        system_prompt = pipeline._prompt_config.get("system", "")
        user_template = pipeline._prompt_config.get("user_template", "")
        prompt = user_template.format(question=request.message, context=context)

        def generate():
            full = []
            def callback(chunk):
                full.append(chunk)
                yield chunk

            pipeline.generator.generate_streaming(
                prompt=prompt,
                system_prompt=system_prompt,
                callback=callback,
            )

            # Save after streaming completes
            save_chat_message(session_id, "user", request.message)
            save_chat_message(session_id, "assistant", "".join(full), {
                "sources": len(context_result["sources"]),
            })

        return StreamingResponse(
            generate(),
            media_type="text/plain",
            headers={
                "X-Accel-Buffering": "no",
                "X-Session-ID": session_id,
            }
        )

    except Exception as e:
        logger.exception(f"Stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
