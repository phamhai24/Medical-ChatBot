"""Chat endpoints - ask and stream."""

import json
import queue
import threading
import time
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
    """Gửi câu hỏi và nhận câu trả lời dạng streaming (NDJSON: chunk/sources/done/error lines)."""
    pipeline = get_pipeline()
    pipeline._lazy_init()

    session_id = request.session_id or generate_session_id()

    def _line(payload: dict) -> str:
        return json.dumps(payload, ensure_ascii=False) + "\n"

    try:
        retrieved_docs = pipeline.retriever.retrieve(request.message, top_k=request.top_k)

        if not retrieved_docs:
            def empty():
                yield _line({"type": "chunk", "text": "Xin lỗi, tôi không tìm thấy thông tin phù hợp."})
                yield _line({"type": "sources", "sources": []})
                yield _line({"type": "done", "session_id": session_id})

            return StreamingResponse(
                empty(),
                media_type="application/x-ndjson",
                headers={"X-Session-ID": session_id},
            )

        context_result = pipeline._format_context_with_citations(retrieved_docs)
        context = context_result["context"]

        system_prompt = pipeline._prompt_config.get("system", "")
        user_template = pipeline._prompt_config.get("user_template", "")
        prompt = user_template.format(question=request.message, context=context)

        def generate():
            # generate_streaming() calls its callback synchronously from inside a
            # blocking network call, so a background thread + queue is the
            # standard bridge to turn that into an actual incremental generator
            # instead of collecting everything before yielding anything.
            chunk_queue: "queue.Queue" = queue.Queue()
            full: list[str] = []
            errors: list[Exception] = []

            def callback(chunk: str) -> None:
                full.append(chunk)
                chunk_queue.put(chunk)

            def run() -> None:
                try:
                    pipeline.generator.generate_streaming(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        callback=callback,
                    )
                except Exception as exc:  # noqa: BLE001 - surfaced to the client below
                    errors.append(exc)
                finally:
                    chunk_queue.put(None)  # sentinel: generation finished

            thread = threading.Thread(target=run, daemon=True)
            thread.start()

            while True:
                item = chunk_queue.get()
                if item is None:
                    break
                yield _line({"type": "chunk", "text": item})

            thread.join()

            if errors:
                logger.exception(f"Stream generation error: {errors[0]}")
                yield _line({"type": "error", "message": str(errors[0])})
                return

            answer = "".join(full)
            cited_questions = pipeline._extract_cited_questions(
                answer, context_result["position_to_question"]
            )
            sources = pipeline._filter_sources_for_display(
                answer, context_result["sources"], cited_questions
            )

            yield _line({"type": "sources", "sources": sources})
            yield _line({"type": "done", "session_id": session_id})

            save_chat_message(session_id, "user", request.message)
            save_chat_message(session_id, "assistant", answer, {
                "sources": len(sources),
            })

        return StreamingResponse(
            generate(),
            media_type="application/x-ndjson",
            headers={
                "X-Accel-Buffering": "no",
                "X-Session-ID": session_id,
            }
        )

    except Exception as e:
        logger.exception(f"Stream error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
