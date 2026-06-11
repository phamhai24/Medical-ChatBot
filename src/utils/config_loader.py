"""Config loader - load YAML config into Settings."""

import yaml
from pathlib import Path
from typing import Any

from src.core.config import get_settings


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load YAML configuration file and merge with environment settings.

    YAML values can override defaults but .env values always win.
    """
    settings = get_settings()
    path = Path(config_path) if config_path else settings.config_path

    config: dict[str, Any] = {}

    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # Flatten and merge with settings
    merged = _build_rag_config(config, settings)

    return merged


def _build_rag_config(yaml_config: dict[str, Any], settings) -> dict[str, Any]:
    """Build the full config dict matching the original YAML structure."""
    generation_model = (
        settings.api_generator_model
        if settings.generator_mode == "api"
        else settings.generator_model
    )

    return {
        "rag": {
            "data": {
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "min_chunk_length": settings.chunk_min_length,
                "input_path": str(settings.project_root / "data" / "processed" / "data.json"),
            },
            "embedding": {
                "model_name": settings.embedding_model,
                "device": settings.embedding_device,
                "normalize_embeddings": settings.embedding_normalize,
                "batch_size": settings.embedding_batch_size,
                "max_length": settings.embedding_max_length,
            },
            "vector_store": {
                "type": settings.vector_store_type,
                "persist_directory": str(settings.project_root / settings.vector_store_path),
                "collection_name": settings.vector_store_collection,
                "search_type": settings.vector_search_type,
                "fetch_k": settings.vector_fetch_k,
                "lambda_mult": settings.vector_lambda_mult,
            },
            "retrieval": {
                "top_k": settings.retrieval_top_k,
                "score_threshold": settings.retrieval_score_threshold,
                "fetch_k": settings.retrieval_fetch_k,
                "vector_weight": settings.retrieval_vector_weight,
                "bm25_weight": settings.retrieval_bm25_weight,
            },
            "generation": {
                "model_name": generation_model,
                "max_new_tokens": settings.generator_max_tokens,
                "temperature": settings.generator_temperature,
                "top_p": settings.generator_top_p,
                "top_k": settings.generator_top_k,
                "do_sample": settings.generator_do_sample,
                "repetition_penalty": settings.generator_repetition_penalty,
                "streaming": True,
                "mode": settings.generator_mode,
                "api_provider": settings.api_generator_provider,
                "api_key": settings.api_generator_api_key,
                "api_base_url": settings.api_generator_base_url,
                "api_max_retries": settings.api_generator_max_retries,
                "api_retry_backoff": settings.api_generator_retry_backoff,
                "api_retry_max_backoff": settings.api_generator_retry_max_backoff,
                "api_circuit_breaker_threshold": (
                    settings.api_generator_circuit_breaker_threshold
                ),
                "api_circuit_breaker_cooldown": (
                    settings.api_generator_circuit_breaker_cooldown
                ),
            },
            "prompt": yaml_config.get("rag", {}).get("prompt", _default_prompt()),
        },
        "api": {
            "host": settings.api_host,
            "port": settings.api_port,
            "reload": settings.api_reload,
            "cors_origins": settings.api_cors_origins,
        },
    }


def _default_prompt() -> dict[str, str]:
    return {
        "system": (
            "Bạn là trợ lý y tế chuyên nghiệp. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng "
            "dựa trên thông tin được cung cấp trong ngữ cảnh (context) bên dưới.\n\n"
            "HƯỚNG DẪN QUAN TRỌNG:\n"
            "- Chỉ trả lời dựa trên thông tin có trong ngữ cảnh. Không bịa đặt thông tin.\n"
            "- Nếu ngữ cảnh không chứa thông tin để trả lời câu hỏi, hãy nói rõ rằng bạn không tìm thấy thông tin đó.\n"
            "- Trả lời bằng tiếng Việt, rõ ràng và dễ hiểu.\n"
            "- Nếu câu hỏi yêu cầu thông tin y tế chuyên môn, hãy trình bày logic và chi tiết.\n"
            "- Luôn nhắc nhở người dùng tham khảo ý kiến bác sĩ cho các quyết định y tế quan trọng.\n"
            "- Không đưa ra chẩn đoán y khoa cụ thể cho cá nhân."
        ),
        "user_template": (
            "Ngữ cảnh (Context):\n{context}\n\n"
            "Câu hỏi: {question}\n\n"
            "Trả lời:"
        ),
    }
