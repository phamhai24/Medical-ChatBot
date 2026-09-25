"""Rewrite a follow-up question into a standalone one using the chat history.

Retrieval only sees the text it is given. A follow-up such as "Có cách chữa nào
dứt điểm không?" carries no disease name, so on its own it matches whichever
article talks most about "chữa dứt điểm" (Crohn, suy dinh dưỡng...) instead of
the disease the user was just asking about. Condensing it with the recent
history ("Bệnh áp xe não có cách chữa dứt điểm không?") fixes both retrieval
and generation.

Any failure falls back to the original question: condensing must never break
the chat, only improve it.
"""

import logging
import re
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

CONDENSE_SYSTEM_PROMPT = (
    "Bạn viết lại câu hỏi mới nhất của người dùng thành MỘT câu hỏi độc lập, "
    "hiểu được mà không cần đọc lịch sử hội thoại.\n"
    "- Thay các từ như 'nó', 'bệnh này', 'thuốc đó', hoặc chủ đề bị lược bỏ, "
    "bằng tên cụ thể đã được nhắc trong lịch sử.\n"
    "- Giữ nguyên ý định và tiếng Việt. Không thêm thông tin mới, không trả lời câu hỏi.\n"
    "- Nếu câu hỏi đã đủ nghĩa hoặc chuyển sang chủ đề mới, giữ nguyên câu hỏi.\n"
    "Chỉ trả về đúng câu hỏi đã viết lại."
)

_LABEL_RE = re.compile(r"^\s*(câu hỏi (độc lập|viết lại)|câu hỏi)\s*:\s*", re.IGNORECASE)
_QUOTES = "\"'“”‘’`"


def _clean(text: str) -> str:
    first_line = next((line for line in (text or "").splitlines() if line.strip()), "")
    return _LABEL_RE.sub("", first_line).strip().strip(_QUOTES).strip()


def condense_question(
    question: str,
    history: Optional[Sequence[dict[str, Any]]],
    generator: Any,
    max_messages: int = 6,
    max_chars_per_message: int = 400,
) -> str:
    """Return a standalone version of `question`, or `question` itself.

    Args:
        question: The latest user message.
        history: Previous messages, oldest first, as {"role", "content"} dicts.
        generator: Anything with generate(prompt, system_prompt=, max_tokens=,
            temperature=) -> str (the API generator). Without it, no rewrite.
        max_messages: How many of the most recent messages to use.
        max_chars_per_message: Truncate each message (answers are long).
    """
    turns = [
        m for m in (history or [])
        if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()
    ][-max_messages:]
    if not turns or not callable(getattr(generator, "generate", None)):
        return question

    transcript = "\n".join(
        f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: "
        f"{str(m['content']).strip()[:max_chars_per_message]}"
        for m in turns
    )
    prompt = (
        f"Lịch sử hội thoại:\n{transcript}\n\n"
        f"Câu hỏi mới nhất: {question}\n\n"
        f"Câu hỏi độc lập:"
    )

    try:
        output = generator.generate(
            prompt,
            system_prompt=CONDENSE_SYSTEM_PROMPT,
            max_tokens=120,
            temperature=0.0,
        )
    except Exception as e:  # never let condensing break the chat
        logger.warning(f"Question condensing failed, using original question: {e}")
        return question

    rewritten = _clean(output)
    if not rewritten or len(rewritten) > max(300, 3 * len(question)):
        return question
    if rewritten != question:
        logger.info(f"Condensed follow-up question: {question!r} -> {rewritten!r}")
    return rewritten
