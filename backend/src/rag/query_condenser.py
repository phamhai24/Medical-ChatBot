"""Resolve follow-up questions against the conversation topic.

Retrieval only sees the text it is given. A follow-up such as "gợi ý cho tôi
một vài loại thuốc chữa tại nhà" carries no disease name, so on its own it
matches whichever article talks most about home remedies (genital warts,
Crohn...) instead of the disease the user was asking about.

An earlier version asked the LLM to rewrite the question "if needed". That
left the decision to the model, which kept grammatical questions unchanged
and the bug came back phrasing by phrasing. Now the work is split:

* The LLM only names the topic the latest message is about and proposes a
  standalone question, as JSON. Naming "what disease are we discussing?" is a
  much easier and more stable task than judging whether a rewrite is needed.
* Code enforces the rules: the retrieval question always contains the topic;
  a topic that appears neither in the message nor in the conversation is
  rejected in favour of the tracked one; non-medical messages keep the
  tracked topic for later turns; if the LLM fails, the tracked topic is still
  inserted.

The resolved topic is also passed to the answer model (question_with_topic)
so that, if retrieval still returns another disease, the answer says the
information was not found instead of answering about that other disease.
"""

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

logger = logging.getLogger(__name__)

RESOLVE_SYSTEM_PROMPT = """\
Bạn phân tích tin nhắn mới nhất của người dùng trong cuộc trò chuyện với một chatbot y tế và trả về JSON với 2 trường:

"topic": tên bệnh, thuốc hoặc vấn đề sức khỏe cụ thể mà tin nhắn mới nhất ĐANG HỎI VỀ.
  - Nếu tin nhắn không tự nêu tên (ví dụ "nó có lây không", "gợi ý vài loại thuốc", "bao lâu thì khỏi"), thì đó là chủ đề đang trao đổi gần nhất (xem "Chủ đề hiện tại" và lịch sử).
  - Nếu tin nhắn nêu rõ một bệnh/thuốc khác, lấy chủ đề mới đó.
  - Chủ đề phải là một bệnh, thuốc hoặc vấn đề sức khỏe cụ thể, KHÔNG phải cụm chung chung như "thuốc", "chữa tại nhà", "sức khỏe".
  - Nếu tin nhắn không phải câu hỏi y tế (chào hỏi, cảm ơn...), trả null.
  - Viết ngắn gọn theo đúng cách gọi trong hội thoại.
"question": tin nhắn mới nhất viết lại thành một câu hỏi độc lập có chứa tên chủ đề, giữ nguyên ý định, không trả lời câu hỏi.

Ví dụ khi đang nói về viêm gan B:
- "gợi ý vài loại thuốc điều trị" -> {"topic": "viêm gan B", "question": "Gợi ý vài loại thuốc điều trị viêm gan B"}
- "Triệu chứng của sỏi thận là gì?" -> {"topic": "sỏi thận", "question": "Triệu chứng của sỏi thận là gì?"}
- "cảm ơn bạn" -> {"topic": null, "question": "cảm ơn bạn"}

Chỉ trả về JSON."""


@dataclass(frozen=True)
class ResolvedQuestion:
    question: str                 # what retrieval (and the answer model) should use
    topic: Optional[str]          # topic to carry forward to the next turn
    # Topic THIS message is about: None for a non-medical message ("cảm ơn nhé"),
    # which still carries `topic` forward but must not be answered about it.
    question_topic: Optional[str] = None


def _norm(text: str) -> str:
    text = (text or "").lower().replace("-", " ")
    return re.sub(r"\s+", " ", text).strip()


def _with_topic(question: str, topic: Optional[str]) -> str:
    """The question itself if it already names the topic, else topic-prefixed."""
    if not topic or _norm(topic) in _norm(question):
        return question
    return f"{topic}: {question}"


def _clean_topic(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    topic = value.strip().strip("\"'“”").strip()
    return topic if 0 < len(topic) <= 80 else None


def _parse(output: str) -> dict:
    match = re.search(r"\{.*\}", output or "", re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in LLM output: {output!r}")
    return json.loads(match.group(0))


def resolve_question(
    question: str,
    history: Optional[Sequence[dict[str, Any]]],
    generator: Any,
    previous_topic: Optional[str] = None,
    max_messages: int = 6,
    max_chars_per_message: int = 400,
) -> ResolvedQuestion:
    """Resolve `question` against the conversation so far.

    Args:
        question: The latest user message.
        history: Previous messages, oldest first, as {"role", "content"} dicts.
        generator: Anything with generate(prompt, system_prompt=, max_tokens=,
            temperature=) -> str (the API generator).
        previous_topic: Topic tracked from the previous turn, if known.
        max_messages: How many of the most recent messages to show the LLM.
        max_chars_per_message: Truncate each message (answers are long).
    """
    turns = [
        m for m in (history or [])
        if m.get("role") in ("user", "assistant") and str(m.get("content") or "").strip()
    ][-max_messages:]
    previous_topic = _clean_topic(previous_topic)

    if not turns and not previous_topic:
        return ResolvedQuestion(question, None)  # first turn: nothing to resolve against

    fallback = ResolvedQuestion(_with_topic(question, previous_topic), previous_topic, previous_topic)
    if not callable(getattr(generator, "generate", None)):
        return fallback

    transcript = "\n".join(
        f"{'Người dùng' if m['role'] == 'user' else 'Trợ lý'}: "
        f"{str(m['content']).strip()[:max_chars_per_message]}"
        for m in turns
    )
    prompt = (
        f"Chủ đề hiện tại: {previous_topic or 'chưa rõ'}\n\n"
        f"Lịch sử hội thoại:\n{transcript or '(trống)'}\n\n"
        f"Tin nhắn mới nhất: {question}"
    )

    try:
        data = _parse(generator.generate(
            prompt, system_prompt=RESOLVE_SYSTEM_PROMPT, max_tokens=200, temperature=0.0,
        ))
    except Exception as e:  # never let resolving break the chat
        logger.warning(f"Follow-up resolution failed, using tracked topic: {e}")
        return fallback

    topic = _clean_topic(data.get("topic"))
    if topic is None:
        # Not a medical question (thanks, greeting): answer it as-is, but keep
        # the tracked topic so the next follow-up still resolves.
        return ResolvedQuestion(question, previous_topic)

    # A topic must come from somewhere: the message itself or the conversation.
    # Anything else was invented by the model; fall back to the tracked topic.
    conversation = _norm(" ".join([previous_topic or ""] + [str(m["content"]) for m in turns]))
    if _norm(topic) not in _norm(question) and _norm(topic) not in conversation:
        logger.warning(f"Rejected topic {topic!r}: not in the message or conversation")
        topic = previous_topic
        if topic is None:
            return ResolvedQuestion(question, None)

    rewritten = str(data.get("question") or "").strip()
    if not rewritten or len(rewritten) > max(300, 3 * len(question)):
        rewritten = question
    resolved = _with_topic(rewritten, topic)
    if resolved != question:
        logger.info(f"Resolved follow-up: {question!r} -> {resolved!r} (topic={topic!r})")
    return ResolvedQuestion(resolved, topic, topic)


def question_with_topic(question: str, topic: Optional[str]) -> str:
    """The question as shown to the answer model, with the topic being discussed.

    Pairs with the system-prompt rule in config/rag_config.yaml: when the
    retrieved context is about another disease, answer that the information
    was not found rather than answering about the other disease.
    """
    if not topic:
        return question
    return f"{question}\n(Chủ đề đang trao đổi: {topic})"
