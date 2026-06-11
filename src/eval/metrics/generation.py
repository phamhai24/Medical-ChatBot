"""Generation quality metrics."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def faithfulness_score(
    answer: str,
    context: str,
) -> float:
    """
    Faithfulness: Does the answer stick to the provided context?
    Simple heuristic: check how many factual claims in answer appear in context.

    Args:
        answer: Generated answer
        context: Retrieved context

    Returns:
        Score between 0 and 1
    """
    if not answer or not context:
        return 0.0

    context_lower = context.lower()
    answer_lower = answer.lower()

    # Extract key terms from context (words > 4 chars, not common stopwords)
    stopwords = {
        "của", "và", "là", "có", "được", "trong", "với", "cho", "không",
        "này", "các", "những", "để", "từ", "một", "đã", "đến", "bệnh",
        "triệu", "chứng", "thuốc", "điều", "trị", "phòng", "theo", "như",
    }

    # Count context key terms
    import re
    context_terms = set(
        w for w in re.findall(r"\w+", context_lower, re.UNICODE)
        if len(w) > 4 and w not in stopwords
    )

    # Count how many appear in answer
    answer_terms = set(
        w for w in re.findall(r"\w+", answer_lower, re.UNICODE)
        if len(w) > 4 and w not in stopwords
    )

    if not context_terms:
        return 1.0

    matching = len(context_terms & answer_terms)
    total = len(context_terms)

    return min(matching / total, 1.0)


def answer_relevance_score(
    answer: str,
    question: str,
) -> float:
    """
    Answer Relevance: Does the answer address the question?
    Simple heuristic: check overlap between question keywords and answer.

    Args:
        answer: Generated answer
        question: Original question

    Returns:
        Score between 0 and 1
    """
    if not answer or not question:
        return 0.0

    answer_lower = answer.lower()
    question_lower = question.lower()

    import re
    q_terms = set(
        w for w in re.findall(r"\w+", question_lower, re.UNICODE)
        if len(w) > 2
    )
    a_terms = set(
        w for w in re.findall(r"\w+", answer_lower, re.UNICODE)
        if len(w) > 2
    )

    if not q_terms:
        return 1.0

    # Check if question terms appear in answer
    overlap = len(q_terms & a_terms)
    return min(overlap / len(q_terms), 1.0)


def context_precision_score(
    retrieved_docs: list[dict],
    expected_topics: list[str],
) -> float:
    """
    Context Precision: Are the top-ranked contexts relevant?

    Args:
        retrieved_docs: List of retrieved documents
        expected_topics: Expected topic keywords

    Returns:
        Score between 0 and 1
    """
    if not retrieved_docs or not expected_topics:
        return 0.0

    expected_lower = [t.lower() for t in expected_topics]
    relevant_count = 0

    for i, doc in enumerate(retrieved_docs):
        text = doc.get("text", "").lower()
        if any(topic in text for topic in expected_lower):
            relevant_count += 1

    # Weight by position (top docs matter more)
    weighted = 0.0
    for i, doc in enumerate(retrieved_docs):
        text = doc.get("text", "").lower()
        if any(topic in text for topic in expected_lower):
            weighted += 1.0 / (i + 1)

    ideal_weighted = sum(1.0 / (i + 1) for i in range(len(retrieved_docs)))

    if ideal_weighted == 0:
        return 0.0

    return min(weighted / ideal_weighted, 1.0)


def context_recall_score(
    retrieved_docs: list[dict],
    expected_topics: list[str],
) -> float:
    """
    Context Recall: How many expected topics are covered by retrieved docs?

    Args:
        retrieved_docs: List of retrieved documents
        expected_topics: Expected topic keywords

    Returns:
        Score between 0 and 1
    """
    if not retrieved_docs or not expected_topics:
        return 0.0

    expected_lower = [t.lower() for t in expected_topics]
    covered = 0

    for topic in expected_lower:
        for doc in retrieved_docs:
            text = doc.get("text", "").lower()
            if topic in text:
                covered += 1
                break

    return covered / len(expected_lower)


def generation_metrics_summary(
    answer: str,
    question: str,
    context: str,
    retrieved_docs: list[dict],
    expected_topics: list[str],
) -> dict[str, Any]:
    """Compute all generation metrics and return as dict."""
    return {
        "faithfulness": faithfulness_score(answer, context),
        "answer_relevance": answer_relevance_score(answer, question),
        "context_precision": context_precision_score(retrieved_docs, expected_topics),
        "context_recall": context_recall_score(retrieved_docs, expected_topics),
    }
