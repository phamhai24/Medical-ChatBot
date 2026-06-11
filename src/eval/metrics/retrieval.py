"""Evaluation metrics for retrieval quality."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def hit_rate(retrieved_docs: list[dict], expected_topics: list[str]) -> float:
    """
    Hit Rate: Does the retrieved set contain at least one relevant document?

    Args:
        retrieved_docs: List of retrieved documents
        expected_topics: List of expected topic keywords

    Returns:
        1.0 if hit, 0.0 otherwise
    """
    if not retrieved_docs or not expected_topics:
        return 0.0

    expected_lower = [t.lower() for t in expected_topics]

    for doc in retrieved_docs:
        text = doc.get("text", "").lower()
        for topic in expected_lower:
            if topic in text:
                return 1.0

    return 0.0


def mean_reciprocal_rank(
    retrieved_docs: list[dict],
    expected_topics: list[str],
) -> float:
    """
    MRR: Average of reciprocal ranks of first relevant document.

    Args:
        retrieved_docs: List of retrieved documents
        expected_topics: List of expected topic keywords

    Returns:
        MRR score between 0 and 1
    """
    if not retrieved_docs or not expected_topics:
        return 0.0

    expected_lower = [t.lower() for t in expected_topics]

    for i, doc in enumerate(retrieved_docs):
        text = doc.get("text", "").lower()
        for topic in expected_lower:
            if topic in text:
                return 1.0 / (i + 1)

    return 0.0


def ndcg_at_k(
    retrieved_docs: list[dict],
    expected_topics: list[str],
    k: int = 5,
) -> float:
    """
    NDCG@k: Normalized Discounted Cumulative Gain at k.

    Measures ranking quality - relevant docs should appear at the top.

    Args:
        retrieved_docs: List of retrieved documents (ordered by relevance)
        expected_topics: List of expected topic keywords
        k: Cutoff position

    Returns:
        NDCG@k score between 0 and 1
    """
    if not retrieved_docs or not expected_topics:
        return 0.0

    expected_lower = [t.lower() for t in expected_topics]
    docs_to_evaluate = retrieved_docs[:k]

    # Relevance grades
    def relevance(doc: dict) -> float:
        text = doc.get("text", "").lower()
        score = 0.0
        for topic in expected_lower:
            if topic in text:
                score += 1.0
        return min(score / len(expected_lower), 1.0)

    # DCG
    dcg = 0.0
    for i, doc in enumerate(docs_to_evaluate):
        rel = relevance(doc)
        dcg += rel / (i + 1)

    # Ideal DCG
    ideal_relevances = sorted([1.0] * len(expected_lower), reverse=True)
    idcg = sum(r / (i + 1) for i, r in enumerate(ideal_relevances[:k]))

    if idcg == 0:
        return 0.0

    return dcg / idcg


def precision_at_k(
    retrieved_docs: list[dict],
    expected_topics: list[str],
    k: int = 5,
) -> float:
    """
    Precision@k: Fraction of retrieved docs that are relevant.

    Args:
        retrieved_docs: List of retrieved documents
        expected_topics: List of expected topic keywords
        k: Cutoff position

    Returns:
        Precision@k score between 0 and 1
    """
    if not retrieved_docs or not expected_topics:
        return 0.0

    expected_lower = [t.lower() for t in expected_topics]
    docs_to_evaluate = retrieved_docs[:k]
    relevant = 0

    for doc in docs_to_evaluate:
        text = doc.get("text", "").lower()
        if any(topic in text for topic in expected_lower):
            relevant += 1

    return relevant / k


def recall_at_k(
    retrieved_docs: list[dict],
    expected_topics: list[str],
    k: int = 5,
) -> float:
    """
    Recall@k: Fraction of relevant topics found in top-k results.

    Args:
        retrieved_docs: List of retrieved documents
        expected_topics: List of expected topic keywords
        k: Cutoff position

    Returns:
        Recall@k score between 0 and 1
    """
    if not retrieved_docs or not expected_topics:
        return 0.0

    expected_lower = [t.lower() for t in expected_topics]
    docs_to_evaluate = retrieved_docs[:k]
    found_topics = set()

    for doc in docs_to_evaluate:
        text = doc.get("text", "").lower()
        for topic in expected_lower:
            if topic in text:
                found_topics.add(topic)

    return len(found_topics) / len(expected_lower)


def average_precision(
    retrieved_docs: list[dict],
    expected_topics: list[str],
) -> float:
    """
    Average Precision: Precision at each relevant position.

    Args:
        retrieved_docs: List of retrieved documents
        expected_topics: List of expected topic keywords

    Returns:
        Average precision score
    """
    if not retrieved_docs or not expected_topics:
        return 0.0

    expected_lower = [t.lower() for t in expected_topics]
    num_relevant = 0
    sum_precision = 0.0

    for i, doc in enumerate(retrieved_docs):
        text = doc.get("text", "").lower()
        if any(topic in text for topic in expected_lower):
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)

    if num_relevant == 0:
        return 0.0

    return sum_precision / num_relevant


def retrieval_metrics_summary(
    retrieved_docs: list[dict],
    expected_topics: list[str],
    k: int = 5,
) -> dict[str, Any]:
    """Compute all retrieval metrics and return as dict."""
    return {
        "hit_rate": hit_rate(retrieved_docs, expected_topics),
        "mrr": mean_reciprocal_rank(retrieved_docs, expected_topics),
        "ndcg@k": ndcg_at_k(retrieved_docs, expected_topics, k),
        f"precision@{k}": precision_at_k(retrieved_docs, expected_topics, k),
        f"recall@{k}": recall_at_k(retrieved_docs, expected_topics, k),
        "avg_precision": average_precision(retrieved_docs, expected_topics),
    }
