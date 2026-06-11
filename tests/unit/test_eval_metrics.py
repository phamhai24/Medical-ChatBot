"""Unit tests for evaluation metrics."""

import pytest
from src.eval.metrics.retrieval import (
    hit_rate, mean_reciprocal_rank, ndcg_at_k,
    precision_at_k, recall_at_k, retrieval_metrics_summary,
)
from src.eval.metrics.generation import (
    faithfulness_score, answer_relevance_score,
    generation_metrics_summary,
)


class TestRetrievalMetrics:
    def test_hit_rate_hit(self):
        """Hit rate should be 1.0 when relevant doc is retrieved."""
        docs = [
            {"text": "Tiểu đường là bệnh đường huyết cao"},
            {"text": "Cao huyết áp là bệnh tim"},
        ]
        topics = ["tiểu đường"]
        assert hit_rate(docs, topics) == 1.0

    def test_hit_rate_miss(self):
        """Hit rate should be 0.0 when no relevant doc is retrieved."""
        docs = [
            {"text": "Vitamin C tốt cho sức khỏe"},
            {"text": "Canxi tốt cho xương"},
        ]
        topics = ["tiểu đường"]
        assert hit_rate(docs, topics) == 0.0

    def test_hit_rate_empty(self):
        """Hit rate should be 0.0 with empty inputs."""
        assert hit_rate([], ["tiểu đường"]) == 0.0
        assert hit_rate([{"text": "abc"}], []) == 0.0

    def test_mrr_perfect_ranking(self):
        """MRR should be 1.0 when first result is relevant."""
        docs = [
            {"text": "Tiểu đường triệu chứng đái tháo đường"},
            {"text": "Cao huyết áp"},
        ]
        topics = ["tiểu đường"]
        assert mean_reciprocal_rank(docs, topics) == 1.0

    def test_mrr_second_ranking(self):
        """MRR should be 0.5 when second result is first relevant."""
        docs = [
            {"text": "Vitamin C"},
            {"text": "Tiểu đường bệnh lý"},
        ]
        topics = ["tiểu đường"]
        assert mean_reciprocal_rank(docs, topics) == 0.5

    def test_mrr_no_relevant(self):
        """MRR should be 0.0 when no relevant docs."""
        docs = [
            {"text": "Vitamin A"},
            {"text": "Vitamin B"},
        ]
        topics = ["tiểu đường"]
        assert mean_reciprocal_rank(docs, topics) == 0.0

    def test_ndcg_at_k(self):
        """NDCG@k should be between 0 and 1."""
        docs = [
            {"text": "Tiểu đường triệu chứng"},
            {"text": "Tiểu đường type 2"},
        ]
        topics = ["tiểu đường"]
        score = ndcg_at_k(docs, topics, k=5)
        assert 0.0 <= score <= 1.0

    def test_precision_at_k(self):
        """Precision@k should be between 0 and 1."""
        docs = [
            {"text": "Tiểu đường"},
            {"text": "Cao huyết áp"},
            {"text": "Vitamin C"},
        ]
        topics = ["tiểu đường"]
        score = precision_at_k(docs, topics, k=3)
        assert 0.0 <= score <= 1.0
        assert score == 1 / 3

    def test_recall_at_k(self):
        """Recall@k should be between 0 and 1."""
        docs = [
            {"text": "Tiểu đường"},
            {"text": "Cao huyết áp"},
        ]
        topics = ["tiểu đường", "cao huyết áp"]
        score = recall_at_k(docs, topics, k=2)
        assert 0.0 <= score <= 1.0
        assert score == 1.0

    def test_retrieval_metrics_summary(self):
        """retrieval_metrics_summary should return all metrics."""
        docs = [
            {"text": "Tiểu đường triệu chứng"},
            {"text": "Cao huyết áp"},
        ]
        topics = ["tiểu đường"]
        summary = retrieval_metrics_summary(docs, topics, k=5)

        assert "hit_rate" in summary
        assert "mrr" in summary
        assert "ndcg@k" in summary
        assert "precision@k" in summary
        assert "recall@k" in summary


class TestGenerationMetrics:
    def test_faithfulness_high(self):
        """Faithfulness should be high when answer matches context."""
        context = "Tiểu đường type 2 là bệnh mãn tính. Triệu chứng bao gồm khát nhiều, đi tiểu nhiều."
        answer = "Tiểu đường type 2 là bệnh mãn tính với triệu chứng khát nhiều và đi tiểu nhiều."
        score = faithfulness_score(answer, context)
        assert score > 0.5

    def test_faithfulness_low(self):
        """Faithfulness should be low when answer differs from context."""
        context = "Tiểu đường type 2 là bệnh mãn tính."
        answer = "Ung thư phổi là bệnh ác tính rất nguy hiểm."
        score = faithfulness_score(answer, context)
        assert score < 0.5

    def test_faithfulness_empty(self):
        """Faithfulness should be 0 with empty inputs."""
        assert faithfulness_score("", "context") == 0.0
        assert faithfulness_score("answer", "") == 0.0

    def test_answer_relevance(self):
        """Answer relevance should be high when answer addresses question."""
        question = "Triệu chứng bệnh tiểu đường là gì?"
        answer = "Triệu chứng bệnh tiểu đường bao gồm khát nhiều nước và đi tiểu thường xuyên."
        score = answer_relevance_score(answer, question)
        assert score > 0.3

    def test_generation_metrics_summary(self):
        """generation_metrics_summary should return all metrics."""
        summary = generation_metrics_summary(
            answer="Triệu chứng tiểu đường là khát nhiều.",
            question="Triệu chứng bệnh tiểu đường là gì?",
            context="Tiểu đường có triệu chứng khát nhiều.",
            retrieved_docs=[{"text": "Tiểu đường có triệu chứng khát nhiều."}],
            expected_topics=["tiểu đường", "triệu chứng"],
        )

        assert "faithfulness" in summary
        assert "answer_relevance" in summary
        assert "context_precision" in summary
        assert "context_recall" in summary
