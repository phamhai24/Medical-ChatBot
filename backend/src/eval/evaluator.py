"""Main evaluation orchestrator."""

import logging
import time
from typing import Any, Optional

from src.eval.benchmarks.medical_qa import get_benchmark_dataset
from src.eval.metrics.generation import generation_metrics_summary
from src.eval.metrics.llm_judge import LLMJudge
from src.eval.metrics.retrieval import retrieval_metrics_summary
from src.eval.reporters.report import EvaluationReporter

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Main evaluation orchestrator for the RAG pipeline.

    Runs benchmark questions through the pipeline and evaluates
    both retrieval and generation quality.
    """

    def __init__(
        self,
        pipeline,
        benchmark_dataset: Optional[list[dict[str, Any]]] = None,
        llm_judge: Optional[LLMJudge] = None,
        top_k: int = 5,
    ):
        """
        Args:
            pipeline: RAGPipeline instance
            benchmark_dataset: List of benchmark Q&A pairs
            llm_judge: LLMJudge instance for LLM-as-Judge evaluation
            top_k: Number of documents to retrieve per query
        """
        self.pipeline = pipeline
        self.benchmark = benchmark_dataset or get_benchmark_dataset()
        self.llm_judge = llm_judge
        self.top_k = top_k

    def run(
        self,
        run_llm_judge: bool = False,
        use_api_judge: bool = False,
        show_progress: bool = True,
    ) -> dict[str, Any]:
        """
        Run the full evaluation pipeline.

        Args:
            run_llm_judge: Whether to run LLM-as-Judge evaluation
            use_api_judge: Use API generator for judge (vs local)
            show_progress: Show progress bars

        Returns:
            Dictionary with per-question results and summary statistics
        """
        from tqdm import tqdm

        logger.info(f"Starting evaluation on {len(self.benchmark)} benchmark questions")

        self.pipeline._lazy_init()

        results = []
        summary_metrics = {
            "total_questions": len(self.benchmark),
            "successful": 0,
            "no_results": 0,
            "errors": 0,
            "total_latency_ms": 0.0,
            "hit_rates": [],
            "mrrs": [],
            "ndcgs": [],
            "precisions": [],
            "recalls": [],
            "faithfulness_scores": [],
            "relevance_scores": [],
            "context_precision_scores": [],
            "context_recall_scores": [],
            "llm_judge_scores": [],
        }

        iterator = tqdm(self.benchmark, disable=not show_progress, desc="Evaluating")

        for item in iterator:
            result = self._evaluate_single(
                item,
                run_llm_judge=run_llm_judge,
                use_api_judge=use_api_judge,
            )
            results.append(result)

            # Update summary metrics
            status = result.get("status", "")
            if status == "success":
                summary_metrics["successful"] += 1
            elif status == "no_results":
                summary_metrics["no_results"] += 1
            else:
                summary_metrics["errors"] += 1

            summary_metrics["total_latency_ms"] += result.get("latency_ms", 0)

            ret = result.get("retrieval_metrics", {})
            gen = result.get("generation_metrics", {})

            summary_metrics["hit_rates"].append(ret.get("hit_rate", 0))
            summary_metrics["mrrs"].append(ret.get("mrr", 0))
            summary_metrics["ndcgs"].append(ret.get("ndcg@k", 0))
            summary_metrics["precisions"].append(ret.get("precision@k", 0))
            summary_metrics["recalls"].append(ret.get("recall@k", 0))
            summary_metrics["faithfulness_scores"].append(gen.get("faithfulness", 0))
            summary_metrics["relevance_scores"].append(gen.get("answer_relevance", 0))
            summary_metrics["context_precision_scores"].append(gen.get("context_precision", 0))
            summary_metrics["context_recall_scores"].append(gen.get("context_recall", 0))

            llm_j = result.get("llm_judge", {})
            if llm_j:
                summary_metrics["llm_judge_scores"].append(llm_j.get("overall", 3))

            if show_progress:
                avg_hit = sum(summary_metrics["hit_rates"]) / max(len(summary_metrics["hit_rates"]), 1)
                iterator.set_postfix({"hit_rate": f"{avg_hit:.2%}"})

        # Compute summary statistics
        summary = self._compute_summary(summary_metrics)

        return {
            "results": results,
            "summary": summary,
            "metadata": {
                "model": self.pipeline._generation_config.get("model_name", ""),
                "embedding_model": self.pipeline._embedding_config.get("model_name", ""),
                "top_k": self.top_k,
                "benchmark_size": len(self.benchmark),
            },
        }

    def _evaluate_single(
        self,
        item: dict[str, Any],
        run_llm_judge: bool = False,
        use_api_judge: bool = False,
    ) -> dict[str, Any]:
        """Evaluate a single benchmark question."""
        question = item["question"]
        expected_topics = item.get("expected_topics", [])
        result = {
            "id": item.get("id", ""),
            "question": question,
            "category": item.get("category", ""),
            "difficulty": item.get("difficulty", ""),
            "expected_topics": expected_topics,
            "status": "success",
            "retrieval_metrics": {},
            "generation_metrics": {},
            "llm_judge": {},
            "latency_ms": 0,
            "sources": [],
            "answer": "",
        }

        start = time.perf_counter()

        try:
            # Retrieve
            retrieved_docs = self.pipeline.retriever.retrieve(question, top_k=self.top_k)

            if not retrieved_docs:
                result["status"] = "no_results"
                result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
                return result

            # Get context
            context_result = self.pipeline.retriever.get_context_with_citations(
                question, top_k=self.top_k
            )
            context = context_result["context"]

            # Generate answer
            response = self.pipeline.query(
                question=question,
                top_k=self.top_k,
                include_sources=False,
            )

            result["answer"] = response.answer
            result["sources"] = response.sources

            # Retrieval metrics
            result["retrieval_metrics"] = retrieval_metrics_summary(
                retrieved_docs, expected_topics, k=self.top_k
            )

            # Generation metrics
            result["generation_metrics"] = generation_metrics_summary(
                answer=response.answer,
                question=question,
                context=context,
                retrieved_docs=retrieved_docs,
                expected_topics=expected_topics,
            )

            # LLM-as-Judge
            if run_llm_judge and self.llm_judge:
                try:
                    result["llm_judge"] = self.llm_judge.evaluate(
                        question=question,
                        answer=response.answer,
                        context=context,
                        use_api=use_api_judge,
                    )
                except Exception as e:
                    logger.warning(f"LLM judge failed for {item.get('id', '?')}: {e}")

        except Exception as e:
            logger.error(f"Evaluation error for {item.get('id', '?')}: {e}")
            result["status"] = "error"
            result["error"] = str(e)

        result["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
        return result

    def _compute_summary(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Compute aggregate summary statistics."""
        import statistics

        def mean(vals):
            return statistics.mean(vals) if vals else 0.0

        def median(vals):
            return statistics.median(vals) if vals else 0.0

        def p95(vals):
            if not vals:
                return 0.0
            sorted_vals = sorted(vals)
            idx = int(len(sorted_vals) * 0.95)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]

        n = max(len(metrics["hit_rates"]), 1)

        return {
            "total_questions": metrics["total_questions"],
            "successful": metrics["successful"],
            "no_results": metrics["no_results"],
            "errors": metrics["errors"],
            "success_rate": metrics["successful"] / n,
            "hit_rate": mean(metrics["hit_rates"]),
            "mrr": mean(metrics["mrrs"]),
            "ndcg@k": mean(metrics["ndcgs"]),
            "precision@k": mean(metrics["precisions"]),
            "recall@k": mean(metrics["recalls"]),
            "avg_precision": mean(metrics.get("mrrs", [])),
            "avg_faithfulness": mean(metrics["faithfulness_scores"]),
            "avg_relevance": mean(metrics["relevance_scores"]),
            "avg_context_precision": mean(metrics["context_precision_scores"]),
            "avg_context_recall": mean(metrics["context_recall_scores"]),
            "avg_llm_judge": mean(metrics["llm_judge_scores"]),
            "avg_latency_ms": metrics["total_latency_ms"] / n,
            "latency_p50_ms": median([r.get("latency_ms", 0) for r in []]),
            "latency_p95_ms": p95([r.get("latency_ms", 0) for r in []]),
        }

    def generate_reports(
        self,
        eval_results: dict[str, Any],
        output_dir: str = "reports",
        formats: list[str] = ["html", "csv"],
    ) -> dict[str, str]:
        """
        Generate evaluation reports.

        Args:
            eval_results: Results from run()
            output_dir: Output directory for reports
            formats: List of formats to generate ("html", "csv")

        Returns:
            Dict mapping format to output path
        """
        reporter = EvaluationReporter(output_dir)
        paths = {}

        results = eval_results.get("results", [])
        summary = eval_results.get("summary", {})
        metadata = eval_results.get("metadata", {})

        if "html" in formats:
            paths["html"] = reporter.generate_html_report(results, summary, metadata)

        if "csv" in formats:
            paths["csv"] = reporter.generate_csv_report(results)

        return paths
