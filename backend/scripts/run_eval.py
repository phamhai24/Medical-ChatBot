"""Run evaluation benchmark."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.evaluator import Evaluator
from src.rag.pipeline import RAGPipeline
from src.utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation benchmark")
    parser.add_argument("--config", default="config/rag_config.yaml")
    parser.add_argument("--output", default="reports", help="Output directory for reports")
    parser.add_argument("--format", nargs="+", default=["html", "csv"], choices=["html", "csv"])
    parser.add_argument("--llm-judge", action="store_true", help="Run LLM-as-Judge evaluation")
    parser.add_argument("--use-api-judge", action="store_true", help="Use API for LLM judge")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = RAGPipeline(config)

    print("Initializing evaluator...")
    evaluator = Evaluator(pipeline, top_k=args.top_k)

    print(f"Running evaluation on benchmark dataset...")
    results = evaluator.run(
        run_llm_judge=args.llm_judge,
        use_api_judge=args.use_api_judge,
        show_progress=not args.no_progress,
    )

    summary = results["summary"]
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Questions tested:  {summary['total_questions']}")
    print(f"Success rate:     {summary['success_rate']:.2%}")
    print()
    print("Retrieval:")
    print(f"  Hit Rate:       {summary['hit_rate']:.2%}")
    print(f"  MRR:            {summary['mrr']:.4f}")
    print(f"  NDCG@5:         {summary['ndcg@k']:.2%}")
    print(f"  Precision@5:    {summary['precision@k']:.2%}")
    print()
    print("Generation:")
    print(f"  Faithfulness:   {summary['avg_faithfulness']:.2%}")
    print(f"  Answer Relev.:  {summary['avg_relevance']:.2%}")
    print(f"  Context Prec.:  {summary['avg_context_precision']:.2%}")
    print(f"  Context Recall: {summary['avg_context_recall']:.2%}")
    if summary.get('llm_judge_scores'):
        print(f"  LLM-Judge:      {summary['avg_llm_judge']:.2f}/5")
    print()
    print(f"Avg Latency:      {summary['avg_latency_ms']:.0f}ms")
    print("=" * 50)

    paths = evaluator.generate_reports(
        results,
        output_dir=args.output,
        formats=args.format,
    )
    print("\nReports generated:")
    for fmt, path in paths.items():
        print(f"  [{fmt}] {path}")

    return results


if __name__ == "__main__":
    main()
