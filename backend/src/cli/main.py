"""CLI entry point for the medical RAG chatbot."""

import argparse
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(
        description="Medical RAG Chatbot CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ─── ingest ────────────────────────────────────────────────────────────────
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest data into the vector store",
    )
    ingest_parser.add_argument("--config", default="config/rag_config.yaml")
    ingest_parser.add_argument("--data", default="data/processed/data.json")
    ingest_parser.add_argument("--rebuild", action="store_true", help="Clear existing index first")
    ingest_parser.add_argument("--batch-size", type=int, default=100)
    ingest_parser.add_argument("--no-progress", action="store_true")

    # ─── eval ─────────────────────────────────────────────────────────────────
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run evaluation benchmark",
    )
    eval_parser.add_argument("--output", default="reports")
    eval_parser.add_argument("--format", nargs="+", default=["html", "csv"], choices=["html", "csv"])
    eval_parser.add_argument("--llm-judge", action="store_true")
    eval_parser.add_argument("--use-api", action="store_true", help="Use API generator for LLM judge")
    eval_parser.add_argument("--top-k", type=int, default=5)
    eval_parser.add_argument("--no-progress", action="store_true")

    # ─── serve ─────────────────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser(
        "serve",
        help="Start the API server",
    )
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.add_argument("--workers", type=int, default=1)

    # ─── stats ─────────────────────────────────────────────────────────────────
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show pipeline statistics",
    )

    args = parser.parse_args()

    if args.command == "ingest":
        from src.rag.pipeline import RAGPipeline
        from src.utils.config_loader import load_config

        config = load_config(args.config)
        pipeline = RAGPipeline(config)

        print(f"Ingesting data from {args.data}...")
        stats = pipeline.ingest(
            data_path=args.data,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
            show_progress=not args.no_progress,
        )

        print("\n=== Ingestion Complete ===")
        for k, v in stats.items():
            print(f"  {k}: {v}")

    elif args.command == "eval":
        from src.eval.evaluator import Evaluator
        from src.rag.pipeline import RAGPipeline
        from src.utils.config_loader import load_config
        from src.eval.metrics.llm_judge import LLMJudge
        from src.eval.reporters.report import EvaluationReporter

        config = load_config()
        pipeline = RAGPipeline(config)

        print("Running evaluation...")
        evaluator = Evaluator(pipeline, top_k=args.top_k)

        results = evaluator.run(
            run_llm_judge=args.llm_judge,
            use_api_judge=args.use_api,
            show_progress=not args.no_progress,
        )

        print("\n=== Evaluation Summary ===")
        summary = results["summary"]
        print(f"  Hit Rate:     {summary.get('hit_rate', 0):.2%}")
        print(f"  MRR:          {summary.get('mrr', 0):.4f}")
        print(f"  NDCG@5:       {summary.get('ndcg@k', 0):.2%}")
        print(f"  Faithfulness: {summary.get('avg_faithfulness', 0):.2%}")
        print(f"  Relevance:    {summary.get('avg_relevance', 0):.2%}")
        print(f"  Avg Latency:  {summary.get('avg_latency_ms', 0):.0f}ms")

        # Generate reports
        paths = evaluator.generate_reports(
            results,
            output_dir=args.output,
            formats=args.format,
        )
        print(f"\nReports saved:")
        for fmt, path in paths.items():
            print(f"  [{fmt}] {path}")

    elif args.command == "serve":
        import uvicorn
        from src.api.main import app

        print(f"Starting server at http://{args.host}:{args.port}")
        print(f"  Docs: http://{args.host}:{args.port}/docs")
        print(f"  Health: http://{args.host}:{args.port}/health")

        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
        )

    elif args.command == "stats":
        from src.rag.pipeline import RAGPipeline
        from src.utils.config_loader import load_config
        import json

        config = load_config()
        pipeline = RAGPipeline(config)
        pipeline._lazy_init()

        stats = pipeline.get_stats()
        print(json.dumps(stats, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
