"""Run the RAG pipeline against the independent, corpus-free benchmark.

Unlike scripts/run_eval.py (which defaults to the corpus-derived
data/eval/medical_benchmark.json), this loads
data/eval/independent_benchmark_v1.json — a benchmark authored from general
medical knowledge, not sampled from the ingested corpus — so retrieval and
generation scores reflect real generalization instead of the system finding
questions it was effectively built from.

It also wires up LLMJudge with the pipeline's own API generator so
generation quality is scored by an independent model (OpenAI gpt-4o-mini by
default) instead of manual self-evaluation.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.eval.evaluator import Evaluator
from src.eval.metrics.llm_judge import LLMJudge
from src.rag.pipeline import RAGPipeline
from src.utils.config_loader import load_config

DECLINE_PATTERNS = [
    "không tìm thấy", "không có thông tin", "không có dữ liệu",
    "không rõ", "chưa rõ", "không thể xác nhận", "không thể chắc chắn",
    "không có cơ sở", "chưa có nghiên cứu", "không nên tự", "cần thăm khám",
    "tham khảo bác sĩ", "gặp bác sĩ", "liên hệ bác sĩ", "ý kiến bác sĩ",
    "cơ sở y tế", "không đủ thông tin", "chưa được ghi nhận", "không có bằng chứng",
    "không thể cung cấp", "không thể xác định", "không thể đưa ra", "trao đổi trực tiếp với bác sĩ",
]


def looks_like_honest_decline(answer: str) -> bool:
    """Heuristic: did the answer decline / hedge instead of confidently answering?"""
    lower = (answer or "").lower()
    return any(p in lower for p in DECLINE_PATTERNS)


def load_independent_benchmark(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["items"]


def main():
    parser = argparse.ArgumentParser(description="Run independent (corpus-free) RAG evaluation")
    parser.add_argument("--config", default="config/rag_config.yaml")
    parser.add_argument(
        "--benchmark",
        default="data/eval/independent_benchmark_v1.json",
        help="Path to the independent benchmark JSON file",
    )
    parser.add_argument("--output", default="reports", help="Output directory for reports")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge scoring")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N items (smoke test)")
    parser.add_argument("--ids", nargs="+", default=None, help="Only run items with these ids (e.g. a03 a08)")
    args = parser.parse_args()

    benchmark = load_independent_benchmark(ROOT / args.benchmark)
    if args.ids:
        wanted = set(args.ids)
        benchmark = [i for i in benchmark if i["id"] in wanted]
    if args.limit:
        benchmark = benchmark[: args.limit]

    config = load_config(args.config)
    pipeline = RAGPipeline(config)
    pipeline._lazy_init()

    judge = None
    if not args.no_judge:
        judge = LLMJudge(api_generator=pipeline.generator)

    evaluator = Evaluator(pipeline, benchmark_dataset=benchmark, llm_judge=judge, top_k=args.top_k)

    print(f"Running independent evaluation on {len(benchmark)} questions "
          f"({sum(1 for i in benchmark if i.get('is_adversarial'))} adversarial)...")

    results = evaluator.run(
        run_llm_judge=judge is not None,
        use_api_judge=True,
        show_progress=True,
    )

    # Adversarial-subset analysis: honest-failure rate instead of retrieval/faithfulness.
    adversarial_ids = {i["id"] for i in benchmark if i.get("is_adversarial")}
    adv_results = [r for r in results["results"] if r["id"] in adversarial_ids]
    in_scope_results = [r for r in results["results"] if r["id"] not in adversarial_ids]

    honest_declines = sum(1 for r in adv_results if looks_like_honest_decline(r.get("answer", "")))
    hallucination_scores = [
        r["llm_judge"]["hallucination"] for r in adv_results
        if r.get("llm_judge", {}).get("hallucination") is not None
    ]

    results["adversarial_analysis"] = {
        "total_adversarial": len(adv_results),
        "honest_decline_count": honest_declines,
        "honest_decline_rate": honest_declines / len(adv_results) if adv_results else None,
        "avg_hallucination_score": (
            sum(hallucination_scores) / len(hallucination_scores) if hallucination_scores else None
        ),
        "per_item": [
            {
                "id": r["id"],
                "question": r["question"],
                "answer_excerpt": (r.get("answer") or "")[:300],
                "honest_decline_heuristic": looks_like_honest_decline(r.get("answer", "")),
                "judge_hallucination_score": r.get("llm_judge", {}).get("hallucination"),
            }
            for r in adv_results
        ],
    }

    summary = results["summary"]
    print("\n" + "=" * 60)
    print("INDEPENDENT BENCHMARK RESULTS (corpus-free)")
    print("=" * 60)
    print(f"Total questions:     {summary['total_questions']} ({len(in_scope_results)} in-scope, {len(adv_results)} adversarial)")
    print(f"Success rate:        {summary['success_rate']:.2%}")
    print()
    print("Retrieval (in-scope questions drive this; adversarial expected to miss):")
    print(f"  Hit Rate:           {summary['hit_rate']:.2%}")
    print(f"  MRR:                {summary['mrr']:.4f}")
    print(f"  NDCG@k:             {summary['ndcg@k']:.2%}")
    print(f"  Precision@k:        {summary['precision@k']:.2%}")
    print(f"  Recall@k:           {summary['recall@k']:.2%}")
    print()
    print("Generation (LLM-as-judge, independent model):")
    print(f"  Avg judge overall:  {summary['avg_llm_judge']:.2f}/5")
    print()
    print("Adversarial subset (honest-failure / hallucination resistance):")
    aa = results["adversarial_analysis"]
    if aa["honest_decline_rate"] is not None:
        print(f"  Honest-decline rate (heuristic): {aa['honest_decline_rate']:.0%} ({aa['honest_decline_count']}/{aa['total_adversarial']})")
    if aa["avg_hallucination_score"] is not None:
        print(f"  Avg judge hallucination score:   {aa['avg_hallucination_score']:.2f}/5 (higher = less hallucination)")
    print("=" * 60)

    # Reports
    paths = evaluator.generate_reports(results, output_dir=args.output, formats=["html", "csv"])
    print("\nReports generated:")
    for fmt, path in paths.items():
        print(f"  [{fmt}] {path}")

    # Raw JSON dump (includes adversarial_analysis, not covered by the generic reporter)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = Path(args.output) / f"independent_eval_raw_{ts}.json"
    raw_path.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"  [json] {raw_path}")

    return results


if __name__ == "__main__":
    main()
