"""HTML and CSV report generation for evaluation results."""

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Generate evaluation reports in HTML and CSV formats."""

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(
        self,
        results: list[dict[str, Any]],
        summary: dict[str, Any],
        metadata: dict[str, Any],
        output_path: str | None = None,
    ) -> str:
        """Generate a detailed HTML evaluation report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

        html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical RAG Chatbot - Evaluation Report</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f5f7fa; color: #1a1a2e; line-height: 1.6; }}
        .container {{ max-width: 1200px; margin: 0 auto; padding: 2rem; }}
        .header {{ background: linear-gradient(135deg, #0062cc, #00a8cc); color: white; padding: 2rem; border-radius: 12px; margin-bottom: 2rem; }}
        .header h1 {{ font-size: 1.8rem; margin-bottom: 0.5rem; }}
        .header p {{ opacity: 0.9; font-size: 0.9rem; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin-bottom: 2rem; }}
        .summary-card {{ background: white; padding: 1.5rem; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.06); text-align: center; }}
        .summary-card .value {{ font-size: 2rem; font-weight: 700; color: #0062cc; }}
        .summary-card .label {{ color: #666; font-size: 0.85rem; margin-top: 0.25rem; }}
        .summary-card .badge {{ display: inline-block; padding: 0.2rem 0.6rem; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-top: 0.5rem; }}
        .badge-good {{ background: #d4edda; color: #155724; }}
        .badge-warning {{ background: #fff3cd; color: #856404; }}
        .badge-danger {{ background: #f8d7da; color: #721c24; }}
        .section {{ background: white; border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.06); }}
        .section h2 {{ font-size: 1.2rem; margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #f0f0f0; }}
        .question-card {{ padding: 1rem; border: 1px solid #eee; border-radius: 6px; margin-bottom: 1rem; }}
        .question-card .q {{ font-weight: 600; color: #0062cc; margin-bottom: 0.5rem; }}
        .question-card .meta {{ font-size: 0.8rem; color: #888; margin-bottom: 0.5rem; }}
        .question-card .answer {{ background: #f8f9fa; padding: 0.75rem; border-radius: 4px; margin-top: 0.5rem; font-size: 0.9rem; }}
        .metrics-row {{ display: flex; gap: 1rem; flex-wrap: wrap; margin-top: 0.5rem; }}
        .metric {{ background: #f0f4ff; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; }}
        .metric .val {{ font-weight: 600; color: #0062cc; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #eee; }}
        th {{ background: #f8f9fa; font-weight: 600; font-size: 0.85rem; color: #555; }}
        .score-bar {{ height: 6px; background: #e9ecef; border-radius: 3px; margin-top: 0.25rem; }}
        .score-bar-fill {{ height: 100%; border-radius: 3px; transition: width 0.3s; }}
        .footer {{ text-align: center; color: #888; font-size: 0.8rem; margin-top: 2rem; }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Medical RAG Chatbot - Evaluation Report</h1>
        <p>Generated: {timestamp} | Model: {metadata.get('model', 'Unknown')}</p>
    </div>

    <div class="summary-grid">
        <div class="summary-card">
            <div class="value">{summary.get('hit_rate', 0):.2%}</div>
            <div class="label">Hit Rate</div>
            {self._badge(summary.get('hit_rate', 0), 0.8, 0.6)}
        </div>
        <div class="summary-card">
            <div class="value">{summary.get('mrr', 0):.2%}</div>
            <div class="label">MRR</div>
            {self._badge(summary.get('mrr', 0), 0.7, 0.5)}
        </div>
        <div class="summary-card">
            <div class="value">{summary.get('ndcg@k', 0):.2%}</div>
            <div class="label">NDCG@5</div>
            {self._badge(summary.get('ndcg@k', 0), 0.7, 0.5)}
        </div>
        <div class="summary-card">
            <div class="value">{summary.get('avg_faithfulness', 0):.2%}</div>
            <div class="label">Faithfulness</div>
            {self._badge(summary.get('avg_faithfulness', 0), 0.85, 0.7)}
        </div>
        <div class="summary-card">
            <div class="value">{summary.get('avg_relevance', 0):.2%}</div>
            <div class="label">Answer Relevance</div>
            {self._badge(summary.get('avg_relevance', 0), 0.85, 0.7)}
        </div>
        <div class="summary-card">
            <div class="value">{summary.get('avg_llm_judge', 0):.2f}/5</div>
            <div class="label">LLM-Judge Score</div>
            {self._badge(summary.get('avg_llm_judge', 0) / 5, 0.8, 0.6)}
        </div>
        <div class="summary-card">
            <div class="value">{summary.get('total_questions', 0)}</div>
            <div class="label">Questions Tested</div>
        </div>
        <div class="summary-card">
            <div class="value">{summary.get('avg_latency_ms', 0):.0f}ms</div>
            <div class="label">Avg Latency</div>
        </div>
    </div>

    <div class="section">
        <h2>Retrieval Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th><th>Score</th><th>Visual</th>
                </tr>
            </thead>
            <tbody>
                {self._table_row('Hit Rate', summary.get('hit_rate', 0))}
                {self._table_row('Mean Reciprocal Rank (MRR)', summary.get('mrr', 0))}
                {self._table_row('NDCG@5', summary.get('ndcg@k', 0))}
                {self._table_row(f"Precision@5", summary.get('precision@k', 0))}
                {self._table_row(f"Recall@5", summary.get('recall@k', 0))}
                {self._table_row('Average Precision', summary.get('avg_precision', 0))}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Generation Metrics</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th><th>Score</th><th>Visual</th>
                </tr>
            </thead>
            <tbody>
                {self._table_row('Faithfulness', summary.get('avg_faithfulness', 0))}
                {self._table_row('Answer Relevance', summary.get('avg_relevance', 0))}
                {self._table_row('Context Precision', summary.get('avg_context_precision', 0))}
                {self._table_row('Context Recall', summary.get('avg_context_recall', 0))}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Detailed Question Results</h2>
        {self._render_questions(results)}
    </div>

    <div class="footer">
        Medical RAG Chatbot Evaluation Report | {metadata.get('model', 'Unknown')} | {metadata.get('embedding_model', '')}
    </div>
</div>
</body>
</html>"""

        output = output_path or str(self.output_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html")
        with open(output, "w", encoding="utf-8") as f:
            f.write(html)

        logger.info(f"HTML report saved: {output}")
        return output

    def generate_csv_report(
        self,
        results: list[dict[str, Any]],
        output_path: str | None = None,
    ) -> str:
        """Generate a CSV report with all evaluation results."""
        output = output_path or str(self.output_dir / f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

        if not results:
            return output

        fieldnames = [
            "id", "question", "category", "difficulty",
            "hit_rate", "mrr", "ndcg@k", "precision@k", "recall@k",
            "faithfulness", "answer_relevance", "context_precision", "context_recall",
            "llm_judge_accuracy", "llm_judge_completeness", "llm_judge_clarity",
            "llm_judge_safety", "llm_judge_hallucination", "llm_judge_overall",
            "latency_ms", "sources_count", "status",
        ]

        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for r in results:
                row = {
                    "id": r.get("id", ""),
                    "question": r.get("question", ""),
                    "category": r.get("category", ""),
                    "difficulty": r.get("difficulty", ""),
                    "hit_rate": r.get("retrieval_metrics", {}).get("hit_rate", ""),
                    "mrr": r.get("retrieval_metrics", {}).get("mrr", ""),
                    "ndcg@k": r.get("retrieval_metrics", {}).get("ndcg@k", ""),
                    "precision@k": r.get("retrieval_metrics", {}).get(f"precision@k", ""),
                    "recall@k": r.get("retrieval_metrics", {}).get(f"recall@k", ""),
                    "faithfulness": r.get("generation_metrics", {}).get("faithfulness", ""),
                    "answer_relevance": r.get("generation_metrics", {}).get("answer_relevance", ""),
                    "context_precision": r.get("generation_metrics", {}).get("context_precision", ""),
                    "context_recall": r.get("generation_metrics", {}).get("context_recall", ""),
                    "latency_ms": r.get("latency_ms", ""),
                    "sources_count": len(r.get("sources", [])),
                    "status": r.get("status", ""),
                }
                llm_judge = r.get("llm_judge", {})
                row.update({
                    "llm_judge_accuracy": llm_judge.get("accuracy", ""),
                    "llm_judge_completeness": llm_judge.get("completeness", ""),
                    "llm_judge_clarity": llm_judge.get("clarity", ""),
                    "llm_judge_safety": llm_judge.get("safety", ""),
                    "llm_judge_hallucination": llm_judge.get("hallucination", ""),
                    "llm_judge_overall": llm_judge.get("overall", ""),
                })
                writer.writerow(row)

        logger.info(f"CSV report saved: {output}")
        return output

    def _badge(self, value: float, good: float, warn: float) -> str:
        if value >= good:
            cls = "badge-good"
        elif value >= warn:
            cls = "badge-warning"
        else:
            cls = "badge-danger"
        return f'<span class="badge {cls}">{"✓" if value >= good else "⚠" if value >= warn else "✗"}</span>'

    def _table_row(self, label: str, value: float) -> str:
        pct = value * 100
        color = "#0062cc" if value >= 0.7 else "#ffc107" if value >= 0.5 else "#dc3545"
        return f"""<tr>
            <td>{label}</td>
            <td>{value:.4f} ({pct:.1f}%)</td>
            <td>
                <div class="score-bar">
                    <div class="score-bar-fill" style="width: {min(pct, 100):.1f}%; background: {color};"></div>
                </div>
            </td>
        </tr>"""

    def _render_questions(self, results: list[dict[str, Any]]) -> str:
        html_parts = []
        for r in results:
            q = r.get("question", "")
            ret = r.get("retrieval_metrics", {})
            gen = r.get("generation_metrics", {})
            status = r.get("status", "success")
            status_icon = "✓" if status == "success" else "⚠" if status == "no_results" else "✗"

            html_parts.append(f"""<div class="question-card">
                <div class="meta">
                    <span>[{status_icon}] {r.get('id', '?')} | {r.get('category', '')} | {r.get('difficulty', '')}
                </div>
                <div class="q">Q: {q}</div>
                <div class="metrics-row">
                    <span class="metric">Hit: <span class="val">{ret.get('hit_rate', 0):.0%}</span></span>
                    <span class="metric">MRR: <span class="val">{ret.get('mrr', 0):.2%}</span></span>
                    <span class="metric">NDCG: <span class="val">{ret.get('ndcg@k', 0):.2%}</span></span>
                    <span class="metric">Faith: <span class="val">{gen.get('faithfulness', 0):.0%}</span></span>
                    <span class="metric">Rel: <span class="val">{gen.get('answer_relevance', 0):.0%}</span></span>
                    <span class="metric">Latency: <span class="val">{r.get('latency_ms', 0):.0f}ms</span></span>
                </div>
            </div>""")
        return "\n".join(html_parts)
