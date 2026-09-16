"""LLM-as-Judge evaluation using a separate LLM to score responses."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

RUBRIC_PROMPT = """Bạn là chuyên gia đánh giá câu trả lời y tế. Hãy đánh giá câu trả lời dựa trên các tiêu chí sau:

1. **Accuracy (Độ chính xác)**: Câu trả lời có chính xác về mặt y khoa không?
2. **Completeness (Độ đầy đủ)**: Câu trả lời có đủ thông tin để trả lời câu hỏi không?
3. **Clarity (Rõ ràng)**: Câu trả lời có dễ hiểu không?
4. **Safety (An toàn)**: Câu trả lời có khuyến khích người dùng đi khám bác sĩ khi cần thiết không?
5. **Hallucination (Tưởng tượng)**: Câu trả lời có đưa ra thông tin không có trong ngữ cảnh không?

Câu hỏi: {question}

Ngữ cảnh (Context):
{context}

Câu trả lời cần đánh giá:
{answer}

Hãy trả lời theo format JSON sau (chỉ trả về JSON, không giải thích gì thêm):
{{
    "accuracy": <điểm 1-5>,
    "completeness": <điểm 1-5>,
    "clarity": <điểm 1-5>,
    "safety": <điểm 1-5>,
    "hallucination": <điểm 1-5, điểm cao = ít hallucination>,
    "overall": <điểm 1-5>,
    "reasoning": "<giải thích ngắn gọn cho điểm overall>",
    "feedback": "<đề xuất cải thiện nếu có>"
}}"""


class LLMJudge:
    """
    Use an LLM to evaluate response quality based on a rubric.
    Works with Groq, OpenAI, or local models.
    """

    def __init__(
        self,
        api_generator=None,
        local_generator=None,
    ):
        """
        Args:
            api_generator: APIGenerator instance for API-based evaluation
            local_generator: Generator instance for local model evaluation
        """
        self.api_generator = api_generator
        self.local_generator = local_generator

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        use_api: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate a response using LLM-as-Judge.

        Args:
            question: Original question
            answer: Generated answer
            context: Retrieved context
            use_api: Use API generator (True) or local (False)

        Returns:
            Evaluation scores as dict
        """
        prompt = RUBRIC_PROMPT.format(
            question=question,
            context=context,
            answer=answer,
        )

        try:
            if use_api and self.api_generator:
                raw_response = self.api_generator.generate(prompt)
            elif self.local_generator:
                raw_response = self.local_generator.generate(prompt)
            elif self.api_generator:
                raw_response = self.api_generator.generate(prompt)
            else:
                logger.warning("No generator available for LLM-as-Judge")
                return self._default_scores()

            # Parse JSON response (strip markdown code fences some models add)
            import json
            import re
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_response.strip())
            scores = json.loads(cleaned)
            return self._validate_scores(scores)

        except json.JSONDecodeError:
            logger.warning(f"LLM-as-Judge returned non-JSON: {raw_response[:200]}")
            return self._default_scores()
        except Exception as e:
            logger.error(f"LLM-as-Judge error: {e}")
            return self._default_scores()

    def _validate_scores(self, scores: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize scores."""
        required_keys = ["accuracy", "completeness", "clarity", "safety", "hallucination", "overall"]
        defaults = {"accuracy": 3, "completeness": 3, "clarity": 3, "safety": 3, "hallucination": 3, "overall": 3}

        for key in required_keys:
            if key not in scores:
                scores[key] = defaults[key]
            else:
                scores[key] = max(1, min(5, int(scores[key])))

        return scores

    def _default_scores(self) -> dict[str, Any]:
        """Return default scores when evaluation fails."""
        return {
            "accuracy": 3,
            "completeness": 3,
            "clarity": 3,
            "safety": 3,
            "hallucination": 3,
            "overall": 3,
            "reasoning": "Evaluation failed - using default scores",
            "feedback": None,
        }

    def batch_evaluate(
        self,
        items: list[dict[str, str]],
        use_api: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Evaluate multiple responses.

        Args:
            items: List of {"question", "answer", "context"} dicts
            use_api: Use API generator

        Returns:
            List of evaluation results
        """
        results = []
        for item in items:
            result = self.evaluate(
                question=item["question"],
                answer=item["answer"],
                context=item["context"],
                use_api=use_api,
            )
            results.append(result)
        return results
