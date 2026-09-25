"""Unit tests for follow-up question condensing (multi-turn retrieval)."""

from src.rag.query_condenser import condense_question

HISTORY = [
    {"role": "user", "content": "Bệnh áp xe não là gì, nguyên nhân và hậu quả ra sao?"},
    {"role": "assistant", "content": "Áp xe não là ổ mủ trong nhu mô não, thường do nhiễm khuẩn..."},
]


class FakeGenerator:
    def __init__(self, output="Bệnh áp xe não có cách chữa dứt điểm không?", error=None):
        self.output = output
        self.error = error
        self.calls = []

    def generate(self, prompt, system_prompt=None, max_tokens=None, temperature=None, **kwargs):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt,
                           "max_tokens": max_tokens, "temperature": temperature})
        if self.error:
            raise self.error
        return self.output


def test_no_history_returns_question_without_calling_llm():
    gen = FakeGenerator()
    assert condense_question("Có cách chữa nào dứt điểm không?", [], gen) == "Có cách chữa nào dứt điểm không?"
    assert gen.calls == []


def test_follow_up_is_rewritten_with_topic_from_history():
    """Regression: the follow-up used to be retrieved on its own, returning an unrelated disease."""
    gen = FakeGenerator()
    result = condense_question("Có cách chữa nào dứt điểm không?", HISTORY, gen)

    assert result == "Bệnh áp xe não có cách chữa dứt điểm không?"
    assert len(gen.calls) == 1
    call = gen.calls[0]
    assert "áp xe não" in call["prompt"]  # history reaches the rewriter
    assert "Có cách chữa nào dứt điểm không?" in call["prompt"]
    assert call["temperature"] == 0.0


def test_llm_failure_falls_back_to_original_question():
    gen = FakeGenerator(error=RuntimeError("API down"))
    assert condense_question("Có cách chữa nào dứt điểm không?", HISTORY, gen) == "Có cách chữa nào dứt điểm không?"


def test_empty_or_runaway_output_falls_back_to_original_question():
    q = "Có cách chữa nào dứt điểm không?"
    assert condense_question(q, HISTORY, FakeGenerator(output="   ")) == q
    assert condense_question(q, HISTORY, FakeGenerator(output="x " * 400)) == q


def test_output_is_cleaned_of_labels_and_quotes():
    gen = FakeGenerator(output='Câu hỏi độc lập: "Bệnh áp xe não có cách chữa dứt điểm không?"\n')
    assert condense_question("Có cách chữa nào dứt điểm không?", HISTORY, gen) == (
        "Bệnh áp xe não có cách chữa dứt điểm không?"
    )


def test_only_recent_turns_are_sent_and_long_messages_truncated():
    long_history = [{"role": "user", "content": f"câu hỏi cũ số {i}"} for i in range(20)]
    long_history.append({"role": "assistant", "content": "a" * 5000})
    gen = FakeGenerator()
    condense_question("Còn gì nữa không?", long_history, gen, max_messages=4, max_chars_per_message=300)

    prompt = gen.calls[0]["prompt"]
    assert "câu hỏi cũ số 0" not in prompt       # old turns dropped
    assert "câu hỏi cũ số 19" in prompt          # recent turns kept
    assert "a" * 301 not in prompt               # long answers truncated


def test_generator_without_generate_method_is_ignored():
    assert condense_question("Có cách chữa nào dứt điểm không?", HISTORY, object()) == (
        "Có cách chữa nào dứt điểm không?"
    )
