"""Unit tests for follow-up resolution (multi-turn topic tracking).

The LLM only names the topic and proposes a standalone question (as JSON);
the rules that make retrieval safe are enforced in code and tested here with a
fake generator. Real-model behavior is covered by
tests/integration/test_query_condenser_live.py.
"""

import json

from src.rag.query_condenser import question_with_topic, resolve_question

HISTORY = [
    {"role": "user", "content": "Bệnh áp xe não là gì, nguyên nhân và hậu quả ra sao?"},
    {"role": "assistant", "content": "Áp xe não là ổ mủ trong nhu mô não, thường do nhiễm khuẩn..."},
]
FOLLOW_UP = "gợi ý cho tôi một vài loại thuốc chữa tại nhà"


class FakeGenerator:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    def generate(self, prompt, system_prompt=None, max_tokens=None, temperature=None, **kwargs):
        self.calls.append({"prompt": prompt, "system_prompt": system_prompt, "temperature": temperature})
        if self.error:
            raise self.error
        return self.output


def llm(topic, question):
    return FakeGenerator(output=json.dumps({"topic": topic, "question": question}, ensure_ascii=False))


# ─── No LLM call needed ──────────────────────────────────────────────────────

def test_first_turn_is_unchanged_and_makes_no_llm_call():
    gen = llm("áp xe não", "x")
    result = resolve_question("Bệnh áp xe não là gì?", [], gen)
    assert result.question == "Bệnh áp xe não là gì?"
    assert result.topic is None
    assert gen.calls == []


# ─── Code-enforced rules ─────────────────────────────────────────────────────

def test_topic_is_inserted_by_code_when_the_llm_leaves_the_question_unchanged():
    """Regression: the old prompt-only approach kept this question as-is (it is
    grammatical), so it was retrieved without the disease."""
    result = resolve_question(FOLLOW_UP, HISTORY, llm("áp xe não", FOLLOW_UP))
    assert "áp xe não" in result.question.lower()
    assert result.topic == "áp xe não"


def test_llm_rewrite_that_already_names_the_topic_is_used_as_is():
    rewritten = "Gợi ý vài loại thuốc chữa áp xe não tại nhà"
    result = resolve_question(FOLLOW_UP, HISTORY, llm("áp xe não", rewritten))
    assert result.question == rewritten


def test_topic_match_ignores_case_and_hyphens():
    rewritten = "Thuốc chữa Áp-xe não tại nhà"
    result = resolve_question(FOLLOW_UP, HISTORY, llm("áp xe não", rewritten))
    assert result.question == rewritten


def test_explicit_new_topic_replaces_the_previous_one():
    q = "Triệu chứng của bệnh gút là gì?"
    result = resolve_question(q, HISTORY, llm("gút", q), previous_topic="áp xe não")
    assert result.question == q
    assert result.topic == "gút"


def test_non_medical_message_keeps_the_previous_topic_for_later_turns():
    result = resolve_question("cảm ơn nhé", HISTORY, llm(None, "cảm ơn nhé"), previous_topic="áp xe não")
    assert result.question == "cảm ơn nhé"
    assert result.topic == "áp xe não"


def test_non_medical_message_is_not_answered_about_the_carried_topic():
    """Regression: "cảm ơn nhé" was sent to the answer model with the note
    "(Chủ đề đang trao đổi: tiểu đường type 2)" and got a diabetes answer. The
    topic is carried to the NEXT turn, but this message is not about it."""
    result = resolve_question("cảm ơn nhé", HISTORY, llm(None, "cảm ơn nhé"), previous_topic="tiểu đường")
    assert result.question_topic is None
    assert question_with_topic(result.question, result.question_topic) == "cảm ơn nhé"


def test_medical_follow_up_is_about_its_topic():
    result = resolve_question(FOLLOW_UP, HISTORY, llm("áp xe não", FOLLOW_UP))
    assert result.question_topic == "áp xe não"


def test_topic_found_neither_in_message_nor_history_falls_back_to_previous_topic():
    """A topic the model invented (not in the message, not in the conversation)
    is rejected by code in favour of the tracked topic."""
    result = resolve_question(FOLLOW_UP, HISTORY, llm("viêm xoang", FOLLOW_UP), previous_topic="áp xe não")
    assert result.topic == "áp xe não"
    assert "áp xe não" in result.question.lower()


def test_previous_topic_alone_is_enough_without_history():
    gen = llm("áp xe não", FOLLOW_UP)
    result = resolve_question(FOLLOW_UP, [], gen, previous_topic="áp xe não")
    assert "áp xe não" in result.question.lower()
    assert "áp xe não" in gen.calls[0]["prompt"]


# ─── Failure handling ────────────────────────────────────────────────────────

def test_llm_error_falls_back_to_previous_topic_insertion():
    """Even with the LLM down, a follow-up keeps the tracked topic."""
    result = resolve_question(FOLLOW_UP, HISTORY, FakeGenerator(error=RuntimeError("API down")),
                              previous_topic="áp xe não")
    assert "áp xe não" in result.question.lower()
    assert result.topic == "áp xe não"


def test_llm_error_without_a_tracked_topic_returns_the_original_question():
    result = resolve_question(FOLLOW_UP, HISTORY, FakeGenerator(error=RuntimeError("API down")))
    assert result.question == FOLLOW_UP


def test_invalid_json_is_handled_like_an_llm_error():
    result = resolve_question(FOLLOW_UP, HISTORY, FakeGenerator(output="không phải JSON"),
                              previous_topic="áp xe não")
    assert "áp xe não" in result.question.lower()


def test_runaway_rewrite_is_ignored():
    result = resolve_question(FOLLOW_UP, HISTORY, llm("áp xe não", "x " * 400))
    assert len(result.question) < 200
    assert "áp xe não" in result.question.lower()


def test_generator_without_generate_method_is_ignored():
    result = resolve_question(FOLLOW_UP, HISTORY, object(), previous_topic="áp xe não")
    assert "áp xe não" in result.question.lower()


# ─── Prompt contents ─────────────────────────────────────────────────────────

def test_only_recent_turns_are_sent_and_long_messages_truncated():
    long_history = [{"role": "user", "content": f"câu hỏi cũ số {i}"} for i in range(20)]
    long_history.append({"role": "assistant", "content": "a" * 5000})
    gen = llm("x", "x")
    resolve_question("Còn gì nữa không?", long_history, gen, max_messages=4, max_chars_per_message=300)

    prompt = gen.calls[0]["prompt"]
    assert "câu hỏi cũ số 0" not in prompt
    assert "câu hỏi cũ số 19" in prompt
    assert "a" * 301 not in prompt
    assert gen.calls[0]["temperature"] == 0.0


# ─── Generation guard (B) ────────────────────────────────────────────────────

def test_question_with_topic_adds_the_topic_note_for_the_answer_model():
    assert question_with_topic("có lây không", None) == "có lây không"
    noted = question_with_topic("có lây không", "áp xe não")
    assert noted.startswith("có lây không")
    assert "Chủ đề đang trao đổi: áp xe não" in noted
