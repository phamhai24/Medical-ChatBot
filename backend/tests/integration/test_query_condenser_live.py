"""Live checks of follow-up condensing against the real API LLM.

Skipped unless OPENAI_API_KEY is configured. The unit tests in
tests/unit/test_query_condenser.py cover the plumbing with a fake generator;
these cover what only the real model can: whether it rewrites the follow-ups
users actually send, and leaves new topics alone.

Run: pytest tests/integration/test_query_condenser_live.py -v
"""

import pytest

from src.core.config import get_settings
from src.rag.api_generator import APIGenerator
from src.rag.query_condenser import resolve_question

settings = get_settings()
pytestmark = pytest.mark.skipif(
    not settings.api_generator_api_key, reason="needs an API key for the generator"
)

# A real session (verbatim, as the condenser receives it: answers cut at 400
# chars). Shortened paraphrases of these answers did NOT reproduce the bug —
# the real second answer is a "không tìm thấy thông tin" reply, and after it
# the model kept the next follow-up unchanged.
HISTORY = [
    {"role": "user", "content": "bệnh áp xe não nguyên nhân do đâu? Hậu quả nếu không phát hiện kịp thời sẽ như thế nào"},
    {"role": "assistant", "content": (
        "Bệnh áp xe não có nhiều nguyên nhân gây ra, trong đó thường gặp nhất là nhiễm khuẩn từ các "
        "ổ nhiễm trùng gần sọ não như viêm tai, viêm xoang, hoặc từ chấn thương ở đầu. Ngoài ra, áp xe "
        "não cũng có thể hình thành do nhiễm khuẩn theo đường máu từ các bệnh như áp-xe phổi, viêm màng "
        "phổi, hoặc viêm tủy xương [1][2]. Các vi khuẩn gây áp xe não thường gặp bao gồm Bacteroides, "
        "tụ cầu, liên cầu, và trong "
    )},
    {"role": "user", "content": "có phương pháp nào chữa dứt điểm tại nhà không"},
    {"role": "assistant", "content": (
        "Tôi không tìm thấy thông tin nào trong dữ liệu được cung cấp về phương pháp chữa dứt điểm bệnh "
        "áp xe não tại nhà. Bệnh áp xe não là một tình trạng nghiêm trọng và cần được điều trị bởi các "
        "bác sĩ chuyên khoa. Việc điều trị thường bao gồm sử dụng kháng sinh, thuốc hỗ trợ và có thể cần "
        "can thiệp phẫu thuật tùy thuộc vào tình trạng cụ thể của bệnh nhân. \n\nTôi khuyên bạn nên "
        "tham khảo ý kiến bác sĩ để"
    )},
]


@pytest.fixture(scope="module")
def generator():
    return APIGenerator(
        provider=settings.api_generator_provider,
        model=settings.api_generator_model,
        api_key=settings.api_generator_api_key,
    )


@pytest.mark.parametrize("follow_up", [
    # Real report: grammatical and "complete-sounding", but names no disease.
    "gợi ý cho tôi một vài loại thuốc chữa tại nhà",
    "có phương pháp nào chữa dứt điểm tại nhà không",
    "nó có lây không",
    "bệnh này kiêng ăn gì",
    "cần làm những xét nghiệm gì",
    # Phrasings unlike the prompt's examples, to check the rule generalizes.
    "trẻ em có bị không",
    "bao lâu thì khỏi",
    "có cần mổ không bác sĩ",
])
def test_follow_up_without_subject_gets_the_disease_from_history(generator, follow_up):
    resolved = resolve_question(follow_up, HISTORY, generator)
    assert "áp xe não" in resolved.question.lower(), resolved
    assert "áp xe não" in (resolved.topic or "").lower(), resolved


@pytest.mark.parametrize("new_topic, expected, forbidden", [
    ("Triệu chứng của bệnh gút là gì?", "gút", "áp xe"),
    ("thế còn viêm màng não thì sao, có nguy hiểm không?", "viêm màng não", "áp xe"),
])
def test_explicit_new_topic_is_not_pulled_back(generator, new_topic, expected, forbidden):
    resolved = resolve_question(new_topic, HISTORY, generator, previous_topic="áp xe não")
    assert expected in resolved.question.lower(), resolved
    assert forbidden not in resolved.question.lower(), resolved
    assert expected in (resolved.topic or "").lower(), resolved


def test_non_medical_message_is_left_alone_but_keeps_the_topic(generator):
    resolved = resolve_question("cảm ơn bạn nhiều", HISTORY, generator, previous_topic="áp xe não")
    assert "áp xe" not in resolved.question.lower(), resolved
    assert resolved.topic == "áp xe não", resolved


def test_follow_up_after_a_thank_you_still_resolves_via_tracked_topic(generator):
    """The last messages are small talk; only the tracked topic carries the disease."""
    small_talk = [
        {"role": "user", "content": "cảm ơn bạn nhiều"},
        {"role": "assistant", "content": "Không có gì, chúc bạn nhiều sức khỏe!"},
    ]
    resolved = resolve_question("vậy có cần kiêng ăn gì không", small_talk, generator,
                                previous_topic="áp xe não")
    assert "áp xe não" in resolved.question.lower(), resolved
