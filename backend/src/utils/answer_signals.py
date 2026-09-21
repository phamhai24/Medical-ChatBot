"""Heuristics for judging whether a generated RAG answer actually used its retrieved context.

These patterns were tuned empirically against real generations (see
reports/Independent_Benchmark_Report_v1_20260916.md and the citation audit that
followed it): phrases like "tham khảo bác sĩ" were deliberately excluded because
the system prompt appends a doctor-consultation disclaimer to nearly every
answer, good or bad, which would make that phrase useless as a decline signal.
"""

DECLINE_PATTERNS = [
    "không tìm thấy",
    "không có thông tin",
    "không có dữ liệu",
    "không thể xác nhận",
    "không có cơ sở",
    "không đủ thông tin",
    "chưa được ghi nhận",
    "không có bằng chứng",
    "không thể cung cấp",
    "không thể xác định",
    "không thể đưa ra",
]

# Beyond this vector-store distance, a retrieved doc is unlikely to genuinely
# support an answer. Chosen from empirical distance distributions: correctly-
# cited sources cluster around 0.21-0.47 (p90), while sources shown alongside
# an honest "not found" answer average 0.54. The two distributions overlap, so
# this is a secondary safety net, not a precise classifier on its own.
SOURCE_RELEVANCE_THRESHOLD = 0.55


def looks_like_decline(answer: str) -> bool:
    """True if the answer text reads as an honest "I didn't find this" rather than a real answer."""
    lower = (answer or "").lower()
    return any(p in lower for p in DECLINE_PATTERNS)
