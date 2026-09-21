"""Unit tests for the TextChunker."""

import pytest
from src.rag.chunker import TextChunker


class TestTextChunker:
    def test_chunk_text_short_text(self):
        """Short text should return as single chunk."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64, min_chunk_length=50)
        text = "Triệu chứng bệnh tiểu đường."
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_long_text(self):
        """Long text should be split into multiple chunks."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20, min_chunk_length=30)
        text = "A" * 500
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1

    def test_chunk_text_empty(self):
        """Empty text should return empty list."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64)
        assert chunker.chunk_text("") == []
        assert chunker.chunk_text(None) == []

    def test_chunk_qa_pairs(self, sample_qa_data):
        """Q&A pairs should be chunked with question prepended."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64, min_chunk_length=50)
        documents = chunker.chunk_qa_pairs(sample_qa_data, include_question=True)

        assert len(documents) == len(sample_qa_data)
        for doc in documents:
            assert "chunk_id" in doc
            assert "text" in doc
            assert "question" in doc
            assert "Câu hỏi:" in doc["text"]
            assert "Trả lời:" in doc["text"]

    def test_chunk_qa_pairs_without_question(self, sample_qa_data):
        """Q&A pairs can be chunked without question prepended."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64, min_chunk_length=50)
        documents = chunker.chunk_qa_pairs(sample_qa_data, include_question=False)

        assert len(documents) == len(sample_qa_data)
        for doc in documents:
            assert "Câu hỏi:" not in doc["text"]

    def test_chunk_stats(self, sample_qa_data):
        """Chunk stats should be computed correctly."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64, min_chunk_length=50)
        documents = chunker.chunk_qa_pairs(sample_qa_data)
        stats = chunker.get_chunk_stats(documents)

        assert "total_chunks" in stats
        assert stats["total_chunks"] == len(documents)
        assert "avg_length" in stats
        assert "min_length" in stats
        assert "max_length" in stats

    def test_chunk_ids_unique(self, sample_qa_data):
        """All chunk IDs should be unique."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64, min_chunk_length=50)
        documents = chunker.chunk_qa_pairs(sample_qa_data)
        ids = [doc["chunk_id"] for doc in documents]
        assert len(ids) == len(set(ids))

    def test_chunk_text_never_splits_mid_sentence_when_boundaries_exist(self):
        """Paragraph/sentence-packing should keep every chunk ending on a real boundary."""
        paragraphs = [
            "Đây là đoạn đầu tiên nói về bệnh cao huyết áp. Nó khá dài để test.",
            "Đoạn thứ hai nói về nguyên nhân gây bệnh và các yếu tố nguy cơ liên quan.",
            "Đoạn thứ ba nói về cách điều trị và phòng ngừa bệnh hiệu quả nhất.",
        ]
        text = "\n\n".join(paragraphs)
        chunker = TextChunker(chunk_size=80, chunk_overlap=0, min_chunk_length=10)

        chunks = chunker.chunk_text(text)

        # Every chunk should end with sentence-ending punctuation (or be the
        # single final leftover), never a mid-word/mid-sentence hard cut.
        for chunk in chunks:
            assert chunk.rstrip()[-1] in ".!?", f"chunk did not end on a boundary: {chunk!r}"

    def test_chunk_text_hard_slices_only_when_no_punctuation_at_all(self):
        """Text with real sentence punctuation is packed by sentence, not hard-sliced."""
        # One long sentence, but it does end with punctuation — same text with a
        # period is one sentence-unit and, since it doesn't exceed chunk_size
        # itself here, is kept whole rather than hard-sliced mid-word.
        sentence = "Đây là một câu " + ("rất " * 10) + "dài."
        chunker = TextChunker(chunk_size=len(sentence) + 10, chunk_overlap=0, min_chunk_length=10)
        assert chunker.chunk_text(sentence) == [sentence]

        # No sentence-ending punctuation anywhere and longer than chunk_size:
        # the bounded-size hard-slice safety net kicks in (documented tradeoff;
        # this should not occur in real prose).
        no_punctuation = "rất " * 40
        chunker2 = TextChunker(chunk_size=50, chunk_overlap=0, min_chunk_length=10)
        chunks = chunker2.chunk_text(no_punctuation)
        assert len(chunks) > 1
        # All but possibly the last (which may absorb a too-short trailing
        # piece per the min_chunk_length merge, rather than dropping it).
        assert all(len(c) <= 50 for c in chunks[:-1])

    def test_chunk_text_respects_chunk_size_for_normal_paragraphs(self):
        """Packed chunks shouldn't meaningfully exceed chunk_size (barring one oversized unit)."""
        text = "\n\n".join(f"Đoạn số {i} có một vài câu ngắn để test việc đóng gói." for i in range(20))
        chunker = TextChunker(chunk_size=200, chunk_overlap=0, min_chunk_length=10)

        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 200 + 50  # small slack for the paragraph join

    def test_chunk_text_bounds_a_single_oversized_sentence_among_others(self):
        """A run-on sentence longer than chunk_size, mixed with short ones, must still be bounded."""
        short = "Câu ngắn."
        long_sentence = "Đây là một câu rất dài không có dấu chấm ở giữa " + ("từ " * 100) + "kết thúc."
        text = f"{short} {long_sentence} {short}"
        chunker = TextChunker(chunk_size=100, chunk_overlap=0, min_chunk_length=10)

        chunks = chunker.chunk_text(text)

        assert all(len(c) <= 100 for c in chunks), [len(c) for c in chunks]

    def test_chunk_qa_pairs_adds_part_hint_only_for_multi_chunk_records(self, sample_qa_data):
        """A single-chunk record's title shouldn't get a '(phần 1/1)' suffix that adds no info."""
        chunker = TextChunker(chunk_size=512, chunk_overlap=64, min_chunk_length=50)
        documents = chunker.chunk_qa_pairs(sample_qa_data)

        for doc in documents:
            assert "(phần" not in doc["text"]  # sample data is short, single-chunk

        long_answer = "\n\n".join(f"Đoạn {i}: " + "nội dung " * 30 for i in range(10))
        multi_chunk_data = [{"id": "x", "question": "Câu hỏi dài", "answer": long_answer}]
        multi_docs = chunker.chunk_qa_pairs(multi_chunk_data)

        assert len(multi_docs) > 1
        assert all(f"(phần {i + 1}/{len(multi_docs)})" in doc["text"] for i, doc in enumerate(multi_docs))
