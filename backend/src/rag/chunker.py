"""Text chunking for medical Q&A data"""

import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class TextChunker:
    """
    Split documents into smaller chunks for embedding.

    Packs whole paragraphs (falling back to whole sentences for any paragraph
    that alone exceeds chunk_size) greedily up to chunk_size, instead of
    cutting at a fixed character offset and searching backward for a nearby
    break — that approach silently falls back to a hard mid-sentence cut
    whenever no paragraph/sentence boundary happens to fall in its search
    window (the last 30-50% of the chunk). Paragraph-packing chunks always
    end at a real sentence or paragraph boundary — except for the
    pathological case of text with no sentence-ending punctuation at all
    longer than chunk_size, which still gets hard-sliced as a bounded-size
    safety net (this should essentially never happen in real prose).

    Default chunk_size is 1200 chars (raised from 512): this corpus's records
    are frequently long articles (median ~9.7k chars, 26% exceed 20k, some
    reach 400k+), and at 512 chars they fragment into so many chunks per
    article that retrieval's top-k can end up dominated by several chunks of
    one weakly-relevant document instead of genuinely different candidates
    (see reports/ for real examples). 1200 was chosen from real token counts
    against this project's embedding model (paraphrase-multilingual-MiniLM,
    512-token limit): ~1200 Vietnamese chars measured at ~330 tokens average
    (max observed 380 across a corpus sample) — enough margin below 512 to
    not silently truncate at embed time, while cutting total corpus chunks by
    roughly 60% (649k -> ~261k, estimated from real record-length data).
    """

    def __init__(
        self,
        chunk_size: int = 1200,
        chunk_overlap: int = 100,
        min_chunk_length: int = 50,
        separator: str = "\n"
    ):
        """
        Args:
            chunk_size: Maximum number of characters per chunk
            chunk_overlap: A unit (paragraph/sentence) up to this many characters
                is carried over into the next chunk for continuity across the
                boundary. 0 disables carry-over.
            min_chunk_length: Minimum chunk length to keep; a trailing chunk
                shorter than this is merged into the previous one instead of
                being dropped, so no content is silently lost.
            separator: Unused, kept for backward-compatible construction.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length
        self.separator = separator

    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks, preferring paragraph then sentence boundaries."""
        if not text or len(text.strip()) < self.min_chunk_length:
            return [text] if text else []

        units = self._split_into_units(text)
        chunks = self._pack_units(units)

        # Merge a too-short trailing chunk into the previous one rather than
        # dropping content that didn't happen to fill a whole chunk. Pop first,
        # then index: `chunks[-2] = f(chunks.pop())` evaluates chunks.pop() as
        # part of the right-hand side, which mutates the list's length before
        # the left-hand chunks[-2] index is resolved against it — silently
        # merging into the wrong element.
        if len(chunks) > 1 and len(chunks[-1]) < self.min_chunk_length:
            last = chunks.pop()
            chunks[-1] = f"{chunks[-1]}\n\n{last}"

        return chunks

    def _split_into_units(self, text: str) -> List[str]:
        """Paragraphs, with any paragraph bigger than chunk_size further split by sentence."""
        paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        units = []
        for para in paragraphs:
            if len(para) <= self.chunk_size:
                units.append(para)
            else:
                units.extend(self._split_oversized_paragraph(para))
        return units

    def _split_oversized_paragraph(self, text: str) -> List[str]:
        """Pack a paragraph's sentences up to chunk_size; hard-slice only as a last resort."""
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if len(sentences) <= 1:
            return self._hard_slice(text)

        pieces = []
        buf = ""
        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                # This single sentence alone exceeds chunk_size (e.g. a long
                # run-on list with no internal punctuation) — flush whatever's
                # buffered first, then hard-slice this sentence on its own so
                # it can't silently produce an oversized chunk downstream.
                if buf:
                    pieces.append(buf)
                    buf = ""
                pieces.extend(self._hard_slice(sentence))
                continue
            candidate = f"{buf} {sentence}".strip() if buf else sentence
            if buf and len(candidate) > self.chunk_size:
                pieces.append(buf)
                buf = sentence
            else:
                buf = candidate
        if buf:
            pieces.append(buf)
        return pieces

    def _hard_slice(self, text: str) -> List[str]:
        """Bounded-size fallback slice, used only when no sentence boundary is usable."""
        return [text[i:i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]

    def _pack_units(self, units: List[str]) -> List[str]:
        """Greedily pack paragraph/sentence units into chunks up to chunk_size."""
        if not units:
            return []

        chunks = []
        buf: List[str] = []

        for unit in units:
            candidate = buf + [unit]
            if buf and len("\n\n".join(candidate)) > self.chunk_size:
                chunks.append("\n\n".join(buf))
                if self.chunk_overlap > 0 and len(buf[-1]) <= self.chunk_overlap:
                    buf = [buf[-1], unit]
                else:
                    buf = [unit]
            else:
                buf = candidate

        if buf:
            chunks.append("\n\n".join(buf))

        return chunks

    def chunk_qa_pairs(
        self,
        data: List[Dict[str, Any]],
        include_question: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Chunk Q&A pairs from medical dataset.

        Args:
            data: List of {"question": ..., "answer": ...} records
            include_question: Whether to include question in chunk context

        Returns:
            List of chunked documents with metadata
        """
        documents = []
        doc_id = 0

        for record in data:
            question = record.get("question", "")
            answer = record.get("answer", "")

            if not answer:
                continue

            # Chunk the answer (answers are typically longer)
            answer_chunks = self.chunk_text(answer)
            total = len(answer_chunks)

            for chunk_idx, chunk_text in enumerate(answer_chunks):
                chunk_id = f"{doc_id}_{chunk_idx}"

                if include_question:
                    # Prepend question for better context. For a multi-chunk
                    # article, also note which part this is — the title alone
                    # describes the whole article, not necessarily what this
                    # specific chunk covers (see class docstring).
                    title = f"{question} (phần {chunk_idx + 1}/{total})" if total > 1 else question
                    combined_text = f"Câu hỏi: {title}\n\nTrả lời: {chunk_text}"
                else:
                    combined_text = chunk_text

                documents.append({
                    "chunk_id": chunk_id,
                    "text": combined_text,
                    "question": question,
                    "chunk_index": chunk_idx,
                    "total_chunks": total,
                    "source": "medical_qa"
                })

            doc_id += 1

        logger.info(f"Created {len(documents)} chunks from {doc_id} Q&A pairs")
        return documents

    def get_chunk_stats(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about chunks."""
        if not documents:
            return {}

        lengths = [len(doc["text"]) for doc in documents]

        return {
            "total_chunks": len(documents),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_characters": sum(lengths),
        }
