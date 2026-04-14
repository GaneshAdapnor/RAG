"""
Text chunking with sliding window and configurable overlap.

WHY CHARACTER-BASED CHUNKING (not token-based):
    Token-based chunking (using tiktoken or a HuggingFace tokenizer) requires
    loading a tokenizer and adds ~50ms per document. For a production system
    using sentence-transformers, character count is a reliable proxy because
    `all-MiniLM-L6-v2` has a 256-word-piece limit ≈ 1,500–2,000 characters.
    Our default of 500 characters sits comfortably inside that limit.

CHUNK SIZE RATIONALE (500 chars, 50 char overlap):
    - 500 chars ≈ 80–120 words ≈ 3–6 sentences for typical prose.
    - This is a "goldilocks" size:
        * Too small (< 100 chars): A sentence may be split mid-clause. The
          retrieved chunk won't have enough context to answer most questions.
        * Too large (> 1,000 chars): The embedding has to represent too many
          topics at once, diluting similarity scores for specific queries.
    - 50-char overlap (10% of chunk size) prevents information loss at
      boundaries. E.g., a key fact straddling the boundary appears in both
      adjacent chunks, so at least one will be retrieved.

    TRADE-OFFS:
        Precision vs Recall:
          Smaller chunks → higher precision (retrieved text is tightly focused)
          Larger chunks  → higher recall (less risk of missing context)
          We bias slightly toward precision, trusting the LLM to compose an
          answer from 3–5 small, focused chunks rather than 1 large noisy one.

        Context preservation:
          The 50-char overlap ensures sentence continuity. If chunk N ends with
          "...which resulted in", chunk N+1 also starts from that sentence,
          preserving the full thought in at least one chunk.

        Cost implications:
          LLM APIs charge per token. Smaller chunks keep the context window
          tight. With top_k=5 and 500-char chunks, the max context sent to the
          LLM is ~2,500 chars ≈ 600 tokens — negligible cost per query.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from app.utils.text_extraction import PageContent

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """
    A single text chunk ready for embedding and indexing.

    Fields align with FAISS metadata: doc_id and chunk_id together form the
    primary key; page is null for TXT sources.
    """

    doc_id: str
    chunk_id: int           # Zero-based index within this document
    text: str
    page: Optional[int]     # 1-based PDF page; None for TXT
    char_start: int         # Character offset of chunk start in its page text
    char_end: int           # Character offset of chunk end in its page text


def chunk_pages(
    pages: List[PageContent],
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[TextChunk]:
    """
    Chunk a list of page texts using a sliding window algorithm.

    Algorithm:
        For each page, apply the sliding window independently. This is
        intentional: we do NOT merge pages before chunking because that would
        make page-number attribution impossible. Each page's text is split into
        overlapping windows of `chunk_size` characters, stepping `chunk_size -
        chunk_overlap` characters between windows.

    Args:
        pages: Extracted page contents from text_extraction.
        doc_id: UUID of the parent document.
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Number of characters to overlap between consecutive chunks.

    Returns:
        Ordered list of TextChunk objects with sequential chunk_ids.

    Raises:
        ValueError: If chunk_overlap >= chunk_size (would produce infinite loop).
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})."
        )

    chunks: List[TextChunk] = []
    chunk_id = 0
    step = chunk_size - chunk_overlap

    for page_content in pages:
        text = page_content.text
        text_len = len(text)

        if text_len == 0:
            continue

        # Sliding window over the page text
        start = 0
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk_text = text[start:end].strip()

            # Skip chunks that are pure whitespace (e.g., a page is mostly newlines)
            if not chunk_text:
                start += step
                continue

            # Guard: skip micro-chunks that are too small to be meaningful
            # (can occur at the tail of a short page)
            if len(chunk_text) < 20 and start > 0:
                logger.debug(
                    "doc_id=%s: Skipping micro-chunk at position %d (length %d).",
                    doc_id, start, len(chunk_text),
                )
                break

            chunks.append(
                TextChunk(
                    doc_id=doc_id,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    page=page_content.page,
                    char_start=start,
                    char_end=end,
                )
            )
            chunk_id += 1
            start += step

    logger.info(
        "doc_id=%s: Generated %d chunks from %d pages (chunk_size=%d, overlap=%d).",
        doc_id, len(chunks), len(pages), chunk_size, chunk_overlap,
    )
    return chunks
