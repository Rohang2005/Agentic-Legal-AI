"""
Text chunker for splitting legal documents into overlapping chunks.

Designed for embedding and retrieval; preserves context with overlap.
"""

import re
from typing import List


class TextChunker:
    """
    Splits long text into chunks with configurable size and overlap.
    Tries to break on sentence boundaries when possible.
    Uses max_chars and max_chunks to avoid MemoryError on very large documents.
    """

    # Cap total text and number of chunks to avoid MemoryError
    DEFAULT_MAX_CHARS = 2_000_000  # ~2M characters
    DEFAULT_MAX_CHUNKS = 5000

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        separators: List[str] = None,
        max_chars: int = None,
        max_chunks: int = None,
    ):
        """
        Initialize the chunker.

        Args:
            chunk_size: Target size of each chunk in characters.
            chunk_overlap: Overlap between consecutive chunks (must be < chunk_size).
            separators: List of split points (default: paragraph, newline, sentence, space).
            max_chars: Max total characters to chunk (default 2M); rest is truncated.
            max_chunks: Max number of chunks to return (default 5000); prevents OOM.
        """
        self.chunk_size = min(chunk_size, 4096)  # sanity cap
        self.chunk_overlap = min(max(0, chunk_overlap), self.chunk_size - 1)  # ensure < chunk_size
        self.separators = separators or ["\n\n", "\n", ". ", " "]
        self.max_chars = max_chars if max_chars is not None else self.DEFAULT_MAX_CHARS
        self.max_chunks = max_chunks if max_chunks is not None else self.DEFAULT_MAX_CHUNKS

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple heuristic)."""
        text = text.strip()
        if not text:
            return []
        # Split on sentence-ending punctuation followed by space or end
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def chunk_by_size(self, text: str) -> List[str]:
        """
        Split text into chunks of roughly chunk_size with overlap.

        Uses separators to try to break at natural boundaries.
        Truncates to max_chars and stops after max_chunks to avoid MemoryError.
        """
        text = text.strip()
        if not text:
            return []

        # Truncate to avoid processing huge documents in one go
        length = min(len(text), self.max_chars)
        text = text[:length]

        chunks = []
        start = 0

        while start < length and len(chunks) < self.max_chunks:
            end = min(start + self.chunk_size, length)
            segment = text[start:end]

            if end < length:
                # Try to find a good break point within the segment
                best_break = -1
                for sep in self.separators:
                    idx = segment.rfind(sep)
                    if idx > self.chunk_size // 2:
                        best_break = idx + len(sep)
                        break
                if best_break > 0:
                    segment = segment[:best_break]
                    end = start + best_break

            segment = segment.strip()
            if segment:
                chunks.append(segment)

            next_start = end - self.chunk_overlap
            # Ensure we always advance to prevent infinite loop (e.g. if overlap >= size)
            if next_start <= start:
                next_start = end
            start = next_start
            if start >= length:
                break
            if start < 0:
                start = 0

        return chunks

    def chunk(self, text: str) -> List[str]:
        """
        Chunk the input text. Alias for chunk_by_size.

        Args:
            text: Full document text.

        Returns:
            List of text chunks.
        """
        return self.chunk_by_size(text)
