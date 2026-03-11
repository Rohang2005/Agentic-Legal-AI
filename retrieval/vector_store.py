"""
FAISS vector store for legal document chunks.

Stores embeddings from BGE model and supports similarity search for RAG.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.embedding_model import EmbeddingModel


class LegalVectorStore:
    """
    FAISS index over legal document chunks with metadata (text) storage.
    Uses BAAI/bge-large-en-v1.5 embeddings.
    """

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None, index_path: Optional[Path] = None):
        """
        Args:
            embedding_model: Model for encoding text. If None, creates default BGE model.
            index_path: Optional path to save/load FAISS index and metadata.
        """
        self.embedding_model = embedding_model or EmbeddingModel()
        self.index_path = Path(index_path) if index_path else None
        self._index: Optional["faiss.Index"] = None
        self._texts: List[str] = []

    def _get_dim(self) -> int:
        return self.embedding_model.load().get_sentence_embedding_dimension()

    def add_texts(self, texts: List[str], batch_size: int = 32) -> None:
        """
        Encode texts and add to FAISS index.

        Args:
            texts: List of chunk strings.
            batch_size: Batch size for encoding.
        """
        if faiss is None:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
        if not texts:
            return
        vectors = self.embedding_model.encode(texts, batch_size=batch_size)
        vectors = np.array(vectors).astype("float32")
        dim = vectors.shape[1]
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)  # Inner product for normalized vectors = cosine sim
        self._index.add(vectors)
        self._texts.extend(texts)

    def similarity_search(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Return top-k most similar chunks to the query.

        Args:
            query: Query string.
            k: Number of results.

        Returns:
            List of (chunk_text, score) tuples. Score is cosine similarity (higher = more similar).
        """
        if self._index is None or not self._texts:
            return []
        q = self.embedding_model.encode_single(query)
        q = np.array([q], dtype="float32")
        scores, indices = self._index.search(q, min(k, len(self._texts)))
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self._texts):
                results.append((self._texts[idx], float(scores[0][i])))
        return results

    def save(self, path: Optional[Path] = None) -> None:
        """Save FAISS index to disk. Metadata (texts) saved as .txt companion file."""
        path = path or self.index_path
        if not path or self._index is None:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path.with_suffix(".faiss")))
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._texts, f, ensure_ascii=False, indent=0)

    def load(self, path: Optional[Path] = None) -> None:
        """Load FAISS index and metadata from disk."""
        path = path or self.index_path
        if not path:
            return
        path = Path(path)
        idx_path = path.with_suffix(".faiss")
        meta_path = path.with_suffix(".json")
        if not idx_path.exists():
            return
        self._index = faiss.read_index(str(idx_path))
        self._texts = []
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._texts = json.load(f)

    @property
    def is_empty(self) -> bool:
        return self._index is None or len(self._texts) == 0
