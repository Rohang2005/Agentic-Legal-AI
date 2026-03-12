"""
Embedding model for legal document chunks.

Uses BAAI/bge-large-en-v1.5 for dense embeddings compatible with FAISS retrieval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

from config.hf_config import HF_TOKEN


class EmbeddingModel:
    """
    Wraps SentenceTransformer (BGE) for encoding legal text into vectors.
    """

    DEFAULT_MODEL_ID = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: Optional[str] = None,
        require_gpu: bool = False,
    ):
        self.model_id = model_id
        self._model: Optional[SentenceTransformer] = None
        self.require_gpu = require_gpu
        if device is not None:
            self._device = device
        else:
            try:
                import torch
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self._device = "cpu"

    def load(self) -> SentenceTransformer:
        """Load the SentenceTransformer model. Returns cached model if already loaded."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            if self.require_gpu and self._device != "cuda":
                raise RuntimeError(
                    "GPU execution required for embeddings, but CUDA is unavailable in this environment."
                )

            try:
                self._model = SentenceTransformer(
                    self.model_id,
                    device=self._device,
                    token=HF_TOKEN,
                )
            except TypeError:
                self._model = SentenceTransformer(
                    self.model_id,
                    device=self._device,
                    use_auth_token=HF_TOKEN,
                )
        return self._model

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False):
        """
        Encode a list of texts into embedding vectors.

        Returns:
            Numpy array of shape (n_texts, embedding_dim).
        """
        model = self.load()
        return model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
        )

    def encode_single(self, text: str):
        """Encode a single string. Returns 1D array."""
        return self.encode([text])[0]

    @property
    def dimension(self) -> int:
        """Embedding dimension of the model."""
        return self.load().get_sentence_embedding_dimension()
