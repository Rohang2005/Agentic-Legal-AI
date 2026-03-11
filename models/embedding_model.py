"""
Embedding model for legal document chunks.

Uses BAAI/bge-large-en-v1.5 for dense embeddings compatible with FAISS retrieval.
"""

from typing import List
from sentence_transformers import SentenceTransformer

from config.hf_config import HF_TOKEN


class EmbeddingModel:
    """
    Wraps SentenceTransformer (BGE) for encoding legal text into vectors.
    """

    DEFAULT_MODEL_ID = "BAAI/bge-large-en-v1.5"

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = None):
        """
        Initialize the embedding model.

        Args:
            model_id: HuggingFace model id for sentence-transformers.
            device: Device to run on ('cuda', 'cpu', or None for auto).
        """
        self.model_id = model_id
        self._model: SentenceTransformer = None
        self._device = device

    def load(self) -> SentenceTransformer:
        """Load the SentenceTransformer model. Returns cached model if already loaded."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(
                    self.model_id,
                    device=self._device,
                    token=HF_TOKEN,
                )
            except TypeError:
                # Older sentence-transformers may use use_auth_token
                self._model = SentenceTransformer(
                    self.model_id,
                    device=self._device,
                    use_auth_token=HF_TOKEN,
                )
        return self._model

    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = False):
        """
        Encode a list of texts into embedding vectors.

        Args:
            texts: List of strings to embed.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.

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
