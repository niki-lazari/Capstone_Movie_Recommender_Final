"""
embeddings.py

Centralized utilities for building and using sentence-transformer embeddings
for the Capstone Movie Recommender.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


class EmbeddingService:
    """
    Wraps a SentenceTransformer model with consistent methods for
    encoding movie metadata and query text.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert a list of text strings into normalized embedding vectors.

        Args:
            texts (List[str]): Input text list
            batch_size (int): Batch size for the transformer model

        Returns:
            np.ndarray: (N, D) embedding matrix, normalized along axis=1
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text string into a normalized embedding vector.

        Args:
            text (str): Input text

        Returns:
            np.ndarray: (1, D) normalized embedding vector
        """
        return self.encode_texts([text])[0]
