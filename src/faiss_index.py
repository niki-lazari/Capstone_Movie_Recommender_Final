"""
faiss_index.py

Utilities for building, saving, loading, and querying a FAISS index for
the Capstone Movie Recommender.
"""

import faiss
import numpy as np
from typing import Tuple


class FaissIndex:
    """
    Wrapper around a FAISS index for cosine or L2 similarity search.
    """

    def __init__(self, embedding_dim: int, metric: str = "L2"):
        """
        Args:
            embedding_dim (int): Size of the embedding vectors.
            metric (str): "L2" or "cosine" similarity.
        """
        self.embedding_dim = embedding_dim

        if metric.upper() == "COSINE":
            # Normalize embeddings before addition
            self.index = faiss.IndexFlatIP(embedding_dim)
            self.use_cosine = True
        else:
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.use_cosine = False

    def add(self, embeddings: np.ndarray):
        """
        Add an embedding matrix of shape (N, D) to the index.
        """
        if self.use_cosine:
            # Ensure embeddings are normalized
            faiss.normalize_L2(embeddings)

        self.index.add(embeddings.astype(np.float32))

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            query_vec (np.ndarray): (1, D) or (D,) query embedding
            top_k (int): number of results to return

        Returns:
            (distances, indices)
        """
        # Ensure shape is (1, D) for FAISS
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        if self.use_cosine:
            faiss.normalize_L2(query_vec)

        distances, indices = self.index.search(query_vec.astype(np.float32), top_k)
        return distances[0], indices[0]


    def save(self, path: str):
        """
        Save the FAISS index to disk.
        """
        faiss.write_index(self.index, path)

    @classmethod
    def load(cls, path: str):
        """
        Load a FAISS index from disk and infer its dimensions.
        """
        index = faiss.read_index(path)
        dim = index.d
        obj = cls(dim)
        obj.index = index
        return obj
