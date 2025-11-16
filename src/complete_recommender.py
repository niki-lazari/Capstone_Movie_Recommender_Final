"""
complete_recommender.py

Initial modular version of the Capstone Movie Recommender.
This connects embeddings, FAISS, tagging, and scoring into
a single unified class that can be expanded later.
"""

import pandas as pd
import numpy as np

from typing import List, Dict

from .embeddings import EmbeddingService
from .faiss_index import FaissIndex
from .tagging import normalize_tag_list, tag_overlap_score, compute_rarity_boost
from .scoring import compute_final_score


class CompleteRecommender:
    """
    Minimal working recommender that:
    - Loads movie metadata
    - Encodes embeddings
    - Builds a FAISS index
    - Computes tag rarity weights
    - Scores movie similarity using embedding + tag overlap
    """

    def __init__(self, movies_df: pd.DataFrame, tag_column: str = "tags"):
        self.movies = movies_df.copy()
        self.tag_column = tag_column

        # --- Normalize tags ---
        self.movies[self.tag_column] = self.movies[self.tag_column].apply(
            lambda tags: normalize_tag_list(tags) if isinstance(tags, list) else []
        )

        # --- Compute rarity boosts ---
        all_movie_tags = self.movies[self.tag_column].tolist()
        self.tag_rarity = compute_rarity_boost(all_movie_tags)

        # --- Embeddings ---
        self.embedder = EmbeddingService()
        self.movie_texts = (
            self.movies["title"].fillna("") + " " +
            self.movies["overview"].fillna("")
        ).tolist()

        # --- Encode movies ---
        self.movie_embeddings = self.embedder.encode_texts(self.movie_texts)

        # --- Build FAISS index ---
        dim = self.movie_embeddings.shape[1]
        self.index = FaissIndex(embedding_dim=dim)
        self.index.add(self.movie_embeddings)

    def recommend(self, query: str, top_k: int = 20) -> pd.DataFrame:
        """
        End-to-end recommendation:
        - Encode the query
        - Query FAISS
        - Compute tag overlap
        - Combine scores
        - Return ranked results
        """
        query_vec = self.embedder.encode_single(query)
        distances, indices = self.index.search(query_vec, top_k=top_k)

        # Convert distances → similarity (FAISS L2 distances are reversed)
        sim_scores = 1 / (1 + distances)

        results = []
        for score, idx in zip(sim_scores, indices):
            movie = self.movies.iloc[idx]
            movie_tags = set(movie[self.tag_column])
            query_tags = set(normalize_tag_list(query.split()))

            tag_score = tag_overlap_score(query_tags, movie_tags, self.tag_rarity)

            final = compute_final_score(
                embedding_score=float(score),
                tag_score=float(tag_score),
                popularity=movie.get("popularity", None),
                year=movie.get("year", None),
            )

            results.append({
                "movie_id": movie.get("id", None),
                "title": movie.get("title", None),
                "score": final,
                "embedding_score": float(score),
                "tag_score": float(tag_score),
                "year": movie.get("year", None),
                "popularity": movie.get("popularity", None),
            })

        # Sort descending by final score
        results_df = pd.DataFrame(results).sort_values("score", ascending=False)
        return results_df.reset_index(drop=True)
