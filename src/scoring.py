"""
scoring.py

Unified scoring functions for combining embedding similarity,
tag overlap, and metadata adjustments in the Capstone Recommender.
"""

from typing import Optional


def normalize(value: float, min_val: float, max_val: float) -> float:
    """
    Normalize a value to [0, 1] given a min/max range.
    Safely handles edge cases.
    """
    if max_val == min_val:
        return 0.0
    return (value - min_val) / (max_val - min_val)


def compute_final_score(
    embedding_score: float,
    tag_score: float,
    popularity: Optional[float] = None,
    year: Optional[int] = None,
    popularity_weight: float = 0.10,
    tag_weight: float = 0.25,
    embedding_weight: float = 0.65,
) -> float:
    """
    Combine multiple scoring components into a single relevance score.

    Args:
        embedding_score (float): cosine similarity or L2-based similarity
        tag_score (float): rarity-adjusted tag overlap score
        popularity (Optional[float]): movie popularity (0–1000+)
        year (Optional[int]): release year
        *weights: weighting factors

    Returns:
        float: final weighted score
    """

    # Normalize popularity
    pop_norm = 0.0
    if popularity is not None:
        pop_norm = normalize(popularity, 0.0, 1000.0)

    # Normalize release year (favor recent decades)
    yr_norm = 0.0
    if year is not None:
        yr_norm = normalize(year, 1950, 2025)

    # Weighted sum
    final = (
        embedding_score * embedding_weight +
        tag_score * tag_weight +
        pop_norm * popularity_weight +
        yr_norm * 0.05   # small boost for recency
    )

    return float(final)
