"""
tagging.py

Tag preprocessing, normalization, and similarity scoring utilities for the
Capstone Movie Recommender.
"""

import re
from typing import List, Dict, Set
from collections import Counter


def normalize_tag(tag: str) -> str:
    """
    Normalize tags to lowercase alphanumeric + spaces.
    """
    if not isinstance(tag, str):
        return ""
    tag = tag.lower().strip()
    tag = re.sub(r"[^a-z0-9\s]", "", tag)
    tag = re.sub(r"\s+", " ", tag)
    return tag


def normalize_tag_list(tags: List[str]) -> List[str]:
    """
    Normalize and deduplicate a list of tags.
    """
    cleaned = [normalize_tag(t) for t in tags if isinstance(t, str)]
    cleaned = [t for t in cleaned if t]  # remove empty
    return list(set(cleaned))


def compute_rarity_boost(all_movies_tags: List[List[str]]) -> Dict[str, float]:
    """
    Compute global rarity weights for tags across the dataset.
    Rare tags get boosted, common tags get penalized.
    """
    flattened = [tag for tags in all_movies_tags for tag in tags]
    counts = Counter(flattened)

    # Convert counts → rarity scores
    rarity = {}
    for tag, count in counts.items():
        # Inverted frequency: rare tags get higher weights
        rarity[tag] = 1.0 / (1.0 + count)

    return rarity


def tag_overlap_score(user_tags: Set[str], movie_tags: Set[str], rarity: Dict[str, float]) -> float:
    """
    Weighted overlap between user query tags and a movie's tag set.
    """
    shared = user_tags.intersection(movie_tags)
    if not shared:
        return 0.0

    # Sum rarity-adjusted overlap
    return sum(rarity.get(tag, 0.0) for tag in shared)
