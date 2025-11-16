import pandas as pd
import numpy as np
from pathlib import Path

def normalize_tag_array(val):
    """
    Convert numpy ndarray of tags into a clean Python list of strings.
    Handles empty arrays, NaNs, etc.
    """
    if isinstance(val, np.ndarray):
        # Convert ndarray -> list of strings
        return [str(x).strip().lower() for x in val.tolist()]
    elif isinstance(val, list):
        return [str(x).strip().lower() for x in val]
    else:
        return []

def build_clean_movies_df():
    df = pd.read_parquet("data/tmdb_with_review_tags.parquet")

    # Convert review_tags from ndarray -> list[str]
    df["clean_tags"] = df["review_tags"].apply(normalize_tag_array)

    movies = pd.DataFrame({
        "id": df["tmdb_id"],
        "title": df["tmdb_title"],
        "overview": df["overview"],
        "tags": df["clean_tags"],          # <--- use clean tags!
        "year": pd.to_numeric(df["year"], errors="coerce"),
        "popularity": pd.to_numeric(df["popularity"], errors="coerce").fillna(0),
    })

    print("\n=== Clean movies df (first 5 rows) ===")
    print(movies.head())

    return movies

if __name__ == "__main__":
    build_clean_movies_df()
