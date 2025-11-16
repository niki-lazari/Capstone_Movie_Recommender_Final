import pandas as pd
from pathlib import Path

# Load the full dataset
df = pd.read_parquet("data/tmdb_with_review_tags.parquet")

# Create a minimal clean DataFrame for the recommender
movies = pd.DataFrame({
    "id": df["tmdb_id"],
    "title": df["tmdb_title"],
    "overview": df["overview"],
    "tags": df["review_tags"],        # already lists of strings
    "year": df["year"],
    "popularity": df["popularity"],
})

print("\n=== Clean Movies DF Shape ===")
print(movies.shape)

print("\n=== Columns ===")
print(movies.columns)

print("\n=== First 5 Rows ===")
print(movies.head())
