import pandas as pd

from src.complete_recommender import CompleteRecommender
from normalize_tags import build_clean_movies_df


def main():
    # -------------------------------------------------------
    # Build the minimal movies DataFrame expected by the recommender
    # using our normalization helper
    # -------------------------------------------------------
    movies = build_clean_movies_df()

    # For speed on first run, optionally limit to 5000 rows
    movies = movies.head(5000)

    print("\n=== Movies DF ready for recommender (after normalization) ===")
    print(movies.head())
    print(movies.dtypes)

    # -------------------------------------------------------
    # Initialize recommender
    # -------------------------------------------------------
    print("\nBuilding recommender (this may take a bit on first run)...")
    rec = CompleteRecommender(movies)

    # -------------------------------------------------------
    # Try some realistic queries
    # -------------------------------------------------------
    queries = [
        "batman joker dark gritty superhero",
        "space exploration black hole time travel",
        "romantic tearjerker love story",
        "gritty inspiring war movie",
    ]

    for q in queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {q}")
        results = rec.recommend(q, top_k=10)

        # Show top 5 nicely
        print(results[["title", "year", "score", "embedding_score", "tag_score"]].head())


if __name__ == "__main__":
    main()


