import pandas as pd
from src.complete_recommender import CompleteRecommender

# -------------------------------------------------------
# Create a tiny mock dataset
# -------------------------------------------------------
data = {
    "id": [1, 2, 3],
    "title": [
        "The Dark Knight",
        "Interstellar",
        "The Notebook"
    ],
    "overview": [
        "Batman fights chaos in Gotham.",
        "A journey through space and time.",
        "A romantic drama about enduring love."
    ],
    "tags": [
        ["batman", "dark", "action"],
        ["space", "sci-fi", "future"],
        ["romance", "drama", "love"]
    ],
    "popularity": [900, 850, 600],
    "year": [2008, 2014, 2004]
}

df = pd.DataFrame(data)

# -------------------------------------------------------
# Initialize recommender
# -------------------------------------------------------
rec = CompleteRecommender(df)

# -------------------------------------------------------
# Try some queries
# -------------------------------------------------------
queries = [
    "batman joker dark",
    "space black hole time",
    "romantic love story"
]

for q in queries:
    print(f"\nQUERY: {q}")
    result = rec.recommend(q, top_k=3)
    print(result[["title", "score", "embedding_score", "tag_score"]])
