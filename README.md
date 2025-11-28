# Capstone Movie Recommender

A **natural language movie recommendation system** that understands complex queries and uses a 6-signal hybrid architecture to deliver relevant, personalized results.

## Features

- **Natural Language Understanding** - Ask for movies in plain English
- **6-Signal Hybrid Architecture** - Combines collaborative filtering, content similarity, theme matching, sentiment analysis, zero-shot tags, and query relevance
- **Dual-Track Recommendations** - Handles ambiguous queries by showing multiple interpretations
- **User Preference Modes** - Choose between Accuracy, Ratings, or Balanced results
- **Robust Entity Detection** - SpaCy NER + TMDB fallback for actor/director recognition
- **Semantic Tag Expansion** - BERT-based expansion catches related concepts even with incomplete tags

## Example Queries

```
"Julia Roberts Romances from the 90s"
"my girlfriend broke up with me. what would you recommend to make me feel better?"
"dark psychological thrillers with a strong female lead"
"I've got a championship football game friday night and I need something to motivate me"
"Coming-of-age movies from the 2010s with LGBTQ+ themes"
```

## Architecture

```
USER QUERY
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 1. QUERY PARSING                                    │
│    - Genres, decades, moods, themes extracted       │
│    - Years and year ranges identified               │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 2. ENTITY EXTRACTION (NER + TMDB Fallback)          │
│    - SpaCy NER: Detects actors/directors/studios    │
│    - TMDB Fallback: 133,359 actors database         │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 3. SEMANTIC EXPANSION (BERT)                        │
│    - Expands themes using sentence embeddings       │
│    - "heartbreak" → breakup, moving on, hope, etc.  │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 4. CANDIDATE SELECTION                              │
│    - Filter by decade/year, actor/director, genre   │
│    - Result: 100-5000 candidate movies              │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 5. SIGNAL FUSION (6 Signals)                        │
│    - CF: Collaborative filtering (user ratings)     │
│    - Content: Embedding similarity                  │
│    - Theme: LDA topic matching                      │
│    - Sentiment: BERT + actor sentiment matching     │
│    - Tag: Zero-shot classification (322 tags)       │
│    - Query: TMDB keyword matching                   │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 6. DYNAMIC SCORING                                  │
│    - Adjusts weights based on query type            │
│    - Applies user's preference mode                 │
└─────────────────────────────────────────────────────┘
    │
    ▼
         TOP 10-20 MOVIES
```

## Installation

### Prerequisites
- Python 3.10+
- ~16GB RAM recommended (for loading models)

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/capstone-movie-recommender.git
cd capstone-movie-recommender

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### Download Data and Models

The data and model files (~10GB total) are hosted separately due to size:

**[Download from Google Drive](YOUR_GOOGLE_DRIVE_LINK_HERE)**

After downloading:
1. Extract `data.zip` to `data/` folder
2. Extract `models.zip` to `models/` folder

Your folder structure should look like:
```
capstone-movie-recommender/
├── src/
├── data/
│   └── raw/
│       ├── tmdb_fully_enriched.parquet
│       ├── df_all_ratings.csv
│       └── ...
├── models/
│   ├── svd_model.pkl
│   ├── ncf_model.keras
│   ├── faiss_index.bin
│   └── ...
└── requirements.txt
```

## Usage

### Interactive Demo

```bash
python demo_v5.py
```

This launches an interactive session where you can:
1. Type any movie query in natural language
2. Select your preference mode (Accuracy/Ratings/Balanced)
3. View top 10 recommendations
4. Get details about any movie (Director, Cast, Studio, Year, Overview)

### Programmatic Usage

```python
from src.recommender_interactive_v4 import MovieRecommenderInteractiveV4

# Initialize (takes 10-15 seconds to load models)
recommender = MovieRecommenderInteractiveV4()

# Get recommendations
result = recommender.recommend(
    query="dark psychological thrillers from the 90s",
    preference_mode="accuracy",  # or "ratings" or "balanced"
    top_n=10
)

# Print results
result.print_summary()

# Access individual movies
for movie in result.recommendations:
    print(f"{movie.movie_title} ({movie.year}) - Score: {movie.final_score:.3f}")
```

## Preference Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Accuracy** | Pure content matching, ignores ratings | "I want movies that match my query EXACTLY" |
| **Ratings** | Prioritizes highly-rated/popular movies | "I want the BEST movies in this category" |
| **Balanced** | Equal mix of relevance and quality | Most queries (default) |

## Technical Details

### Data Sources
- **TMDB Dataset:** 43,858 movies with full metadata
- **Cast Data:** 133,359 unique actors
- **Zero-shot Tags:** 17,023 movies, 322 semantic tags
- **Embeddings:** 91,187 movies with 384-dim sentence embeddings
- **Themes:** 17,668 movies with 15 LDA topics
- **Ratings:** 330,712 users for collaborative filtering

### Models
- **NER:** SpaCy en_core_web_sm
- **Embeddings:** Sentence-BERT (all-MiniLM-L6-v2)
- **Themes:** LDA (15 topics)
- **Sentiment:** BERT base
- **Collaborative Filtering:** NeuMF (Neural Matrix Factorization)

### Key Modules

| Module | Description |
|--------|-------------|
| `recommender_interactive_v4.py` | Main orchestrator (3,934 lines) |
| `query_parser.py` | Natural language parsing |
| `entity_extractor.py` | NER + TMDB actor fallback |
| `signal_fusion.py` | 6-signal computation |
| `scoring.py` | Dynamic weight adjustment |
| `semantic_tag_expander.py` | BERT-based semantic expansion |
| `data_loader.py` | Centralized data loading |

## Project Structure

```
capstone-movie-recommender/
├── src/
│   ├── recommender_interactive_v4.py  # Main recommender
│   ├── query_parser.py                # NL query parsing
│   ├── entity_extractor.py            # Actor/director detection
│   ├── signal_fusion.py               # 6-signal scoring
│   ├── scoring.py                     # Score combination
│   ├── semantic_tag_expander.py       # BERT expansion
│   ├── data_loader.py                 # Data loading
│   ├── zero_shot_integration.py       # Tag integration
│   ├── curated_movies.py              # Curated lists
│   └── config.yaml                    # Configuration
├── data/                              # Data files (not in repo)
├── models/                            # Model files (not in repo)
├── demo_v5.py                         # Interactive demo
├── requirements.txt                   # Dependencies
└── README.md                          # This file
```

## Requirements

Key dependencies (see `requirements.txt` for full list):

```
pandas>=2.0.0
numpy>=1.26.0,<2.0.0  # Must be <2.0 for scikit-surprise
torch>=2.0.0
tensorflow>=2.15.0
sentence-transformers>=2.2.0
spacy>=3.7.0
scikit-surprise>=1.1.4
faiss-cpu>=1.7.0
```

## Performance

- **Initialization:** 10-15 seconds (one-time model loading)
- **Per query:** 2-5 seconds (depends on candidate pool size)
- **Memory:** ~8-12GB during operation

## Acknowledgments

This project was developed as a capstone project demonstrating hybrid recommendation systems combining:
- Collaborative Filtering (SVD + Neural Collaborative Filtering)
- Content-Based Filtering (BERT embeddings)
- Natural Language Processing (SpaCy, Transformers)
- Zero-shot Classification

## License

MIT License - See LICENSE file for details.
