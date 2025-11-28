---
title: Movie Recommender
emoji: 🎬
colorFrom: purple
colorTo: blue
sdk: streamlit
sdk_version: 1.30.0
python_version: "3.11"
app_file: streamlit_app.py
pinned: false
---

# Capstone Movie Recommender

A **natural language movie recommendation system** that understands complex queries and uses a 6-signal hybrid architecture to deliver relevant, personalized results.

## Features

- **Natural Language Understanding** - Ask for movies in plain English
- **6-Signal Hybrid Architecture** - Combines collaborative filtering, content similarity, theme matching, sentiment analysis, zero-shot tags, and query relevance
- **Dual-Track Recommendations** - Handles ambiguous queries by showing multiple interpretations
- **User Preference Modes** - Choose between Accuracy, Ratings, or Balanced results

## Example Queries

```
"Julia Roberts Romances from the 90s"
"dark psychological thrillers with a strong female lead"
"My girlfriend broke up with me - something to cheer me up"
"Epic war movies set in ancient times"
```

## Technical Details

- **Dataset:** 43,858 movies with full TMDB metadata
- **Models:** SpaCy NER, Sentence-BERT, LDA, Neural Collaborative Filtering
- **Signals:** CF, Content, Theme, Sentiment, Zero-shot Tags, Query Relevance

## Local Setup

```bash
# Clone the repository
git clone https://github.com/Anabasis2025/Capstone_Movie_Recommender_Final.git
cd Capstone_Movie_Recommender_Final

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Run the demo
python demo_v5.py

# Or run the Streamlit app
streamlit run streamlit_app.py
```
