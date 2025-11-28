#!/usr/bin/env python3
"""
=============================================================================
CAPSTONE MOVIE RECOMMENDATION SYSTEM - DEMO V5
=============================================================================

A sophisticated hybrid movie recommendation system featuring:
  - 6-Signal Fusion (CF + Content + Theme + Sentiment + Tag + Query)
  - Natural Language Understanding with BERT-based clause classification
  - Situation+Outcome Query Detection (e.g., "My girlfriend broke up with me")
  - Multi-Theme Query Support (e.g., "nostalgic childhood adventure")
  - Actor Detection (NER + TMDB Fallback with 133,359 actors)
  - Dynamic Inflection Generation for overview matching
  - User Preference Modes (Accuracy / Ratings / Balanced)

Dataset:
  - 43,858 movies with full TMDB metadata
  - 330,712 users for collaborative filtering
  - 133,359 actors for fallback matching
  - 322 zero-shot tags for theme detection
  - 22,818 TMDB keywords for semantic expansion

Author: Kyle
Date: November 2025

SETUP:
  1. Activate the virtual environment:
     Windows: .venv\\Scripts\\activate
     Mac/Linux: source .venv/bin/activate

  2. Run the demo:
     python demo_v5.py

=============================================================================
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

import sys
sys.path.insert(0, 'src')

import logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise during demo

import pandas as pd
from recommender_interactive_v4 import MovieRecommenderInteractiveV4


# =============================================================================
# DISPLAY UTILITIES
# =============================================================================

def print_banner():
    """Print the demo banner."""
    print("\n" + "=" * 90)
    print("""
    ███╗   ███╗ ██████╗ ██╗   ██╗██╗███████╗    ██████╗ ███████╗ ██████╗
    ████╗ ████║██╔═══██╗██║   ██║██║██╔════╝    ██╔══██╗██╔════╝██╔════╝
    ██╔████╔██║██║   ██║██║   ██║██║█████╗      ██████╔╝█████╗  ██║
    ██║╚██╔╝██║██║   ██║╚██╗ ██╔╝██║██╔══╝      ██╔══██╗██╔══╝  ██║
    ██║ ╚═╝ ██║╚██████╔╝ ╚████╔╝ ██║███████╗    ██║  ██║███████╗╚██████╗
    ╚═╝     ╚═╝ ╚═════╝   ╚═══╝  ╚═╝╚══════╝    ╚═╝  ╚═╝╚══════╝ ╚═════╝
    """)
    print("=" * 90)
    print("                    CAPSTONE RECOMMENDATION SYSTEM V5")
    print("=" * 90)
    print("\n  FEATURES:")
    print("    [1] 6-Signal Fusion (CF + Content + Theme + Sentiment + Tag + Query)")
    print("    [2] Situation+Outcome Detection (BERT-based clause classification)")
    print("    [3] Multi-Theme Query Support (nostalgic + adventure = combined)")
    print("    [4] Actor Detection (NER + TMDB Fallback: 133,359 actors)")
    print("    [5] Dynamic Inflection Generation (broke up -> breakup, dumps, etc.)")
    print("    [6] User Preference Modes (Accuracy / Ratings / Balanced)")
    print("\n" + "=" * 90 + "\n")


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 90)
    print(f"  {title}")
    print("=" * 90)


def print_query_info(query, result):
    """Print parsed query information."""
    print(f"\n  Query: \"{query}\"")
    print(f"  Candidates: {result.num_candidates}")

    pq = result.parsed_query
    if hasattr(pq, 'genres') and pq.genres:
        print(f"  Genres: {pq.genres}")
    if hasattr(pq, 'decades') and pq.decades:
        print(f"  Decades: {pq.decades}")
    if hasattr(pq, 'actors') and pq.actors:
        print(f"  Actors: {pq.actors}")
    if hasattr(pq, 'themes') and pq.themes:
        themes_preview = pq.themes[:8] if len(pq.themes) > 8 else pq.themes
        print(f"  Themes: {themes_preview}" + ("..." if len(pq.themes) > 8 else ""))


def print_results(result, top_n=10, show_scores=True):
    """Print recommendation results."""
    print(f"\n  TOP {min(top_n, len(result.recommendations))} RECOMMENDATIONS:")
    print("  " + "-" * 86)

    for i, movie in enumerate(result.recommendations[:top_n], 1):
        if show_scores:
            print(f"  {i:2d}. {movie.movie_title}")
            print(f"      Final: {movie.final_score:.4f} | Query: {movie.query_score:.4f} | "
                  f"Tag: {movie.tag_score:.4f} | CF: {movie.cf_score:.2f}")
        else:
            print(f"  {i:2d}. {movie.movie_title} (Score: {movie.final_score:.4f})")
    print()


def print_dual_track_results(result, top_n=10):
    """Print results - handles both dual-track (entity+mood) and multi-theme modes."""
    all_movies = []  # Collect all movies for selection
    pq = result.parsed_query

    # Check if this is a multi-theme query (has 2+ theme groups, no entities)
    has_entities = bool(pq.actors or pq.directors or pq.studios)

    # KEY ENTITY NOUNS that indicate entity_mood mode, not multi_theme
    KEY_ENTITY_NOUNS = {
        'female', 'male', 'woman', 'man', 'girl', 'boy',
        'dog', 'cat', 'animal', 'pet',
        'lawyer', 'doctor', 'cop', 'detective', 'soldier', 'teacher',
        'mother', 'father', 'daughter', 'son', 'wife', 'husband',
        'girlfriend', 'boyfriend', 'friend', 'family',
        'lead', 'protagonist', 'hero', 'heroine'
    }
    content_nouns = getattr(pq, 'content_nouns', [])
    has_entity_noun = any(noun.lower() in KEY_ENTITY_NOUNS for noun in content_nouns)

    is_multi_theme = (not has_entities and
                      not has_entity_noun and
                      hasattr(pq, 'theme_groups') and
                      len(pq.theme_groups) >= 2)

    if is_multi_theme:
        # MULTI-THEME MODE: Single unified list ranked by BOTH theme scores
        # Re-sort by combined theme scores (multiplication ensures both must be high)
        theme_groups = pq.theme_groups[:2]

        # Get movies that have theme scores computed
        movies_with_scores = [m for m in result.recommendations if hasattr(m, 'theme1_score') and hasattr(m, 'theme2_score')]

        if movies_with_scores:
            # Sort by combined theme score (theme1 * theme2) - rewards movies matching BOTH themes
            # Add small epsilon to avoid zero multiplication killing good single-theme matches
            movies_with_scores.sort(
                key=lambda m: (m.theme1_score + 0.1) * (m.theme2_score + 0.1),
                reverse=True
            )
            all_movies = movies_with_scores[:top_n * 2]
        else:
            # Fallback if theme scores not computed
            all_movies = result.recommendations[:top_n * 2]

        print(f"\n  MULTI-THEME MODE: Movies matching BOTH themes")
        print(f"  Theme 1: {theme_groups[0]}")
        print(f"  Theme 2: {theme_groups[1]}")
        print("  " + "-" * 86)

        print(f"\n  TOP {min(top_n * 2, len(all_movies))} RECOMMENDATIONS:")
        for i, movie in enumerate(all_movies, 1):
            t1 = getattr(movie, 'theme1_score', 0)
            t2 = getattr(movie, 'theme2_score', 0)
            combined = (t1 + 0.1) * (t2 + 0.1)
            print(f"    {i:2d}. {movie.movie_title} (Score: {combined:.4f}, T1:{t1:.2f} T2:{t2:.2f})")

    elif result.dual_track_mode and result.entity_track and result.mood_track:
        # ENTITY+MOOD DUAL-TRACK MODE: Show both tracks (10 each)
        print(f"\n  DUAL-TRACK MODE ACTIVE")
        print("  " + "-" * 86)

        print(f"\n  ENTITY TRACK (content-focused):")
        entity_movies = result.entity_track[:top_n]
        for i, movie in enumerate(entity_movies, 1):
            print(f"    {i:2d}. {movie.movie_title} (Score: {movie.final_score:.4f})")
            all_movies.append(movie)

        print(f"\n  MOOD TRACK (theme/sentiment-focused):")
        mood_movies = result.mood_track[:top_n]
        start_num = len(entity_movies) + 1
        for i, movie in enumerate(mood_movies, start_num):
            print(f"    {i:2d}. {movie.movie_title} (Score: {movie.final_score:.4f})")
            all_movies.append(movie)
    else:
        # SINGLE-TRACK MODE: Show combined results
        all_movies = result.recommendations[:top_n * 2]
        print(f"\n  TOP {min(top_n * 2, len(all_movies))} RECOMMENDATIONS:")
        print("  " + "-" * 86)
        for i, movie in enumerate(all_movies, 1):
            print(f"    {i:2d}. {movie.movie_title} (Score: {movie.final_score:.4f})")

    print()
    return all_movies


def get_preference_mode():
    """Get user's preference mode selection."""
    print("\n  Choose recommendation mode:")
    print("    [1] ACCURACY  - Pure content matching (best for specific queries)")
    print("    [2] RATINGS   - Prioritize highly-rated movies")
    print("    [3] BALANCED  - Mix of both (default)")

    while True:
        choice = input("\n  Enter 1, 2, or 3 (or ENTER for balanced): ").strip()
        if choice == "1":
            return "accuracy"
        elif choice == "2":
            return "ratings"
        elif choice == "3" or choice == "":
            return "balanced"
        print("  Invalid choice. Please enter 1, 2, or 3.")


def get_movie_details(recommender, movie_title):
    """Look up movie details from TMDB data."""
    # Find the movie in the dataframe
    movies_df = recommender.movies

    # Try exact match first
    match = movies_df[movies_df['title'].str.lower() == movie_title.lower()]

    # If no exact match, try contains
    if match.empty:
        match = movies_df[movies_df['title'].str.lower().str.contains(movie_title.lower(), na=False)]

    if match.empty:
        return None

    movie = match.iloc[0]

    details = {
        'title': movie.get('title', movie_title),
        'year': movie.get('year', 'N/A'),
        'director': movie.get('director', 'N/A'),
        'cast': movie.get('cast', 'N/A'),
        'production_companies': movie.get('production_companies', 'N/A'),
        'overview': movie.get('overview', 'N/A'),
    }

    return details


def print_movie_details(details):
    """Print formatted movie details."""
    print("\n  " + "=" * 86)
    print(f"  {details['title']} ({details['year']})")
    print("  " + "=" * 86)

    # Director
    director = details['director']
    if isinstance(director, list):
        director = ', '.join(director[:3])
    elif pd.isna(director):
        director = 'N/A'
    print(f"\n  Director: {director}")

    # Cast
    cast = details['cast']
    if isinstance(cast, list):
        cast = ', '.join(cast[:5])  # Show top 5 actors
    elif pd.isna(cast):
        cast = 'N/A'
    print(f"  Cast: {cast}")

    # Studio/Production Companies
    studio = details['production_companies']
    if isinstance(studio, list):
        studio = ', '.join(studio[:3])
    elif pd.isna(studio):
        studio = 'N/A'
    print(f"  Studio: {studio}")

    # Overview
    overview = details['overview']
    if pd.isna(overview):
        overview = 'No overview available.'
    # Wrap long overviews
    if len(overview) > 500:
        overview = overview[:500] + "..."
    print(f"\n  Overview:\n  {overview}")
    print()


def post_results_menu(recommender, all_movies):
    """Handle post-results interaction: movie details or new query."""
    while True:
        print("\n  " + "-" * 86)
        print("  What would you like to do?")
        print("    [#] Enter a number to learn more about that movie")
        print("    [N] Enter a new query")
        print("    [Q] Quit")

        choice = input("\n  Your choice: ").strip().lower()

        if choice == 'q' or choice == 'quit':
            return 'quit'
        elif choice == 'n' or choice == 'new':
            return 'new_query'
        else:
            # Try to parse as a number
            try:
                num = int(choice)
                if 1 <= num <= len(all_movies):
                    movie = all_movies[num - 1]
                    details = get_movie_details(recommender, movie.movie_title)
                    if details:
                        print_movie_details(details)
                    else:
                        print(f"\n  Could not find details for: {movie.movie_title}")
                else:
                    print(f"\n  Please enter a number between 1 and {len(all_movies)}")
            except ValueError:
                print("\n  Invalid input. Enter a movie number, 'N' for new query, or 'Q' to quit.")


# =============================================================================
# DEMO QUERIES - Showcasing All Features
# =============================================================================

DEMO_QUERIES = [
    {
        "name": "SITUATION + OUTCOME QUERY",
        "query": "My girlfriend broke up with me - I need something to cheer me up",
        "description": "BERT classifies clauses as SITUATION vs OUTCOME. Expands 'broke up' to breakup keywords, maps 'cheer up' to comedy genre.",
        "mode": "accuracy",
        "expected": ["Forgetting Sarah Marshall", "High Fidelity", "500 Days of Summer"]
    },
    {
        "name": "NOSTALGIC MULTI-THEME QUERY",
        "query": "I'm feeling nostalgic for my childhood but also want something new - maybe a modern movie with an old-school adventure vibe",
        "description": "Multi-theme detection: childhood + adventure. Filters to adventure genre, expands 'childhood' and 'school' semantically.",
        "mode": "accuracy",
        "expected": ["Inside Out", "Bridge to Terabithia", "Harry Potter", "Toy Story"]
    },
    {
        "name": "ACTOR + DECADE QUERY (NER)",
        "query": "Bill Murray movies from the 90s",
        "description": "NER detects 'Bill Murray' as PERSON entity. Filters by actor + decade (1990-1999).",
        "mode": "balanced",
        "expected": ["Groundhog Day", "Rushmore", "Ghostbusters II"]
    },
    {
        "name": "ACTOR QUERY (TMDB FALLBACK)",
        "query": "Julia Roberts romances from the 90s",
        "description": "NER fails (sees 'Julia Roberts Romances' as ORG). TMDB fallback matches against 133,359 actors.",
        "mode": "balanced",
        "expected": ["Pretty Woman", "Notting Hill", "My Best Friend's Wedding"]
    },
    {
        "name": "COMPLEX MULTI-CONSTRAINT",
        "query": "dark psychological thrillers from the 90s with a female lead",
        "description": "Genre (thriller) + Mood (dark, psychological) + Decade (90s) + Theme (female lead). Uses entity noun boost.",
        "mode": "accuracy",
        "expected": ["Black Swan", "Gone Girl", "Silence of the Lambs"]
    },
    {
        "name": "MOTIVATIONAL SITUATIONAL",
        "query": "I've got a championship football game friday night and I need to watch something to motivate me",
        "description": "Understands context: sports + motivation + championship. Maps to inspiring sports films.",
        "mode": "accuracy",
        "expected": ["Remember the Titans", "Friday Night Lights", "Rudy"]
    },
    {
        "name": "THEME-BASED (ZERO-SHOT TAGS)",
        "query": "heist movies",
        "description": "Zero-shot tag matching + semantic expansion. Finds movies tagged with 'heist', 'robbery', 'theft'.",
        "mode": "balanced",
        "expected": ["Ocean's Eleven", "The Italian Job", "Heat"]
    },
    {
        "name": "HISTORICAL REVENGE",
        "query": "ancient history revenge epic",
        "description": "Multi-concept AND-logic: must match BOTH 'ancient/historical' AND 'revenge'. Concept coverage scoring.",
        "mode": "accuracy",
        "expected": ["Gladiator", "Braveheart", "Troy"]
    }
]


# =============================================================================
# MAIN DEMO FUNCTIONS
# =============================================================================

def run_curated_demos(recommender):
    """Run the curated demo queries."""
    print_section("CURATED DEMOS - Showcasing System Features")

    for i, demo in enumerate(DEMO_QUERIES, 1):
        print(f"\n  DEMO {i}/{len(DEMO_QUERIES)}: {demo['name']}")
        print("  " + "-" * 86)
        print(f"  {demo['description']}")
        print(f"\n  Query: \"{demo['query']}\"")
        print(f"  Mode: {demo['mode'].upper()}")
        print(f"  Expected to find: {', '.join(demo['expected'][:3])}...")

        input("\n  Press ENTER to run this query...")

        print("\n  Processing...")

        try:
            result = recommender.recommend(
                demo['query'],
                top_n=10,
                max_candidates=500,
                preference_mode=demo['mode']
            )

            print_query_info(demo['query'], result)
            print_results(result, top_n=10, show_scores=True)

            # Check for expected movies
            found = []
            titles = [m.movie_title for m in result.recommendations[:20]]
            for expected in demo['expected']:
                for title in titles:
                    if expected.lower() in title.lower():
                        found.append(title)
                        break

            if found:
                print(f"  Expected movies found: {', '.join(found)}")

        except Exception as e:
            print(f"\n  Error: {e}")

        if i < len(DEMO_QUERIES):
            cont = input("\n  Press ENTER for next demo, or 'q' to skip to interactive mode: ").strip().lower()
            if cont == 'q':
                break

    print_section("CURATED DEMOS COMPLETE")


def run_interactive_mode(recommender):
    """Run interactive query mode."""
    print_section("INTERACTIVE MODE")
    print("\n  Enter your own movie queries!")
    print("  Examples:")
    print("    - 'sci-fi movies from the 80s'")
    print("    - 'I just went through a tough breakup and need something uplifting'")
    print("    - 'Tom Hanks comedies'")
    print("    - 'dark crime thrillers with plot twists'")

    while True:
        try:
            print("\n  " + "-" * 86)
            query = input("  Your query (or 'quit' to exit): ").strip()

            if query.lower() in ['quit', 'exit', 'q']:
                print("\n  Thanks for using the Movie Recommender!")
                break

            if not query:
                continue

            mode = get_preference_mode()

            print(f"\n  Processing in {mode.upper()} mode...")

            result = recommender.recommend(
                query,
                top_n=20,  # Get 20 to support 10 per track in dual-track mode
                max_candidates=500,
                preference_mode=mode
            )

            print_query_info(query, result)

            # Show results and collect all movies for selection
            all_movies = print_dual_track_results(result, top_n=10)

            # Post-results menu
            action = post_results_menu(recommender, all_movies)

            if action == 'quit':
                print("\n  Thanks for using the Movie Recommender!")
                break
            # 'new_query' continues the loop

        except KeyboardInterrupt:
            print("\n\n  Interrupted. Thanks for using the Movie Recommender!")
            break
        except Exception as e:
            print(f"\n  Error: {e}\n")


def main():
    """Main demo entry point."""
    print_banner()

    print("  Initializing recommendation system...")
    print("  (This may take 15-30 seconds on first run)\n")

    try:
        # Suppress INFO logs during initialization
        logging.getLogger('recommender_interactive_v4').setLevel(logging.WARNING)
        logging.getLogger('src.signal_fusion').setLevel(logging.WARNING)
        logging.getLogger('src.data_loader').setLevel(logging.WARNING)
        logging.getLogger('src.zero_shot_integration').setLevel(logging.WARNING)
        logging.getLogger('src.query_parser').setLevel(logging.WARNING)
        logging.getLogger('sentence_transformers').setLevel(logging.WARNING)

        recommender = MovieRecommenderInteractiveV4()

    except Exception as e:
        print(f"\n  ERROR: Failed to initialize recommender: {e}")
        print("\n  Make sure you:")
        print("    1. Activated the virtual environment: .venv\\Scripts\\activate")
        print("    2. Have all data files in data/raw/")
        print("    3. Have all model files in models/")
        return

    print("  System ready!\n")

    # Ask user what they want to do
    print("  What would you like to do?")
    print("    [1] Run curated demos (showcases all features)")
    print("    [2] Jump to interactive mode")
    print("    [3] Run both (demos first, then interactive)")

    while True:
        choice = input("\n  Enter 1, 2, or 3: ").strip()
        if choice in ['1', '2', '3']:
            break
        print("  Invalid choice. Please enter 1, 2, or 3.")

    if choice == '1':
        run_curated_demos(recommender)
    elif choice == '2':
        run_interactive_mode(recommender)
    else:
        run_curated_demos(recommender)
        run_interactive_mode(recommender)

    print_section("DEMO COMPLETE")
    print("""
  System Statistics:
    - Movies: 43,858 with full TMDB metadata
    - Users: 330,712 for collaborative filtering
    - Actors: 133,359 for fallback matching
    - Tags: 322 zero-shot tags
    - Keywords: 22,818 TMDB keywords

  Key Features Demonstrated:
    - Situation+Outcome query detection (BERT clause classification)
    - Multi-theme query support with AND-logic concept scoring
    - Dynamic inflection generation for overview matching
    - 6-signal fusion scoring with preference modes
    - NER + TMDB fallback for actor detection

  Thank you for exploring the Capstone Movie Recommendation System!
""")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
