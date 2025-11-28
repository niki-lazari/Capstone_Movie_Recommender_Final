"""
Streamlit Movie Recommender App
A natural language movie recommendation system with 6-signal hybrid architecture.
"""

import streamlit as st
import time
import pandas as pd

# Page config
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'all_movies_list' not in st.session_state:
    st.session_state.all_movies_list = []
if 'last_query' not in st.session_state:
    st.session_state.last_query = ""

# Custom CSS
st.markdown("""
<style>
    .movie-card {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .track-header {
        background-color: #2d2d2d;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🎬 Movie Recommender")
st.markdown("*Ask for movies in natural language and get personalized recommendations*")

# Initialize recommender (cached to avoid reloading)
@st.cache_resource(show_spinner=True)
def load_recommender():
    """Load the movie recommender system."""
    with st.spinner("Loading recommendation engine... (this may take 30-60 seconds on first load)"):
        from src.recommender_interactive_v4 import MovieRecommenderInteractiveV4
        return MovieRecommenderInteractiveV4()


def get_movie_details(recommender, movie_title):
    """Look up movie details from TMDB data."""
    movies_df = recommender.movies

    # Try exact match first
    match = movies_df[movies_df['title'].str.lower() == movie_title.lower()]

    # If no exact match, try contains
    if match.empty:
        match = movies_df[movies_df['title'].str.lower().str.contains(movie_title.lower(), na=False)]

    if match.empty:
        return None

    movie = match.iloc[0]

    # Helper to safely get values
    def safe_get(key, default='N/A'):
        val = movie.get(key, default)
        if val is None:
            return default
        # Handle lists/arrays - return as-is if not empty
        if isinstance(val, (list, tuple)):
            return val if len(val) > 0 else default
        # Handle scalar NaN (floats)
        if isinstance(val, float):
            try:
                if pd.isna(val):
                    return default
            except (ValueError, TypeError):
                pass
        return val

    return {
        'title': safe_get('title', movie_title),
        'year': safe_get('year', 'N/A'),
        'directors': safe_get('directors', []),
        'cast': safe_get('cast', []),
        'production_companies': safe_get('production_companies', []),
        'overview': safe_get('overview', 'No overview available.'),
        'genres': safe_get('genres', []),
        'runtime': safe_get('runtime', 'N/A'),
        'vote_average': safe_get('vote_average', 'N/A'),
    }


def display_movie_details(details):
    """Display formatted movie details."""
    st.markdown("---")
    st.subheader(f"📽️ {details['title']} ({details['year']})")

    col1, col2 = st.columns(2)

    with col1:
        # Directors
        directors = details['directors']
        if isinstance(directors, list) and directors:
            st.markdown(f"**Director(s):** {', '.join(str(d) for d in directors[:3])}")
        elif directors and directors != 'N/A':
            st.markdown(f"**Director(s):** {directors}")

        # Cast
        cast = details['cast']
        if isinstance(cast, list) and cast:
            st.markdown(f"**Cast:** {', '.join(str(c) for c in cast[:6])}")
        elif cast and cast != 'N/A':
            st.markdown(f"**Cast:** {cast}")

        # Studio
        studios = details['production_companies']
        if isinstance(studios, list) and studios:
            st.markdown(f"**Studio:** {', '.join(str(s) for s in studios[:3])}")

    with col2:
        # Genres
        genres = details['genres']
        if isinstance(genres, list) and genres:
            st.markdown(f"**Genres:** {', '.join(str(g) for g in genres)}")

        # Runtime
        if details['runtime'] and details['runtime'] != 'N/A':
            st.markdown(f"**Runtime:** {details['runtime']} min")

        # Rating
        if details['vote_average'] and details['vote_average'] != 'N/A':
            st.markdown(f"**TMDB Rating:** {details['vote_average']}/10")

    # Overview
    st.markdown("**Overview:**")
    overview = details['overview']
    if len(str(overview)) > 600:
        overview = str(overview)[:600] + "..."
    st.markdown(f"*{overview}*")


def display_movie_row(movie, index):
    """Display a single movie row."""
    col_rank, col_info, col_score = st.columns([0.5, 5, 1])

    with col_rank:
        st.markdown(f"### {index}")

    with col_info:
        year = getattr(movie, 'year', 'N/A')
        st.markdown(f"**{movie.movie_title}** ({year})")

        # Brief overview if available
        overview = getattr(movie, 'overview', '')
        if overview and len(str(overview)) > 150:
            overview = str(overview)[:150] + "..."
        if overview:
            st.caption(overview)

    with col_score:
        score = getattr(movie, 'final_score', 0)
        st.metric("Score", f"{score:.2f}")

    return (index, movie.movie_title, movie)


# Sidebar
with st.sidebar:
    st.header("Settings")

    preference_mode = st.selectbox(
        "Preference Mode",
        options=["balanced", "accuracy", "ratings"],
        index=0,
        help="""
        - **Balanced**: Mix of relevance and quality (recommended)
        - **Accuracy**: Pure content matching, ignores ratings
        - **Ratings**: Prioritizes highly-rated/popular movies
        """
    )

    top_n = st.slider(
        "Results per Track",
        min_value=5,
        max_value=20,
        value=10
    )

    st.markdown("---")
    st.markdown("### Example Queries")
    example_queries = [
        "Julia Roberts romances from the 90s",
        "Dark psychological thrillers with a strong female lead",
        "My girlfriend broke up with me - something to cheer me up",
        "Epic war movies set in ancient times",
        "90s action movies with Arnold Schwarzenegger",
        "Coming-of-age movies from the 2010s",
    ]
    for eq in example_queries:
        if st.button(eq, key=f"example_{eq[:20]}"):
            st.session_state.query_input = eq
            st.session_state.search_results = None  # Clear old results
            st.session_state.all_movies_list = []

# Main content
st.markdown("---")

# Query input
query = st.text_input(
    "What kind of movie are you looking for?",
    value=st.session_state.get("query_input", ""),
    placeholder="e.g., 'sci-fi movies from the 80s with great special effects'",
    key="query_box"
)

# Search button
col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
with col2:
    if st.button("🎲 Random", use_container_width=True):
        import random
        query = random.choice(example_queries)
        st.session_state.query_input = query
        st.session_state.search_results = None
        st.session_state.all_movies_list = []
        st.rerun()

# Load recommender (always needed for details lookup)
recommender = load_recommender()

# Run NEW search only when button clicked
if search_clicked and query:
    try:
        with st.spinner(f"Finding movies matching: '{query}'..."):
            start_time = time.time()
            result = recommender.recommend(
                query=query,
                preference_mode=preference_mode,
                top_n=top_n
            )
            elapsed = time.time() - start_time

        # Store results in session state
        st.session_state.search_results = result
        st.session_state.last_query = query
        st.session_state.search_time = elapsed
        st.session_state.top_n = top_n

        # Build movie list for dropdown
        all_movies = []
        is_dual = (hasattr(result, 'dual_track_mode') and result.dual_track_mode and
                   hasattr(result, 'entity_track') and result.entity_track and
                   hasattr(result, 'mood_track') and result.mood_track)

        if is_dual:
            for i, m in enumerate(result.entity_track[:top_n], 1):
                all_movies.append((i, m.movie_title, m))
            start_idx = len(result.entity_track[:top_n]) + 1
            for i, m in enumerate(result.mood_track[:top_n], start_idx):
                all_movies.append((i, m.movie_title, m))
        else:
            for i, m in enumerate(result.recommendations[:top_n], 1):
                all_movies.append((i, m.movie_title, m))

        st.session_state.all_movies_list = all_movies

    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

# DISPLAY RESULTS from session state
if st.session_state.search_results is not None:
    result = st.session_state.search_results
    top_n = st.session_state.get('top_n', 10)

    if result.recommendations:
        st.success(f"Found movies in {st.session_state.get('search_time', 0):.1f}s for: \"{st.session_state.last_query}\"")

        # Show interpretation if available
        if hasattr(result, 'parsed_query') and result.parsed_query:
            with st.expander("Query Interpretation", expanded=False):
                pq = result.parsed_query
                cols = st.columns(4)
                genres = getattr(pq, 'genres', []) or []
                decades = getattr(pq, 'decades', []) or []
                actors = getattr(pq, 'actors', []) or []
                moods = getattr(pq, 'moods', []) or []
                if genres:
                    cols[0].markdown(f"**Genres:** {', '.join(str(x) for x in genres)}")
                if decades:
                    cols[1].markdown(f"**Decades:** {', '.join(str(x) for x in decades)}")
                if actors:
                    cols[2].markdown(f"**Actors:** {', '.join(str(x) for x in actors)}")
                if moods:
                    cols[3].markdown(f"**Moods:** {', '.join(str(x) for x in moods)}")

        st.markdown("---")

        # Check for DUAL-TRACK mode
        is_dual_track = (hasattr(result, 'dual_track_mode') and result.dual_track_mode and
                         hasattr(result, 'entity_track') and result.entity_track and
                         hasattr(result, 'mood_track') and result.mood_track)

        if is_dual_track:
            st.markdown("### 🎯 Dual-Track Results")
            st.info("Your query has both **content elements** and **mood elements**. Here are recommendations from both perspectives:")

            col_entity, col_mood = st.columns(2)

            with col_entity:
                st.markdown("#### 📌 Entity Track")
                st.caption("Content-focused (actors, themes, genres)")
                for i, movie in enumerate(result.entity_track[:top_n], 1):
                    display_movie_row(movie, i)

            with col_mood:
                st.markdown("#### 💭 Mood Track")
                st.caption("Theme/sentiment-focused")
                start_idx = len(result.entity_track[:top_n]) + 1
                for i, movie in enumerate(result.mood_track[:top_n], start_idx):
                    display_movie_row(movie, i)

        else:
            st.markdown(f"### 🎬 Top {min(top_n, len(result.recommendations))} Recommendations")
            for i, movie in enumerate(result.recommendations[:top_n], 1):
                display_movie_row(movie, i)
                st.markdown("---")

        # MOVIE DETAILS SECTION - Uses session state
        st.markdown("---")
        st.markdown("### 📖 Learn More About a Movie")

        if st.session_state.all_movies_list:
            movie_options = ["Select a movie..."] + [f"{idx}. {title}" for idx, title, _ in st.session_state.all_movies_list]

            selected = st.selectbox(
                "Choose a movie to see details:",
                movie_options,
                key="movie_detail_select"
            )

            if selected != "Select a movie...":
                # Extract movie title from selection
                selected_title = selected.split(". ", 1)[1] if ". " in selected else selected
                details = get_movie_details(recommender, selected_title)

                if details:
                    display_movie_details(details)
                else:
                    st.warning(f"Could not find details for '{selected_title}'")

    else:
        st.warning("No movies found matching your query. Try a different search!")

elif not search_clicked:
    # Show welcome message when no search has been done
    if st.session_state.search_results is None:
        st.info("👆 Enter a query above and click Search to get movie recommendations!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>Built with 6-signal hybrid architecture: Collaborative Filtering, Content Similarity,
        Theme Matching, Sentiment Analysis, Zero-shot Tags, and Query Relevance</p>
        <p>Data: TMDB (43,858 movies) | Models: SpaCy NER, Sentence-BERT, LDA, NeuMF</p>
    </div>
    """,
    unsafe_allow_html=True
)
