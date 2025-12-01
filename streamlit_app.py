"""
Kineto - Streamlit Movie Recommender App
A natural language movie recommendation system with 6-signal hybrid architecture.
Multi-page flow: welcome → auth_menu → login/signup → profile → search

Enhanced Features:
- Movie poster images from TMDB
- "Why This Recommendation?" signal breakdown
- Confidence score badges
- Query history in sidebar
- "Surprise Me" button
- Watchlist functionality
"""

import streamlit as st
import time
import pandas as pd
import re
import os
import random
import base64

# Page config
st.set_page_config(
    page_title="Kineto - Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# TMDB image base URL
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w200"
TMDB_IMAGE_LARGE = "https://image.tmdb.org/t/p/w400"

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize all session state variables."""
    defaults = {
        'page': 'welcome',
        'search_results': None,
        'all_movies_list': [],
        'last_query': '',
        'user_info': {},
        'logged_in': False,
        'appearance': 'dark',
        'query_history': [],
        'watchlist': [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# CUSTOM CSS STYLES
# =============================================================================
def get_theme_css():
    """Return CSS based on appearance setting."""
    base_css = """
    <style>
        .kineto-title {
            font-size: 4rem;
            font-weight: bold;
            text-align: center;
            margin: 20px 0;
        }
        .kineto-letter {
            color: #e50914;
            display: inline-block;
            animation: pulse 2s ease-in-out infinite;
        }
        .kineto-letter:nth-child(1) { animation-delay: 0s; }
        .kineto-letter:nth-child(2) { animation-delay: 0.1s; }
        .kineto-letter:nth-child(3) { animation-delay: 0.2s; }
        .kineto-letter:nth-child(4) { animation-delay: 0.3s; }
        .kineto-letter:nth-child(5) { animation-delay: 0.4s; }
        .kineto-letter:nth-child(6) { animation-delay: 0.5s; }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.8; transform: scale(1.05); }
        }
        .tagline {
            text-align: center;
            font-size: 1.2rem;
            color: #cccccc;
            margin-bottom: 30px;
        }
        .demo-notice {
            background-color: #2d2d2d;
            padding: 10px;
            border-radius: 5px;
            color: #ffc107;
            text-align: center;
            margin: 10px 0;
        }
        .footer-text { color: #888888; }

        /* Movie card styles */
        .movie-card {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #333;
        }

        /* Confidence badge styles */
        .confidence-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: bold;
            margin-left: 10px;
        }
        .badge-high {
            background-color: #28a745;
            color: white;
        }
        .badge-good {
            background-color: #ffc107;
            color: black;
        }
        .badge-try {
            background-color: #6c757d;
            color: white;
        }

        /* Signal bar styles */
        .signal-bar-container {
            margin: 5px 0;
        }
        .signal-bar-label {
            font-size: 0.8rem;
            color: #aaa;
            margin-bottom: 2px;
        }
        .signal-bar {
            height: 8px;
            border-radius: 4px;
            background-color: #333;
            overflow: hidden;
        }
        .signal-bar-fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
        .signal-cf { background-color: #e50914; }
        .signal-content { background-color: #1db954; }
        .signal-theme { background-color: #6366f1; }
        .signal-sentiment { background-color: #f59e0b; }
        .signal-tag { background-color: #ec4899; }
        .signal-query { background-color: #14b8a6; }

        /* Watchlist button */
        .watchlist-btn {
            cursor: pointer;
            font-size: 1.5rem;
        }
    </style>
    """
    return base_css

st.markdown(get_theme_css(), unsafe_allow_html=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def validate_email(email):
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def render_kineto_title():
    """Render the animated Kineto title with fly-in animation."""
    st.markdown("""
        <style>
        @keyframes flyLeft {
            0%{opacity:0;transform:translateX(-100px) rotate(-20deg);}
            80%{opacity:1;transform:translateX(10px) rotate(2deg);}
            100%{opacity:1;transform:translateX(0) rotate(0);}
        }
        @keyframes flyRight {
            0%{opacity:0;transform:translateX(100px) rotate(20deg);}
            80%{opacity:1;transform:translateX(-10px) rotate(-2deg);}
            100%{opacity:1;transform:translateX(0) rotate(0);}
        }
        @keyframes flyTop {
            0%{opacity:0;transform:translateY(-100px) rotate(10deg);}
            80%{opacity:1;transform:translateY(10px) rotate(-2deg);}
            100%{opacity:1;transform:translateY(0) rotate(0);}
        }

        .title-container {
            text-align:center;
            font-size:48px;
            font-weight:bold;
            margin-bottom:10px;
        }

        .title-letter {
            display:inline-block;
            color:#E50914;
            opacity:0;
        }

        .title-letter:nth-child(1){ animation:flyLeft .8s ease forwards .1s; }
        .title-letter:nth-child(2){ animation:flyRight .8s ease forwards .3s; }
        .title-letter:nth-child(3){ animation:flyTop .8s ease forwards .5s; }
        .title-letter:nth-child(4){ animation:flyLeft .8s ease forwards .7s; }
        .title-letter:nth-child(5){ animation:flyRight .8s ease forwards .9s; }
        .title-letter:nth-child(6){ animation:flyTop .8s ease forwards 1.1s; }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="title-container">
            <span class="title-letter">K</span>
            <span class="title-letter">I</span>
            <span class="title-letter">N</span>
            <span class="title-letter">E</span>
            <span class="title-letter">T</span>
            <span class="title-letter">O</span>
        </div>
    """, unsafe_allow_html=True)

def render_logo():
    """Render the Kineto logo, perfectly centered."""
    logo_path = "kineto_logo.png"

    if os.path.exists(logo_path):
        # Read image and encode as base64 so we can control alignment via HTML/CSS
        with open(logo_path, "rb") as f:
            logo_base64 = base64.b64encode(f.read()).decode()

        st.markdown("""
            <style>
                .logo-container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    margin-top: 20px;
                    margin-bottom: 10px;
                }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="logo-container">
                <img src="data:image/png;base64,{logo_base64}" width="200">
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown("<h1 style='text-align: center;'>🎬</h1>", unsafe_allow_html=True)


def navigate_to(page):
    """Navigate to a different page."""
    st.session_state.page = page
    st.rerun()

def get_poster_url(poster_path, size="w200"):
    """Get TMDB poster URL."""
    if poster_path and isinstance(poster_path, str) and poster_path.startswith('/'):
        return f"https://image.tmdb.org/t/p/{size}{poster_path}"
    return None

def calculate_confidence(movie):
    """Calculate confidence score based on signal agreement."""
    scores = [
        getattr(movie, 'cf_score', 0) or 0,
        getattr(movie, 'content_score', 0) or 0,
        getattr(movie, 'theme_score', 0) or 0,
        getattr(movie, 'sentiment_score', 0) or 0,
        getattr(movie, 'tag_score', 0) or 0,
        getattr(movie, 'query_score', 0) or 0,
    ]
    # Weighted average with query_score having highest weight
    weights = [0.10, 0.15, 0.15, 0.05, 0.25, 0.30]
    confidence = sum(s * w for s, w in zip(scores, weights))
    return min(confidence * 100, 99)  # Cap at 99%

def get_confidence_badge(confidence):
    """Get confidence badge HTML."""
    if confidence >= 70:
        return f'<span class="confidence-badge badge-high">{confidence:.0f}% Match</span>'
    elif confidence >= 45:
        return f'<span class="confidence-badge badge-good">{confidence:.0f}% Match</span>'
    else:
        return f'<span class="confidence-badge badge-try">Worth a Try</span>'

def render_signal_bars(movie):
    """Render signal breakdown as progress bars."""
    signals = [
        ("CF (Collaborative)", getattr(movie, 'cf_score', 0) or 0, "signal-cf"),
        ("Content Similarity", getattr(movie, 'content_score', 0) or 0, "signal-content"),
        ("Theme Match", getattr(movie, 'theme_score', 0) or 0, "signal-theme"),
        ("Sentiment", getattr(movie, 'sentiment_score', 0) or 0, "signal-sentiment"),
        ("Zero-shot Tags", getattr(movie, 'tag_score', 0) or 0, "signal-tag"),
        ("Query Relevance", getattr(movie, 'query_score', 0) or 0, "signal-query"),
    ]

    html = ""
    for label, score, css_class in signals:
        width = min(score * 100, 100)
        html += f'''
        <div class="signal-bar-container">
            <div class="signal-bar-label">{label}: {score:.2f}</div>
            <div class="signal-bar">
                <div class="signal-bar-fill {css_class}" style="width: {width}%"></div>
            </div>
        </div>
        '''
    return html

def add_to_query_history(query):
    """Add query to history (max 5)."""
    if query and query not in st.session_state.query_history:
        st.session_state.query_history.insert(0, query)
        st.session_state.query_history = st.session_state.query_history[:5]

def toggle_watchlist(movie_title, movie_year):
    """Toggle movie in watchlist."""
    item = f"{movie_title} ({movie_year})"
    if item in st.session_state.watchlist:
        st.session_state.watchlist.remove(item)
    else:
        st.session_state.watchlist.append(item)

# =============================================================================
# PAGE: WELCOME
# =============================================================================
def page_welcome():
    """Welcome/landing page."""
    st.markdown("<br><br>", unsafe_allow_html=True)

    render_logo()
    render_kineto_title()

    st.markdown("""
        <p class="tagline">
            Your personalized movie recommendation assistant powered by hybrid AI.
        </p>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🎬 Get Started", type="primary", use_container_width=True):
            navigate_to('auth_menu')

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
            <div class="demo-notice">
                📢 <strong>Demo Mode</strong>: This is a demonstration version.
                No real authentication or data storage.
            </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PAGE: AUTH MENU
# =============================================================================
def page_auth_menu():
    """Authentication menu - choose login or signup."""
    st.markdown("<br><br>", unsafe_allow_html=True)

    render_kineto_title()

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔑 Log In", use_container_width=True):
            navigate_to('login')

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🆕 Create Account", use_container_width=True):
            navigate_to('signup')

        st.markdown("<br><br>", unsafe_allow_html=True)

        if st.button("← Back", use_container_width=True):
            navigate_to('welcome')

# =============================================================================
# PAGE: SIGNUP
# =============================================================================
def page_signup():
    """Account creation page."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🆕 Create Account")
        st.markdown("""
            <div class="demo-notice">
                📢 <strong>Demo Mode</strong>: No real account will be created.
                Just fill in any valid format to proceed.
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        email = st.text_input("Email Address", placeholder="your.email@example.com")
        password = st.text_input("Password", type="password", placeholder="Enter password")
        verify_password = st.text_input("Verify Password", type="password", placeholder="Re-enter password")

        error_msg = None
        if email and not validate_email(email):
            error_msg = "Please enter a valid email address."
        elif password and verify_password and password != verify_password:
            error_msg = "Passwords do not match."
        elif password and len(password) < 6:
            error_msg = "Password must be at least 6 characters."

        if error_msg:
            st.error(error_msg)

        st.markdown("<br>", unsafe_allow_html=True)

        col_back, col_create = st.columns(2)
        with col_back:
            if st.button("← Back", use_container_width=True):
                navigate_to('auth_menu')
        with col_create:
            can_create = email and password and verify_password and not error_msg
            if st.button("Create Account", type="primary", use_container_width=True, disabled=not can_create):
                st.session_state.user_info['email'] = email
                st.session_state.logged_in = True
                navigate_to('profile')

# =============================================================================
# PAGE: LOGIN
# =============================================================================
def page_login():
    """Login page."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 🔑 Log In")
        st.markdown("""
            <div class="demo-notice">
                📢 <strong>Demo Mode</strong>: Enter any valid email format to proceed.
            </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        email = st.text_input("Email Address", placeholder="your.email@example.com")
        password = st.text_input("Password", type="password", placeholder="Enter password")

        error_msg = None
        if email and not validate_email(email):
            error_msg = "Please enter a valid email address."

        if error_msg:
            st.error(error_msg)

        st.markdown("<br>", unsafe_allow_html=True)

        col_back, col_login = st.columns(2)
        with col_back:
            if st.button("← Back", use_container_width=True):
                navigate_to('auth_menu')
        with col_login:
            can_login = email and password and not error_msg
            if st.button("Log In", type="primary", use_container_width=True, disabled=not can_login):
                st.session_state.user_info['email'] = email
                st.session_state.logged_in = True
                navigate_to('search')

# =============================================================================
# PAGE: PROFILE
# =============================================================================
def page_profile():
    """Profile setup page."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("## 👤 Let's personalize your experience")
        st.markdown("Tell us a bit about yourself to get better recommendations.")

        st.markdown("<br>", unsafe_allow_html=True)

        # Name
        col_first, col_last = st.columns(2)
        with col_first:
            first_name = st.text_input("First Name", value=st.session_state.user_info.get('first_name', ''))
        with col_last:
            last_name = st.text_input("Last Name", value=st.session_state.user_info.get('last_name', ''))

        # Date of Birth
        st.markdown("**Date of Birth**")
        col_day, col_month, col_year = st.columns(3)
        with col_day:
            day = st.selectbox("Day", options=[''] + list(range(1, 32)), index=0)
        with col_month:
            months = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                     'July', 'August', 'September', 'October', 'November', 'December']
            month = st.selectbox("Month", options=months, index=0)
        with col_year:
            current_year = 2024
            year = st.selectbox("Year", options=[''] + list(range(current_year - 100, current_year - 10)), index=0)

        # Gender
        gender = st.selectbox("Gender", options=['', 'Male', 'Female', 'Non-binary', 'Prefer not to say'], index=0)

        # Phone
        st.markdown("**Phone Number** (optional)")
        col_code, col_phone = st.columns([1, 3])
        with col_code:
            country_codes = ['+1 (US)', '+44 (UK)', '+91 (IN)', '+61 (AU)', '+81 (JP)', '+86 (CN)', '+33 (FR)', '+49 (DE)']
            phone_code = st.selectbox("Code", options=[''] + country_codes, index=0)
        with col_phone:
            phone_number = st.text_input("Phone Number", placeholder="555-123-4567")

        # Location
        st.markdown("**Location** (optional)")
        col_city, col_state = st.columns(2)
        with col_city:
            city = st.text_input("City", placeholder="New York")
        with col_state:
            state = st.text_input("State/Province", placeholder="NY")

        countries = ['', 'United States', 'United Kingdom', 'Canada', 'Australia',
                    'Germany', 'France', 'Japan', 'India', 'Brazil', 'Mexico', 'Other']
        country = st.selectbox("Country", options=countries, index=0)

        st.markdown("<br>", unsafe_allow_html=True)

        col_back, col_next = st.columns(2)
        with col_back:
            if st.button("← Back", use_container_width=True):
                navigate_to('auth_menu')
        with col_next:
            if st.button("Next →", type="primary", use_container_width=True):
                st.session_state.user_info.update({
                    'first_name': first_name,
                    'last_name': last_name,
                    'dob_day': day,
                    'dob_month': month,
                    'dob_year': year,
                    'gender': gender,
                    'phone_code': phone_code,
                    'phone_number': phone_number,
                    'city': city,
                    'state': state,
                    'country': country
                })
                navigate_to('search')

# =============================================================================
# PAGE: SEARCH (Main Recommendation Interface)
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_recommender():
    """Load the movie recommender system."""
    from src.recommender_interactive_v4 import MovieRecommenderInteractiveV4
    return MovieRecommenderInteractiveV4()


def get_movie_details(recommender, movie_title):
    """Look up movie details from TMDB data."""
    movies_df = recommender.movies

    match = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    if match.empty:
        match = movies_df[movies_df['title'].str.lower().str.contains(movie_title.lower(), na=False)]

    if match.empty:
        return None

    movie = match.iloc[0]

    def safe_get(key, default='N/A'):
        val = movie.get(key, default)
        if val is None:
            return default
        if isinstance(val, (list, tuple)):
            return val if len(val) > 0 else default
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
        'poster_path': safe_get('poster_path', None),
    }


def get_movie_poster_path(recommender, movie_title):
    """Get poster path for a movie."""
    movies_df = recommender.movies
    match = movies_df[movies_df['title'].str.lower() == movie_title.lower()]
    if match.empty:
        match = movies_df[movies_df['title'].str.lower().str.contains(movie_title.lower(), na=False)]
    if not match.empty:
        poster = match.iloc[0].get('poster_path', None)
        if poster and isinstance(poster, str):
            return poster
    return None


def display_movie_details(details):
    """Display formatted movie details."""
    st.markdown("---")

    col_poster, col_info = st.columns([1, 3])

    with col_poster:
        poster_url = get_poster_url(details.get('poster_path'), "w400")
        if poster_url:
            st.image(poster_url, width=200)
        else:
            st.markdown("🎬 No poster available")

    with col_info:
        st.subheader(f"📽️ {details['title']} ({details['year']})")

        directors = details['directors']
        if isinstance(directors, list) and directors:
            st.markdown(f"**Director(s):** {', '.join(str(d) for d in directors[:3])}")
        elif directors and directors != 'N/A':
            st.markdown(f"**Director(s):** {directors}")

        cast = details['cast']
        if isinstance(cast, list) and cast:
            st.markdown(f"**Cast:** {', '.join(str(c) for c in cast[:6])}")
        elif cast and cast != 'N/A':
            st.markdown(f"**Cast:** {cast}")

        genres = details['genres']
        if isinstance(genres, list) and genres:
            st.markdown(f"**Genres:** {', '.join(str(g) for g in genres)}")

        if details['runtime'] and details['runtime'] != 'N/A':
            st.markdown(f"**Runtime:** {details['runtime']} min")

        if details['vote_average'] and details['vote_average'] != 'N/A':
            st.markdown(f"**TMDB Rating:** {details['vote_average']}/10")

    st.markdown("**Overview:**")
    overview = details['overview']
    if len(str(overview)) > 800:
        overview = str(overview)[:800] + "..."
    st.markdown(f"*{overview}*")


def display_movie_card(movie, index, recommender):
    """Display enhanced movie card with poster and signals."""
    movie_title = movie.movie_title
    year = getattr(movie, 'year', 'N/A')
    overview = getattr(movie, 'overview', '')
    final_score = getattr(movie, 'final_score', 0)

    # Get poster
    poster_path = get_movie_poster_path(recommender, movie_title)
    poster_url = get_poster_url(poster_path) if poster_path else None

    # Calculate confidence
    confidence = calculate_confidence(movie)

    # Create card layout
    col_poster, col_info, col_actions = st.columns([1, 4, 1])

    with col_poster:
        if poster_url:
            st.image(poster_url, width=100)
        else:
            st.markdown("🎬")

    with col_info:
        # Title with confidence badge
        badge_html = get_confidence_badge(confidence)
        st.markdown(f"**{index}. {movie_title}** ({year}) {badge_html}", unsafe_allow_html=True)

        # Truncated overview
        if overview:
            display_overview = str(overview)[:150] + "..." if len(str(overview)) > 150 else overview
            st.caption(display_overview)

        # Score
        st.markdown(f"**Score:** {final_score:.2f}")

    with col_actions:
        # Watchlist button
        item = f"{movie_title} ({year})"
        is_in_watchlist = item in st.session_state.watchlist
        btn_label = "❤️" if is_in_watchlist else "🤍"
        if st.button(btn_label, key=f"wl_{index}_{movie_title[:10]}", help="Add to watchlist"):
            toggle_watchlist(movie_title, year)
            st.rerun()

    # Expandable signal breakdown
    with st.expander("🔍 Why this recommendation?"):
        st.markdown(render_signal_bars(movie), unsafe_allow_html=True)
        st.caption("These 6 signals from our hybrid AI architecture determined this match.")

    st.markdown("---")


def page_search():
    """Main search/recommendation page."""
    first_name = st.session_state.user_info.get('first_name', 'there')
    if not first_name:
        first_name = 'there'

    st.title(f"Hi {first_name}! 👋")
    st.markdown("### What do you feel like watching?")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        appearance = st.selectbox(
            "Appearance",
            options=["dark", "light", "system"],
            index=["dark", "light", "system"].index(st.session_state.appearance),
            help="Choose your preferred theme"
        )
        if appearance != st.session_state.appearance:
            st.session_state.appearance = appearance
            st.rerun()

        st.markdown("---")

        preference_mode = st.selectbox(
            "Preference Mode",
            options=["balanced", "accuracy", "ratings"],
            index=0,
            help="""
            - **Balanced**: Mix of relevance and quality
            - **Accuracy**: Pure content matching
            - **Ratings**: Prioritizes popular movies
            """
        )

        top_n = st.slider("Results per Track", min_value=5, max_value=20, value=10)

        # Query History
        st.markdown("---")
        st.markdown("### 🕐 Recent Searches")
        if st.session_state.query_history:
            for hist_query in st.session_state.query_history:
                if st.button(f"📝 {hist_query[:30]}...", key=f"hist_{hist_query[:15]}", use_container_width=True):
                    st.session_state.query_input = hist_query
                    st.session_state.search_results = None
                    st.rerun()
        else:
            st.caption("No recent searches")

        # Watchlist
        st.markdown("---")
        watchlist_count = len(st.session_state.watchlist)
        st.markdown(f"### ❤️ Watchlist ({watchlist_count})")
        if st.session_state.watchlist:
            with st.expander("View Watchlist"):
                for item in st.session_state.watchlist:
                    col_item, col_remove = st.columns([4, 1])
                    with col_item:
                        st.markdown(f"• {item}")
                    with col_remove:
                        if st.button("✕", key=f"rm_{item[:10]}"):
                            st.session_state.watchlist.remove(item)
                            st.rerun()
        else:
            st.caption("No movies saved yet")

        st.markdown("---")
        st.markdown("### 💡 Example Queries")
        example_queries = [
            "Julia Roberts romances from the 90s",
            "Dark psychological thrillers with a strong female lead",
            "My girlfriend broke up with me - something to cheer me up",
            "Epic war movies set in ancient times",
            "90s action movies with Arnold Schwarzenegger",
        ]
        for eq in example_queries[:3]:
            if st.button(eq[:35] + "...", key=f"ex_{eq[:15]}", use_container_width=True):
                st.session_state.query_input = eq
                st.session_state.search_results = None
                st.rerun()

        st.markdown("---")
        if st.session_state.user_info.get('email'):
            st.caption(f"Logged in as: {st.session_state.user_info['email']}")

        if st.button("🚪 Log Out", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_info = {}
            st.session_state.page = 'welcome'
            st.session_state.search_results = None
            st.session_state.query_history = []
            st.session_state.watchlist = []
            st.rerun()

    # Main content
    st.markdown("---")

    query = st.text_input(
        "Describe what you're looking for:",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g., 'sci-fi movies from the 80s with great special effects'",
        key="query_box"
    )

    # Action buttons
    col1, col2, col3, col4 = st.columns([1, 1, 1, 3])
    with col1:
        search_clicked = st.button("🔍 Search", type="primary", use_container_width=True)
    with col2:
        if st.button("🎲 Random", use_container_width=True):
            query = random.choice(example_queries)
            st.session_state.query_input = query
            st.session_state.search_results = None
            st.rerun()
    with col3:
        surprise_clicked = st.button("✨ Surprise Me", use_container_width=True,
                                     help="Get a hidden gem recommendation")

    # Custom loading message
    loading_messages = [
        "🎬 Kineto is analyzing your taste...",
        "🎥 Searching through 43,858 movies...",
        "🍿 Finding your perfect match...",
        "🎞️ Running 6-signal hybrid analysis...",
    ]

    # Load recommender
    with st.spinner(random.choice(loading_messages)):
        recommender = load_recommender()

    # Handle Surprise Me
    if surprise_clicked:
        with st.spinner("✨ Finding a hidden gem for you..."):
            # Get random highly-rated but less popular movie
            movies_df = recommender.movies
            hidden_gems = movies_df[
                (movies_df['vote_average'] >= 7.0) &
                (movies_df['vote_count'] >= 100) &
                (movies_df['vote_count'] < 5000)
            ]
            if not hidden_gems.empty:
                gem = hidden_gems.sample(1).iloc[0]
                st.success(f"✨ **Hidden Gem:** {gem['title']} ({gem.get('year', 'N/A')})")
                st.markdown(f"**Rating:** {gem['vote_average']}/10 ({gem['vote_count']} votes)")

                poster_url = get_poster_url(gem.get('poster_path'))
                if poster_url:
                    st.image(poster_url, width=200)

                overview = gem.get('overview', 'No overview available.')
                st.markdown(f"*{overview[:500]}...*" if len(str(overview)) > 500 else f"*{overview}*")

    # Run search
    if search_clicked and query:
        add_to_query_history(query)

        try:
            with st.spinner(random.choice(loading_messages)):
                start_time = time.time()
                result = recommender.recommend(
                    query=query,
                    preference_mode=preference_mode,
                    top_n=top_n
                )
                elapsed = time.time() - start_time

            st.session_state.search_results = result
            st.session_state.last_query = query
            st.session_state.search_time = elapsed
            st.session_state.top_n = top_n

            # Build movie list
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

    # Display results
    if st.session_state.search_results is not None:
        result = st.session_state.search_results
        top_n = st.session_state.get('top_n', 10)

        # Debug info for troubleshooting
        with st.expander("🔧 Debug Info", expanded=False):
            st.write(f"**Recommendations count:** {len(result.recommendations) if result.recommendations else 0}")
            st.write(f"**Num candidates:** {getattr(result, 'num_candidates', 'N/A')}")
            st.write(f"**Dual track mode:** {getattr(result, 'dual_track_mode', False)}")
            if hasattr(result, 'parsed_query') and result.parsed_query:
                pq = result.parsed_query
                st.write(f"**Parsed themes:** {getattr(pq, 'themes', [])}")
                st.write(f"**Parsed keywords:** {getattr(pq, 'keywords', [])}")
                st.write(f"**Adult context:** {getattr(pq, 'adult_context', False)}")

            # Clause detection diagnostics
            try:
                from src.recommender_interactive_v4 import _split_query_into_clauses, detect_situation_outcome_query
                clauses = _split_query_into_clauses(st.session_state.last_query)
                sit_out = detect_situation_outcome_query(st.session_state.last_query)
                st.write(f"**Clauses detected:** {clauses}")
                st.write(f"**Situation+Outcome detected:** {sit_out.get('is_situation_outcome', False)}")
                st.write(f"**Situations:** {sit_out.get('situations', [])}")
                st.write(f"**Outcomes:** {sit_out.get('outcomes', [])}")
            except Exception as e:
                st.write(f"**Clause detection error:** {e}")

        if result.recommendations:
            st.success(f"Found movies in {st.session_state.get('search_time', 0):.1f}s for: \"{st.session_state.last_query}\"")

            # Query interpretation
            if hasattr(result, 'parsed_query') and result.parsed_query:
                with st.expander("🎯 Query Interpretation", expanded=False):
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

            # Dual-track check
            is_dual_track = (hasattr(result, 'dual_track_mode') and result.dual_track_mode and
                             hasattr(result, 'entity_track') and result.entity_track and
                             hasattr(result, 'mood_track') and result.mood_track)

            if is_dual_track:
                st.markdown("### 🎯 Dual-Track Results")
                st.info("Your query has both **content** and **mood** elements. Here are recommendations from both perspectives:")

                tab1, tab2 = st.tabs(["📌 Entity Track", "💭 Mood Track"])

                with tab1:
                    st.caption("Content-focused (actors, themes, genres)")
                    for i, movie in enumerate(result.entity_track[:top_n], 1):
                        display_movie_card(movie, i, recommender)

                with tab2:
                    st.caption("Theme/sentiment-focused")
                    for i, movie in enumerate(result.mood_track[:top_n], 1):
                        display_movie_card(movie, i, recommender)

            else:
                st.markdown(f"### 🎬 Top {min(top_n, len(result.recommendations))} Recommendations")
                for i, movie in enumerate(result.recommendations[:top_n], 1):
                    display_movie_card(movie, i, recommender)

            # Movie details lookup
            st.markdown("---")
            st.markdown("### 📖 Movie Details")

            if st.session_state.all_movies_list:
                movie_options = ["Select a movie..."] + [f"{idx}. {title}" for idx, title, _ in st.session_state.all_movies_list]

                selected = st.selectbox(
                    "Choose a movie to see full details:",
                    movie_options,
                    key="movie_detail_select"
                )

                if selected != "Select a movie...":
                    selected_title = selected.split(". ", 1)[1] if ". " in selected else selected
                    details = get_movie_details(recommender, selected_title)

                    if details:
                        display_movie_details(details)
                    else:
                        st.warning(f"Could not find details for '{selected_title}'")

        else:
            st.warning("No movies found matching your query. Try a different search!")
            # Show diagnostic info for debugging
            st.info(f"""
**Debug Info:**
- Recommendations: {len(result.recommendations) if result.recommendations else 0}
- Candidates: {getattr(result, 'num_candidates', 'N/A')}
- Dual track: {getattr(result, 'dual_track_mode', False)}
- Entity track: {len(result.entity_track) if hasattr(result, 'entity_track') and result.entity_track else 0}
- Mood track: {len(result.mood_track) if hasattr(result, 'mood_track') and result.mood_track else 0}
            """)

    elif not search_clicked and not surprise_clicked:
        if st.session_state.search_results is None:
            st.info("👆 Enter a query above and click Search to get movie recommendations!")

# =============================================================================
# FOOTER
# =============================================================================
def render_footer():
    """Render the footer with links."""
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center;' class='footer-text'>
            <p>Built with 6-signal hybrid architecture: Collaborative Filtering, Content Similarity,
            Theme Matching, Sentiment Analysis, Zero-shot Tags, and Query Relevance</p>
            <p>Data: TMDB (43,858 movies) | Models: SpaCy NER, Sentence-BERT, LDA, NeuMF</p>
            <p>
                <a href="https://github.com/Anabasis2025/Capstone_Movie_Recommender_Final" target="_blank">
                    📦 GitHub Repository
                </a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# MAIN APP ROUTER
# =============================================================================
def main():
    """Main app router based on current page."""
    page = st.session_state.page

    if page == 'welcome':
        page_welcome()
    elif page == 'auth_menu':
        page_auth_menu()
    elif page == 'signup':
        page_signup()
    elif page == 'login':
        page_login()
    elif page == 'profile':
        page_profile()
    elif page == 'search':
        page_search()
    else:
        page_welcome()

    render_footer()

if __name__ == "__main__":
    main()
