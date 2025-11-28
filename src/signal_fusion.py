"""
Signal Fusion Module
Computes all 6 recommendation signals for candidate movies.

FIXES THE CORE BUG: CF and content scores are now actually calculated and used!

The 6 Signals:
1. Collaborative Filtering (CF) - SVD + NeuMF predictions
2. Content Similarity - Embedding cosine similarity
3. Theme Matching - LDA topic overlap
4. Sentiment Matching - Emotion profile similarity  
5. Zero-shot Tag Matching - Tag overlap with rare tag boosting
6. Query Relevance - Semantic search score

Usage:
    fusion = SignalFusion()
    signals = fusion.compute_all_signals(
        candidate_movies=['Toy Story', 'The Matrix'],
        query_keywords=['action', 'sci-fi'],
        user_id=123  # Optional for CF
    )
"""

import pickle
import joblib  # For loading sklearn objects (TfidfVectorizer, etc.)
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
from sklearn.metrics.pairwise import cosine_similarity
import warnings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SignalScores:
    """Container for all signal scores for a single movie."""
    movie_title: str
    cf_score: float = 0.0           # Collaborative filtering
    content_score: float = 0.0      # Content similarity
    theme_score: float = 0.0        # Theme matching
    sentiment_score: float = 0.0    # Sentiment matching
    tag_score: float = 0.0          # Zero-shot tag matching
    query_score: float = 0.0        # Query relevance
    tmdb_keyword_matches: int = 0   # Number of TMDB keywords matched (for conditional zero-shot)
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'movie_title': self.movie_title,
            'cf_score': self.cf_score,
            'content_score': self.content_score,
            'theme_score': self.theme_score,
            'sentiment_score': self.sentiment_score,
            'tag_score': self.tag_score,
            'query_score': self.query_score,
            'tmdb_keyword_matches': self.tmdb_keyword_matches
        }


class SignalFusion:
    """Computes all 6 recommendation signals with robust error handling."""
    
    def __init__(self, 
                 models_dir: str = "models",
                 data_dir: str = "data/raw"):
        """
        Initialize signal fusion with model/data paths.
        
        Args:
            models_dir: Directory containing trained models
            data_dir: Directory containing processed data
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Model/data containers
        self.svd_model = None
        self.ncf_model = None
        self.movie_embeddings = None
        self.movie_themes = None
        self.movie_sentiments = None
        self.zero_shot_tags = None
        self.tfidf_vectorizer = None

        # ID mappings
        self.movie_to_idx = {}
        self.user_to_idx = {}
        self.title_to_embedding_idx = {}  # Maps movie title to embedding array index
        self.freebase_to_embedding_idx = {}  # Maps Freebase ID to embedding array index
        self.title_to_theme_idx = {}      # Maps movie title to theme array index
        self.title_to_freebase = {}       # Maps movie title to Freebase ID
        
        # Model availability flags
        self.cf_available = False
        self.content_available = False
        self.themes_available = False
        self.sentiments_available = False
        self.tags_available = False
        
        # Load everything
        self._load_all()
    
    def _load_all(self):
        """Load all models and data with robust error handling."""
        logger.info("Loading models and data...")
        
        # Load ID mappings (critical)
        self._load_id_mappings()
        
        # Load CF models (SVD + NeuMF)
        self._load_cf_models()
        
        # Load content similarity components
        self._load_content_components()
        
        # Load theme model
        self._load_theme_model()
        
        # Load sentiment data
        self._load_sentiment_data()
        
        # Load zero-shot tags
        self._load_zero_shot_tags()
        
        # Summary
        self._print_availability_summary()
    
    def _load_id_mappings(self):
        """Load movie and user ID mappings."""
        try:
            # Movie mapping
            movie_path = self.data_dir / "movie_to_idx.json"
            with open(movie_path, 'r') as f:
                self.movie_to_idx = json.load(f)
            logger.info(f"✅ Loaded movie_to_idx: {len(self.movie_to_idx)} movies")
            
            # User mapping (optional)
            user_path = self.data_dir / "user_to_idx.json"
            if user_path.exists():
                with open(user_path, 'r') as f:
                    self.user_to_idx = json.load(f)
                logger.info(f"✅ Loaded user_to_idx: {len(self.user_to_idx)} users")
        except Exception as e:
            logger.warning(f"⚠️ Error loading ID mappings: {e}")
    
    def _load_cf_models(self):
        """Load collaborative filtering models (SVD + NeuMF)."""
        # Try SVD
        try:
            svd_path = self.models_dir / "svd_model.pkl"
            with open(svd_path, 'rb') as f:
                # Try standard load
                try:
                    self.svd_model = pickle.load(f)
                except Exception:
                    # Fallback: try with latin1 encoding
                    f.seek(0)
                    self.svd_model = pickle.load(f, encoding='latin1')
            
            logger.info("✅ Loaded SVD model")
            self.cf_available = True
        except Exception as e:
            logger.warning(f"⚠️ Could not load SVD model: {e}")
            logger.warning("   CF scoring will be disabled")
        
        # Try NeuMF
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            ncf_path = self.models_dir / "ncf_model.keras"
            self.ncf_model = keras.models.load_model(ncf_path)
            logger.info("✅ Loaded NeuMF model")
            self.cf_available = True
        except Exception as e:
            logger.warning(f"⚠️ Could not load NeuMF model: {e}")
    
    def _load_content_components(self):
        """Load content-based similarity components."""
        try:
            # Load ENRICHED movie embeddings (43K movies with full TMDB data coverage)
            emb_path = self.models_dir / "enriched_movie_embeddings.npy"
            self.movie_embeddings = np.load(emb_path)
            logger.info(f"✅ Loaded enriched movie embeddings: {self.movie_embeddings.shape}")

            # Load enriched movie IDs (Freebase IDs like 'm/0814255')
            movie_ids_path = self.models_dir / "enriched_movie_ids.npy"
            if movie_ids_path.exists():
                movie_ids = np.load(movie_ids_path, allow_pickle=True)
                # Create mapping: freebase_id -> embedding_array_index
                self.freebase_to_embedding_idx = {
                    freebase_id: idx for idx, freebase_id in enumerate(movie_ids)
                }
                logger.info(f"✅ Created Freebase->embedding mapping: {len(self.freebase_to_embedding_idx)} movies")

                # Load TMDB data to create title -> Freebase ID mapping
                tmdb_path = self.data_dir / "tmdb_fully_enriched.parquet"
                if tmdb_path.exists():
                    tmdb_df = pd.read_parquet(tmdb_path)
                    # Create title -> Freebase ID mapping (if not already loaded)
                    if not self.title_to_freebase:
                        for _, row in tmdb_df.iterrows():
                            title = row.get('tmdb_title') or row.get('original_title')
                            freebase_id = row.get('original_movie_id')
                            if title and freebase_id:
                                self.title_to_freebase[title] = freebase_id

                    # Create title -> embedding index mapping (for backward compatibility)
                    # This maps title -> freebase_id -> embedding_idx
                    self.title_to_embedding_idx = {}
                    for title, freebase_id in self.title_to_freebase.items():
                        if freebase_id in self.freebase_to_embedding_idx:
                            self.title_to_embedding_idx[title] = self.freebase_to_embedding_idx[freebase_id]

                    logger.info(f"✅ Created title->embedding mapping: {len(self.title_to_embedding_idx)} movies")

                self.content_available = True
            else:
                logger.warning("⚠️ enriched_movie_ids.npy not found - content similarity will be limited")

            # Load TF-IDF vectorizer (optional) - uses joblib for sklearn objects
            tfidf_path = self.data_dir / "tfidf_vectorizer.pkl"
            if tfidf_path.exists():
                try:
                    self.tfidf_vectorizer = joblib.load(tfidf_path)
                    logger.info("✅ Loaded TF-IDF vectorizer")
                except Exception as tfidf_err:
                    logger.debug(f"Could not load TF-IDF vectorizer: {tfidf_err}")
        except Exception as e:
            logger.warning(f"⚠️ Could not load content components: {e}")
    
    def _load_theme_model(self):
        """Load LDA theme model and theme assignments."""
        try:
            theme_path = self.data_dir / "movie_themes.pkl"
            with open(theme_path, 'rb') as f:
                theme_data = pickle.load(f)

            # BUG FIX: Restructure theme data from dict format
            # Expected format: {'movie_ids': [...], 'theme_distributions': ndarray, ...}
            if isinstance(theme_data, dict) and 'theme_distributions' in theme_data:
                # Extract parallel arrays
                movie_ids = theme_data.get('movie_ids', [])
                theme_distributions = theme_data.get('theme_distributions')

                # Create mapping: movie_id (Freebase) -> theme_vector
                self.movie_themes = {}
                for idx, movie_id in enumerate(movie_ids):
                    if idx < len(theme_distributions):
                        self.movie_themes[movie_id] = theme_distributions[idx]

                # Load TMDB enriched data to get title -> Freebase ID mapping
                try:
                    tmdb_path = self.data_dir / "tmdb_fully_enriched.parquet"
                    if tmdb_path.exists():
                        tmdb_df = pd.read_parquet(tmdb_path)

                        # Create title -> Freebase ID mapping
                        for _, row in tmdb_df.iterrows():
                            title = row.get('tmdb_title') or row.get('original_title')
                            freebase_id = row.get('original_movie_id')
                            if title and freebase_id:
                                self.title_to_freebase[title] = freebase_id

                        logger.info(f"✅ Created title->Freebase mapping: {len(self.title_to_freebase)} movies")
                except Exception as e:
                    logger.warning(f"⚠️ Could not load title->Freebase mapping: {e}")

                logger.info(f"✅ Loaded movie themes: {len(self.movie_themes)} movies with {theme_distributions.shape[1] if hasattr(theme_distributions, 'shape') else '?'} topics")
                self.themes_available = True
            elif isinstance(theme_data, dict):
                # Already in correct format (movie_title -> theme_vector)
                logger.info(f"✅ Loaded movie themes: {len(theme_data)} movies")
                self.movie_themes = theme_data
                self.themes_available = True
            else:
                logger.info(f"✅ Loaded movie themes: {theme_data.shape if hasattr(theme_data, 'shape') else 'unknown format'}")
                self.movie_themes = theme_data
                self.themes_available = True

        except Exception as e:
            logger.warning(f"⚠️ Could not load theme model: {e}")
    
    def _load_sentiment_data(self):
        """Load sentiment analysis results."""
        try:
            sent_path = self.data_dir / "movie_sentiments.pkl"
            with open(sent_path, 'rb') as f:
                self.movie_sentiments = pickle.load(f)
            logger.info(f"✅ Loaded sentiment data: {self.movie_sentiments.shape}")
            self.sentiments_available = True
        except Exception as e:
            logger.warning(f"⚠️ Could not load sentiment data: {e}")
    
    def _load_zero_shot_tags(self):
        """Load unified zero-shot tags."""
        try:
            # BUG FIX: Handle import path for both standalone and module execution
            try:
                from src.zero_shot_integration import load_unified_zero_shot_tags
            except ImportError:
                # Fallback for standalone execution
                from zero_shot_integration import load_unified_zero_shot_tags

            self.zero_shot_tags = load_unified_zero_shot_tags(self.data_dir)
            logger.info(f"✅ Loaded zero-shot tags: {len(self.zero_shot_tags.columns)-1} tags")
            self.tags_available = True
        except Exception as e:
            logger.warning(f"⚠️ Could not load zero-shot tags: {e}")
    
    def _print_availability_summary(self):
        """Print summary of available signals."""
        logger.info("\n" + "="*60)
        logger.info("SIGNAL AVAILABILITY SUMMARY")
        logger.info("="*60)
        logger.info(f"✅ CF (SVD/NeuMF):     {self.cf_available}")
        logger.info(f"✅ Content Similarity: {self.content_available}")
        logger.info(f"✅ Theme Matching:     {self.themes_available}")
        logger.info(f"✅ Sentiment Matching: {self.sentiments_available}")
        logger.info(f"✅ Zero-shot Tags:     {self.tags_available}")
        logger.info("="*60 + "\n")
    
    def compute_cf_score(self, user_id: Optional[int], movie_title: str) -> float:
        """
        Compute collaborative filtering score.

        Args:
            user_id: User ID (None if cold start)
            movie_title: Movie title

        Returns:
            CF score (0-5 scale, or 0 if unavailable)
        """
        if not self.cf_available:
            return 0.0

        # BUG FIX: For cold start (no user_id), return 0 to let other signals dominate
        # The scoring weights will handle cold start by boosting tag/query signals
        if user_id is None:
            return 0.0
        
        try:
            # Get movie internal ID
            if movie_title not in self.movie_to_idx:
                return 0.0
            
            movie_idx = self.movie_to_idx[movie_title]
            
            # Try SVD prediction
            if self.svd_model is not None:
                try:
                    pred = self.svd_model.predict(user_id, movie_idx)
                    return pred.est  # Surprise prediction object
                except Exception:
                    pass
            
            # Try NeuMF prediction
            if self.ncf_model is not None:
                try:
                    user_idx = self.user_to_idx.get(str(user_id), 0)
                    pred = self.ncf_model.predict(
                        [np.array([user_idx]), np.array([movie_idx])],
                        verbose=0
                    )
                    return float(pred[0][0]) * 5.0  # Scale to 0-5
                except Exception:
                    pass
            
            return 0.0
        except Exception as e:
            logger.debug(f"CF score error for {movie_title}: {e}")
            return 0.0
    
    def compute_content_score(self, query_movie: str, candidate_movie: str) -> float:
        """
        Compute content similarity score.

        Args:
            query_movie: Reference movie title
            candidate_movie: Candidate movie title

        Returns:
            Content similarity score (0-1)
        """
        if not self.content_available:
            return 0.0

        try:
            # BUG FIX: Use title_to_embedding_idx instead of movie_to_idx
            if query_movie not in self.title_to_embedding_idx or candidate_movie not in self.title_to_embedding_idx:
                return 0.0

            query_idx = self.title_to_embedding_idx[query_movie]
            cand_idx = self.title_to_embedding_idx[candidate_movie]

            # Check bounds (should always be valid now)
            if query_idx >= len(self.movie_embeddings) or cand_idx >= len(self.movie_embeddings):
                return 0.0

            # Compute cosine similarity
            query_emb = self.movie_embeddings[query_idx].reshape(1, -1)
            cand_emb = self.movie_embeddings[cand_idx].reshape(1, -1)

            sim = cosine_similarity(query_emb, cand_emb)[0][0]
            return float(max(0.0, sim))  # Ensure non-negative

        except Exception as e:
            logger.debug(f"Content score error: {e}")
            return 0.0
    
    def compute_theme_score(self, query_movie: str, candidate_movie: str) -> float:
        """
        Compute theme matching score using LDA topics.

        Args:
            query_movie: Reference movie title
            candidate_movie: Candidate movie title

        Returns:
            Theme similarity score (0-1)
        """
        if not self.themes_available:
            return 0.0

        try:
            # BUG FIX: Convert titles to Freebase IDs for theme lookup
            if isinstance(self.movie_themes, dict):
                # Try direct lookup first (for Freebase IDs)
                query_key = query_movie if query_movie in self.movie_themes else None
                cand_key = candidate_movie if candidate_movie in self.movie_themes else None

                # If not found, try title -> Freebase ID mapping
                if query_key is None and query_movie in self.title_to_freebase:
                    query_key = self.title_to_freebase[query_movie]

                if cand_key is None and candidate_movie in self.title_to_freebase:
                    cand_key = self.title_to_freebase[candidate_movie]

                # If still not found, no themes available for these movies
                if query_key is None or cand_key is None:
                    return 0.0

                if query_key not in self.movie_themes or cand_key not in self.movie_themes:
                    return 0.0

                query_topics = self.movie_themes[query_key]
                cand_topics = self.movie_themes[cand_key]

                # If topics are vectors, compute cosine similarity
                if isinstance(query_topics, (list, np.ndarray)):
                    query_vec = np.array(query_topics).reshape(1, -1)
                    cand_vec = np.array(cand_topics).reshape(1, -1)
                    sim = cosine_similarity(query_vec, cand_vec)[0][0]
                    return float(max(0.0, sim))

            # Handle DataFrame format
            else:
                # TODO: Implement DataFrame-based theme matching
                pass

            return 0.0
        except Exception as e:
            logger.debug(f"Theme score error: {e}")
            return 0.0
    
    def compute_sentiment_score(self, query_movie: str, candidate_movie: str) -> float:
        """
        Compute sentiment profile similarity.
        
        Args:
            query_movie: Reference movie title
            candidate_movie: Candidate movie title
            
        Returns:
            Sentiment similarity score (0-1)
        """
        if not self.sentiments_available:
            return 0.0
        
        try:
            # Get sentiment vectors
            sentiment_cols = [c for c in self.movie_sentiments.columns if c.startswith('sentiment_')]
            
            query_row = self.movie_sentiments[self.movie_sentiments['movie_title'] == query_movie]
            cand_row = self.movie_sentiments[self.movie_sentiments['movie_title'] == candidate_movie]
            
            if len(query_row) == 0 or len(cand_row) == 0:
                return 0.0
            
            query_vec = query_row[sentiment_cols].values.reshape(1, -1)
            cand_vec = cand_row[sentiment_cols].values.reshape(1, -1)
            
            sim = cosine_similarity(query_vec, cand_vec)[0][0]
            return float(max(0.0, sim))

        except Exception as e:
            logger.debug(f"Sentiment score error: {e}")
            return 0.0

    def compute_situation_outcome_score(self, candidate_movie: str,
                                        metadata: Optional[Dict],
                                        situation_outcome_data: Dict) -> tuple:
        """
        Compute query_score specifically for SITUATION+OUTCOME queries.

        Different from standard query scoring:
        - Entity nouns (girlfriend, dog) are SUPPLEMENTAL (0.20 each)
        - Situation phrases (broke up, died) are PRIMARY (0.40 each)
        - Outcome phrases (cheer up, feel good) are PRIMARY (0.40 each)
        - NO title-entity boost (entity is not prioritized)
        - Genre boost for outcome (+0.25)
        - TMDB all-match boost (+0.25)
        - Zero-shot all-match boost (+0.25) - if applicable

        Args:
            candidate_movie: Movie title
            metadata: TMDB metadata dict
            situation_outcome_data: Dict containing:
                - entity_nouns: List of supplemental entity nouns
                - situation_phrases: List of situation keywords/phrases
                - outcome_phrases: List of outcome keywords/phrases
                - outcome_genres: Set of genres mapped from outcome (e.g., {'comedy'})
                - situation_keywords: Expanded TMDB keywords for situation
                - outcome_tags: Expanded zero-shot tags for outcome

        Returns:
            Tuple of (query_score: float, tmdb_keyword_matches: int)
        """
        if not metadata:
            return (0.0, 0)

        try:
            # Extract data
            entity_nouns = situation_outcome_data.get('entity_nouns', [])
            situation_phrases = situation_outcome_data.get('situation_phrases', [])
            outcome_phrases = situation_outcome_data.get('outcome_phrases', [])
            outcome_genres = situation_outcome_data.get('outcome_genres', set())
            situation_keywords = situation_outcome_data.get('situation_keywords', [])
            outcome_tags = situation_outcome_data.get('outcome_tags', [])

            # Get movie data
            movie_keywords = metadata.get('keywords', [])
            movie_keywords_lower = [str(kw).lower() for kw in movie_keywords] if movie_keywords else []
            overview = (metadata.get('overview', '') or '').lower()
            movie_genres = metadata.get('genres', [])
            movie_genres_lower = [str(g).lower() for g in movie_genres] if movie_genres else []
            genre_ids = metadata.get('genre_ids', [])

            # Genre ID to name mapping
            GENRE_ID_TO_NAME = {
                28: 'action', 12: 'adventure', 16: 'animation', 35: 'comedy',
                80: 'crime', 99: 'documentary', 18: 'drama', 10751: 'family',
                14: 'fantasy', 36: 'history', 27: 'horror', 10402: 'music',
                9648: 'mystery', 10749: 'romance', 878: 'sci-fi', 10770: 'tv movie',
                53: 'thriller', 10752: 'war', 37: 'western'
            }

            # Convert genre_ids to names
            movie_genre_names = set()
            if isinstance(genre_ids, (list, np.ndarray)):
                for gid in genre_ids:
                    if gid in GENRE_ID_TO_NAME:
                        movie_genre_names.add(GENRE_ID_TO_NAME[gid])
            movie_genre_names.update(movie_genres_lower)

            score = 0.0
            tmdb_matches = 0

            # Calculate max possible score (dynamic based on query)
            entity_count = len(entity_nouns)
            situation_count = len(situation_phrases)
            outcome_count = len(outcome_phrases)

            entity_max = entity_count * 0.20
            situation_max = situation_count * 0.40
            outcome_max = outcome_count * 0.40
            base_max = entity_max + situation_max + outcome_max

            logger.debug(f"[SIT+OUT SCORE] {candidate_movie}: entity_nouns={entity_nouns}, situation={situation_phrases}, outcome={outcome_phrases}")
            logger.debug(f"[SIT+OUT SCORE] base_max={base_max:.2f} (entity:{entity_max:.2f} + sit:{situation_max:.2f} + out:{outcome_max:.2f})")

            # =================================================================
            # ENTITY NOUN MATCHING (0.20 each) - Supplemental
            # =================================================================
            entity_matches = 0
            for noun in entity_nouns:
                noun_lower = noun.lower()
                # Check keywords and overview
                if any(noun_lower in kw for kw in movie_keywords_lower) or noun_lower in overview:
                    entity_matches += 1
                    score += 0.20
                    tmdb_matches += 1
                    logger.debug(f"[SIT+OUT] Entity '{noun}' matched in {candidate_movie}")

            # =================================================================
            # SITUATION PHRASE MATCHING (0.40 each) - Primary
            # =================================================================
            situation_matches = 0
            # Check both the original phrases AND expanded keywords
            all_situation_terms = set(situation_phrases + situation_keywords)
            for phrase in situation_phrases:
                phrase_lower = phrase.lower()
                matched = False
                # Check if phrase or any expansion matches
                for term in [phrase_lower] + [sk.lower() for sk in situation_keywords]:
                    if any(term in kw for kw in movie_keywords_lower) or term in overview:
                        matched = True
                        break
                if matched:
                    situation_matches += 1
                    score += 0.40
                    tmdb_matches += 1
                    logger.debug(f"[SIT+OUT] Situation '{phrase}' matched in {candidate_movie}")
                    break  # Only count once per situation phrase

            # =================================================================
            # OUTCOME PHRASE MATCHING (0.40 each) - Primary
            # =================================================================
            outcome_matches = 0
            for phrase in outcome_phrases:
                phrase_lower = phrase.lower()
                matched = False
                # Check if phrase or any expansion matches
                for term in [phrase_lower] + [ot.lower() for ot in outcome_tags]:
                    if any(term in kw for kw in movie_keywords_lower) or term in overview:
                        matched = True
                        break
                if matched:
                    outcome_matches += 1
                    score += 0.40
                    tmdb_matches += 1
                    logger.debug(f"[SIT+OUT] Outcome '{phrase}' matched in {candidate_movie}")
                    break  # Only count once per outcome phrase

            # =================================================================
            # TMDB ALL-MATCH BOOST (+0.25) - if all three categories matched
            # =================================================================
            all_matched = (entity_matches > 0 or entity_count == 0) and \
                         (situation_matches > 0 or situation_count == 0) and \
                         (outcome_matches > 0 or outcome_count == 0)
            if all_matched and (entity_matches + situation_matches + outcome_matches) >= 2:
                score += 0.25
                logger.debug(f"[SIT+OUT] TMDB all-match boost +0.25 for {candidate_movie}")

            # =================================================================
            # GENRE BOOST (+0.25) - if movie has outcome genre
            # =================================================================
            genre_matched = False
            for outcome_genre in outcome_genres:
                if outcome_genre.lower() in movie_genre_names:
                    genre_matched = True
                    score += 0.25
                    logger.debug(f"[SIT+OUT] Genre boost +0.25 for '{outcome_genre}' in {candidate_movie}")
                    break  # Only one genre boost

            # =================================================================
            # ZERO-SHOT TAG BOOST (+0.25) - if movie has zero-shot tags matching outcome
            # =================================================================
            # Check if movie has matching zero-shot tags
            if self.zero_shot_tags is not None and outcome_tags:
                try:
                    movie_row = self.zero_shot_tags[self.zero_shot_tags['title_norm'] == candidate_movie.lower()]
                    if movie_row.empty:
                        normalized = candidate_movie.lower().replace(':', '').replace('-', ' ')
                        movie_row = self.zero_shot_tags[self.zero_shot_tags['title_norm'] == normalized]

                    if not movie_row.empty:
                        tag_cols = [col for col in self.zero_shot_tags.columns
                                   if col not in ['title_norm', 'unified_tags', 'unified_tags_v2']]
                        active_tags = [col for col in tag_cols if movie_row.iloc[0][col] == 1]

                        # Check for outcome tag matches
                        zs_matches = 0
                        for tag in outcome_tags:
                            if tag.lower() in active_tags:
                                zs_matches += 1

                        if zs_matches >= 2:
                            score += 0.25
                            logger.debug(f"[SIT+OUT] Zero-shot boost +0.25 for {zs_matches} tag matches in {candidate_movie}")
                except Exception as e:
                    logger.debug(f"Zero-shot check failed for {candidate_movie}: {e}")

            # NO TITLE-ENTITY BOOST for situation+outcome queries
            # (Entity is supplemental, not prioritized)

            # Normalize score - cap at reasonable max
            max_possible = base_max + 0.75  # base + all three bonuses
            if max_possible > 0:
                normalized_score = min(1.5, score)  # Allow some overflow for strong matches
            else:
                normalized_score = 0.0

            logger.debug(f"[SIT+OUT FINAL] {candidate_movie}: score={normalized_score:.3f}, tmdb_matches={tmdb_matches}")

            return (float(normalized_score), tmdb_matches)

        except Exception as e:
            logger.warning(f"Situation+outcome score error for '{candidate_movie}': {e}")
            return (0.0, 0)

    def compute_tag_score(self, query_tags: List[str], candidate_movie: str,
                         rare_tags: Optional[List[str]] = None) -> float:
        """
        Compute zero-shot tag matching score with rare tag boosting.

        Args:
            query_tags: List of query tags
            candidate_movie: Candidate movie title
            rare_tags: Optional list of rare tags (for boosting)

        Returns:
            Tag matching score (0-1)
        """
        if not self.tags_available or not query_tags:
            return 0.0

        try:
            # Find movie row - try title normalization variations
            movie_row = self.zero_shot_tags[self.zero_shot_tags['title_norm'] == candidate_movie.lower()]

            # BUG FIX: If not found, try without special characters
            if len(movie_row) == 0:
                # Remove apostrophes, colons, etc for better matching
                normalized = candidate_movie.lower().replace("'", "").replace(":", "").replace("-", " ")
                movie_row = self.zero_shot_tags[self.zero_shot_tags['title_norm'] == normalized]

            if len(movie_row) == 0:
                # Movie not in zero-shot tags (only 17k/43k movies have tags)
                # Return small score if any query tag appears in title as fallback
                movie_lower = candidate_movie.lower()
                for tag in query_tags:
                    if tag.lower() in movie_lower:
                        return 0.3  # Partial match based on title
                return 0.0
            
            # Count matching tags
            matches = 0
            boosted_matches = 0

            for tag in query_tags:
                if tag in movie_row.columns and movie_row[tag].values[0] > 0:
                    matches += 1

                    # Boost rare tags
                    if rare_tags and tag in rare_tags:
                        boosted_matches += 2  # Double weight for rare tags
                    else:
                        boosted_matches += 1

            if len(query_tags) == 0:
                return 0.0

            # FIX: Query expansion-friendly scoring
            # For expanded queries (many tags), use OR logic not AND logic
            # Normalize by a fixed denominator to avoid penalizing query expansion
            if matches == 0:
                return 0.0

            # Cap denominator at 3 tags for normalization
            # This way matching 1-2 tags gives good scores even with 10+ query tags
            normalizer = min(3, len(query_tags))
            score = boosted_matches / normalizer
            return float(min(1.0, score))
            
        except Exception as e:
            logger.debug(f"Tag score error: {e}")
            return 0.0
    
    def compute_concept_coverage_score(self, concept_groups: Dict[str, List[str]],
                                       candidate_movie: str,
                                       rare_tags: Optional[List[str]] = None) -> float:
        """
        Compute tag score based on concept coverage (AND logic, not OR).

        This FIXES the semantic expansion problem:
        - Query "historical revenge" has 2 concepts
        - Movie must match tags from BOTH concepts to get score = 1.0
        - Movie matching only 1 concept gets score = 0.5
        - Movie matching 0 concepts gets score = 0.0

        Args:
            concept_groups: Dict mapping concepts to their expanded tags
                Example: {
                    'historical': ['ancient rome', 'medieval', 'war film', ...],
                    'revenge': ['revenge', 'fighting', 'combat', ...]
                }
            candidate_movie: Candidate movie title
            rare_tags: Optional list of rare tags (for boosting within concepts)

        Returns:
            Concept coverage score (0-1) = (concepts matched) / (total concepts)
        """
        if not self.tags_available or not concept_groups:
            return 0.0

        try:
            # Find movie row
            movie_row = self.zero_shot_tags[self.zero_shot_tags['title_norm'] == candidate_movie.lower()]

            if len(movie_row) == 0:
                normalized = candidate_movie.lower().replace("'", "").replace(":", "").replace("-", " ")
                movie_row = self.zero_shot_tags[self.zero_shot_tags['title_norm'] == normalized]

            if len(movie_row) == 0:
                return 0.0

            # Use weighted coverage scoring based on number of tags matched per concept
            # This balances between being too permissive (1 tag) and too restrictive (2+ tags)
            total_concepts = len(concept_groups)
            matched_tags_by_concept = {}  # For debugging
            concept_scores = []  # Individual concept scores

            for concept_name, concept_tags in concept_groups.items():
                matched_tags = []

                for tag in concept_tags:
                    if tag in movie_row.columns and movie_row[tag].values[0] > 0:
                        matched_tags.append(tag)

                # Weighted scoring per concept:
                # 0 tags = 0.0
                # 1 tag  = 0.3 (weak match, reduces false positives)
                # 2 tags = 0.7 (moderate match)
                # 3+ tags = 1.0 (strong match)
                num_matched = len(matched_tags)
                if num_matched == 0:
                    concept_score = 0.0
                elif num_matched == 1:
                    concept_score = 0.3  # Weak evidence
                elif num_matched == 2:
                    concept_score = 0.7  # Moderate evidence
                else:
                    concept_score = 1.0  # Strong evidence

                concept_scores.append(concept_score)
                if num_matched > 0:
                    matched_tags_by_concept[concept_name] = matched_tags

            # Overall coverage is the average of individual concept scores
            if total_concepts == 0:
                return 0.0

            coverage_score = sum(concept_scores) / total_concepts

            # DEBUG: Log for specific movies to understand the problem
            if candidate_movie.lower() in ['finding nemo', 'gladiator', 'braveheart']:
                logger.info(f"\n[DEBUG TAG MATCHING] {candidate_movie}")
                logger.info(f"   Overall coverage score: {coverage_score:.2f}")
                for i, (concept_name, concept_tags) in enumerate(concept_groups.items()):
                    tags = matched_tags_by_concept.get(concept_name, [])
                    score = concept_scores[i]
                    logger.info(f"   - {concept_name}: {len(tags)} tags = {score:.2f} | {tags}")

            return float(coverage_score)

        except Exception as e:
            logger.debug(f"Concept coverage score error: {e}")
            return 0.0

    def compute_query_score(self, query_keywords: List[str], candidate_movie: str,
                           metadata: Optional[Dict] = None,
                           query_tags: Optional[List[str]] = None,
                           entity_nouns: Optional[List[str]] = None,
                           concept_to_expansions: Optional[Dict[str, List[str]]] = None) -> tuple:
        """
        Compute query relevance score using CONCEPT-BASED matching.

        NEW: Uses concept_to_expansions for 1-match-per-concept scoring.
        Each concept (e.g., "female", "thriller") gets at most 1 match,
        regardless of how many expansion synonyms hit.

        Args:
            query_keywords: List of keywords from query (all expansions)
            candidate_movie: Candidate movie title
            metadata: Optional TMDB metadata dict with 'keywords', 'overview', 'genres' fields
            query_tags: Optional list of query tags for zero-shot matching
            entity_nouns: Optional list of entity nouns for NOUN ENTITY BOOST
            concept_to_expansions: Maps concept -> list of expansions (for 1-match-per-concept scoring)

        Returns:
            Tuple of (query_score: float, tmdb_keyword_matches: int)
        """
        if not query_keywords:
            logger.debug(f"Query score: No keywords provided")
            return (0.0, 0)

        try:
            movie_lower = candidate_movie.lower()

            # Prepare movie metadata for matching
            movie_keywords_lower = []
            keywords_str = ""
            overview_lower = ""
            genres_str = ""

            # Genre ID to name mapping for concept matching
            GENRE_ID_TO_NAME = {
                28: 'action', 12: 'adventure', 16: 'animation', 35: 'comedy',
                80: 'crime', 99: 'documentary', 18: 'drama', 10751: 'family',
                14: 'fantasy', 36: 'history', 27: 'horror', 10402: 'music',
                9648: 'mystery', 10749: 'romance', 878: 'sci-fi', 10770: 'tv movie',
                53: 'thriller', 10752: 'war', 37: 'western'
            }

            # Concept to genre mapping (concepts that should match genre_ids)
            CONCEPT_TO_GENRE_IDS = {
                'thriller': [53],
                'thrillers': [53],
                'horror': [27],
                'comedy': [35],
                'comedies': [35],
                'drama': [18],
                'dramas': [18],
                'action': [28],
                'romance': [10749],
                'romantic': [10749],
                'sci-fi': [878],
                'science fiction': [878],
                'fantasy': [14],
                'mystery': [9648],
                'crime': [80],
                'war': [10752],
                'western': [37],
                'animation': [16],
                'animated': [16],
                'adventure': [12],
                'documentary': [99],
                'family': [10751],
                'history': [36],
                'historical': [36],
            }

            movie_genre_ids = []

            if metadata:
                movie_keywords = metadata.get('keywords', [])
                if isinstance(movie_keywords, (list, np.ndarray)) and len(movie_keywords) > 0:
                    movie_keywords_lower = [str(kw).lower() for kw in movie_keywords]
                    keywords_str = " ".join(movie_keywords_lower)

                overview = metadata.get('overview', '')
                if overview and isinstance(overview, str):
                    overview_lower = overview.lower()

                genres = metadata.get('genres', [])
                if isinstance(genres, (list, np.ndarray)) and len(genres) > 0:
                    genres_str = " ".join([str(g).lower() for g in genres])

                # Get genre_ids for genre-based concept matching
                genre_ids = metadata.get('genre_ids', [])
                if isinstance(genre_ids, (list, np.ndarray)) and len(genre_ids) > 0:
                    movie_genre_ids = [int(gid) for gid in genre_ids]

            # CONCEPT-BASED SCORING: Count concepts matched, not individual keywords
            # Each concept gets at most 1 match, regardless of how many expansions hit
            matched_concepts = set()  # Track which concepts are matched
            field_coverage = {}       # Track which fields each concept was found in

            if concept_to_expansions:
                # Use concept-based scoring (1 match per concept max)
                for concept, expansions in concept_to_expansions.items():
                    concept_found_in_fields = []

                    # FIRST: Check if concept matches via genre_id (strongest signal!)
                    # E.g., concept "thriller" matches genre_id 53
                    concept_lower = concept.lower()
                    if concept_lower in CONCEPT_TO_GENRE_IDS and movie_genre_ids:
                        expected_genre_ids = CONCEPT_TO_GENRE_IDS[concept_lower]
                        if any(gid in movie_genre_ids for gid in expected_genre_ids):
                            concept_found_in_fields.append('genre_id')
                            logger.debug(f"[GENRE MATCH] Concept '{concept}' matched via genre_id for '{candidate_movie}'")

                    # Check if ANY expansion of this concept matches in ANY field
                    for expansion in expansions:
                        exp_lower = expansion.lower()

                        # Also check expansion against genre_ids
                        if exp_lower in CONCEPT_TO_GENRE_IDS and movie_genre_ids:
                            expected_genre_ids = CONCEPT_TO_GENRE_IDS[exp_lower]
                            if any(gid in movie_genre_ids for gid in expected_genre_ids):
                                if 'genre_id' not in concept_found_in_fields:
                                    concept_found_in_fields.append('genre_id')

                        # Check title
                        if exp_lower in movie_lower:
                            concept_found_in_fields.append('title')

                        # Check TMDB keywords (substring match in any keyword)
                        if any(exp_lower in kw for kw in movie_keywords_lower):
                            concept_found_in_fields.append('tmdb_keywords')

                        # Check overview
                        if exp_lower in overview_lower:
                            concept_found_in_fields.append('overview')

                        # Check genres (string-based, legacy)
                        if exp_lower in genres_str:
                            concept_found_in_fields.append('genres')

                    # If concept was found in ANY field, count it (max 1 per concept)
                    if concept_found_in_fields:
                        matched_concepts.add(concept)
                        field_coverage[concept] = list(set(concept_found_in_fields))  # Dedupe fields

                # Count concepts matched in TMDB keywords
                tmdb_keyword_matches = sum(1 for concept, fields in field_coverage.items()
                                           if 'tmdb_keywords' in fields)

                # Total concepts and matched concepts for scoring
                total_concepts = len(concept_to_expansions)
                concepts_matched = len(matched_concepts)

                logger.debug(f"[CONCEPT SCORING] {candidate_movie}: {concepts_matched}/{total_concepts} concepts matched")

            else:
                # FALLBACK: Legacy keyword-based scoring (for backwards compatibility)
                matched_terms = set()

                for query_term in query_keywords:
                    term_lower = query_term.lower()
                    term_found_in_fields = []

                    # Check title
                    if term_lower in movie_lower:
                        matched_terms.add(query_term)
                        term_found_in_fields.append('title')

                    # Check TMDB keywords
                    if term_lower in keywords_str:
                        matched_terms.add(query_term)
                        term_found_in_fields.append('tmdb_keywords')

                    # Check overview
                    if term_lower in overview_lower:
                        matched_terms.add(query_term)
                        term_found_in_fields.append('overview')

                    # Check genres
                    if term_lower in genres_str:
                        matched_terms.add(query_term)
                        term_found_in_fields.append('genres')

                    if term_found_in_fields:
                        field_coverage[query_term] = term_found_in_fields

                tmdb_keyword_matches = sum(1 for term, fields in field_coverage.items()
                                           if 'tmdb_keywords' in fields)

                total_concepts = len(query_keywords)
                concepts_matched = len(matched_terms)
                matched_concepts = matched_terms

            # STEP 1: Base score from concept coverage
            # With concept-based scoring, we count CONCEPTS matched, not expansion keywords
            # This prevents movies from getting inflated scores by matching multiple synonyms

            if concepts_matched == 0:
                score = 0.0
            else:
                # Base score: proportion of concepts matched
                base_coverage = concepts_matched / total_concepts if total_concepts > 0 else 0.0

                # STEP 2: Bonus for perfect coverage (matching ALL concepts)
                # Matching ALL concepts is very important!
                perfect_coverage_bonus = 0.5 if concepts_matched == total_concepts else 0.0

                # STEP 3: Field quality bonus
                # Reward terms found in high-quality fields (genre_id > TMDB keywords > overview > title > genres)
                # FIX: Cap the NUMBER of TMDB keyword matches that count (max 4)
                # This prevents documentaries with many generic keywords from outranking thrillers with specific keywords
                field_quality_bonus = 0.0
                tmdb_match_count = 0  # Track TMDB keyword matches separately

                for term, fields in field_coverage.items():
                    if 'genre_id' in fields:
                        # BEST: Genre ID match - this is definitive (movie IS a thriller)
                        field_quality_bonus += 0.20
                    elif 'tmdb_keywords' in fields:
                        # Cap TMDB keyword matches at 4 (quality over quantity)
                        if tmdb_match_count < 4:
                            field_quality_bonus += 0.15  # Best: TMDB metadata
                            tmdb_match_count += 1
                    elif 'overview' in fields:
                        field_quality_bonus += 0.05  # Good: plot description
                    elif 'title' in fields:
                        field_quality_bonus += 0.03  # OK: title match
                    elif 'genres' in fields:
                        field_quality_bonus += 0.02  # Weak: genre match

                # Field quality bonus is now capped by design (max 4 TMDB matches × 0.15 = 0.60)

                # FINAL SCORE = base coverage + bonuses
                score = min(1.0, base_coverage + perfect_coverage_bonus + field_quality_bonus)

            # OPTION 1: TMDB Complete Coverage Boost
            # Reward movies that match ALL query keywords in TMDB (perfect TMDB coverage)
            # FIX: Changed from "any match" to "complete coverage" to reward quality over quantity
            exact_tmdb_matches = 0
            if metadata:
                movie_keywords = metadata.get('keywords', [])
                if isinstance(movie_keywords, (list, np.ndarray)) and len(movie_keywords) > 0:
                    # Check for exact phrase matches (not just substring)
                    movie_keywords_lower = [str(kw).lower() for kw in movie_keywords]
                    for query_phrase in query_keywords:
                        if query_phrase.lower() in movie_keywords_lower:
                            exact_tmdb_matches += 1

            # =================================================================
            # TMDB KEYWORD BOOST (PRIMARY): 0.75 max
            # =================================================================
            # TMDB has 43k movies - this is our primary matching signal
            # 0.15 per keyword match, capped at 5 matches = 0.75 max boost
            # Movies matching ALL query keywords skyrocket to the top
            if exact_tmdb_matches > 0:
                # 0.15 per match, max 5 matches = 0.75 max boost
                tmdb_boost = 0.15 * min(5, exact_tmdb_matches)
                score = min(1.0, score + tmdb_boost)
                logger.debug(f"[TMDB BOOST] +{tmdb_boost:.3f} for {exact_tmdb_matches} keyword matches (max 5)")

            # =================================================================
            # ZERO-SHOT TAG BOOST (SECONDARY): 0.25 max
            # =================================================================
            # Zero-shot has 17k movies - supplementary signal for movies with tags
            # 0.05 per tag match, capped at 5 matches = 0.25 max boost
            exact_zs_matches = 0

            if self.zero_shot_tags is not None and query_tags:
                try:
                    # Find movie in zero-shot data
                    movie_row = self.zero_shot_tags[self.zero_shot_tags['title_norm'] == candidate_movie.lower()]
                    if movie_row.empty:
                        # Try fuzzy match
                        normalized = candidate_movie.lower().replace(':', '').replace('-', ' ')
                        movie_row = self.zero_shot_tags[self.zero_shot_tags['title_norm'] == normalized]

                    if not movie_row.empty:
                        # Get tag columns (skip metadata columns)
                        tag_cols = [col for col in self.zero_shot_tags.columns
                                   if col not in ['title_norm', 'unified_tags', 'unified_tags_v2']]

                        # Get active tags for this movie
                        active_tags = [col for col in tag_cols if movie_row.iloc[0][col] == 1]

                        # Check for exact matches with query tags (NOT expanded keywords)
                        for query_tag in query_tags:
                            query_tag_lower = query_tag.lower()
                            if query_tag_lower in active_tags:
                                exact_zs_matches += 1
                except Exception as e:
                    logger.debug(f"Zero-shot exact match check failed: {e}")

            # Apply zero-shot boost: 0.05 per match, max 5 = 0.25 max
            if exact_zs_matches > 0:
                zs_boost = 0.05 * min(5, exact_zs_matches)
                score = min(1.0, score + zs_boost)
                logger.debug(f"[ZERO-SHOT BOOST] +{zs_boost:.3f} for {exact_zs_matches} tag matches (max 5)")

            # =================================================================
            # NOUN ENTITY BOOST: +0.30 if movie matches entity nouns
            # =================================================================
            # Entity nouns are the KEY concepts in the query (female, dog, lawyer, etc.)
            # Movies matching these MUST rank higher than movies that don't
            # This ensures "Gone Girl" (has 'woman', 'wife') beats "Scarface" (no female keywords)

            # NOUN VARIANTS: Map entity nouns to all related keywords
            # "female" should match: woman, girl, wife, mother, daughter, etc.
            NOUN_VARIANTS = {
                'female': ['female', 'woman', 'girl', 'wife', 'mother', 'daughter', 'sister', 'heroine', 'femme'],
                'male': ['male', 'man', 'boy', 'husband', 'father', 'son', 'brother', 'hero'],
                'dog': ['dog', 'puppy', 'canine', 'pet dog'],
                'cat': ['cat', 'kitten', 'feline', 'pet cat'],
                'lawyer': ['lawyer', 'attorney', 'legal', 'court', 'trial'],
                'doctor': ['doctor', 'physician', 'medical', 'hospital', 'nurse'],
                'cop': ['cop', 'police', 'detective', 'officer', 'law enforcement'],
                'soldier': ['soldier', 'military', 'army', 'war', 'combat'],
                'girlfriend': ['girlfriend', 'ex-girlfriend', 'fiancee', 'lover', 'partner'],
                'boyfriend': ['boyfriend', 'ex-boyfriend', 'fiance', 'lover', 'partner'],
                'wife': ['wife', 'woman', 'married', 'spouse'],
                'husband': ['husband', 'man', 'married', 'spouse'],
            }

            entity_noun_matches = 0
            if entity_nouns and metadata:
                movie_keywords = metadata.get('keywords', [])
                if isinstance(movie_keywords, (list, np.ndarray)) and len(movie_keywords) > 0:
                    movie_keywords_lower = [str(kw).lower() for kw in movie_keywords]
                    movie_keywords_str = ' '.join(movie_keywords_lower)
                    overview = metadata.get('overview', '') or ''
                    overview_lower = overview.lower()

                    for noun in entity_nouns:
                        noun_lower = noun.lower()
                        # Get variants for this noun (or just use the noun itself)
                        variants = NOUN_VARIANTS.get(noun_lower, [noun_lower])

                        # Check if ANY variant appears in keywords
                        for variant in variants:
                            if any(variant in kw for kw in movie_keywords_lower):
                                entity_noun_matches += 1
                                break  # Only count once per noun
                            # Also check overview
                            if variant in overview_lower:
                                entity_noun_matches += 1
                                break

            if entity_noun_matches > 0:
                # 0.10 per noun match, no cap - scales with number of entity nouns in query
                # Each entity noun concept can only match once (enforced by break in loop above)
                # Entity nouns are the most important concepts (female, dog, lawyer, etc.)
                noun_boost = 0.10 * entity_noun_matches
                score = min(1.5, score + noun_boost)  # Allow score to exceed 1.0 temporarily
                logger.debug(f"[NOUN ENTITY BOOST] +{noun_boost:.3f} for {entity_noun_matches} entity noun matches")

            # =================================================================
            # TITLE-ENTITY BOOST: +0.25 if movie TITLE contains entity word/variant
            # =================================================================
            # If the entity is literally IN THE TITLE, this is a massive signal
            # E.g., "Single White Female" contains "Female" → BIG BOOST
            # E.g., "War Dogs" contains "War" → BIG BOOST for war queries
            # E.g., "Anaconda" contains snake-related → BIG BOOST for snake queries
            # E.g., "The Boxer" contains "Boxer" → BIG BOOST for boxing queries
            # This should put these movies close to or in the top 10 regardless of TMDB keywords

            title_entity_boost = 0.0
            if entity_nouns:
                title_lower = candidate_movie.lower()
                # Split title into words for matching
                title_words = title_lower.replace('-', ' ').replace(':', ' ').replace("'", '').split()

                for noun in entity_nouns:
                    noun_lower = noun.lower()
                    # Get variants for this noun
                    variants = NOUN_VARIANTS.get(noun_lower, [noun_lower])

                    # Check if ANY variant is a word in the title
                    for variant in variants:
                        # Check both as whole word and substring for compound words
                        if variant in title_words or variant in title_lower:
                            title_entity_boost = 0.25  # Moderate boost - same as noun entity boost
                            logger.debug(f"[TITLE-ENTITY BOOST] +{title_entity_boost:.3f} for '{variant}' in title '{candidate_movie}'")
                            break
                    if title_entity_boost > 0:
                        break  # Only apply once

                if title_entity_boost > 0:
                    score = min(1.5, score + title_entity_boost)

            # DEBUG logging for all movies with matches
            if score > 0:
                matched_list = list(matched_concepts)
                field_info = {term: fields for term, fields in field_coverage.items()}
                logger.debug(f"Query score for '{candidate_movie}': {score:.3f} | Matched: {matched_list} | Fields: {field_info}")

            # Special debug for Argo
            if 'argo' in candidate_movie.lower():
                logger.info(f"\n[ARGO DEBUG] Query score: {score:.3f}")
                logger.info(f"[ARGO DEBUG] Matched concepts: {list(matched_concepts)} ({concepts_matched}/{total_concepts})")
                logger.info(f"[ARGO DEBUG] Field coverage: {dict(field_coverage)}")
                logger.info(f"[ARGO DEBUG] Exact TMDB matches: {exact_tmdb_matches}")
                logger.info(f"[ARGO DEBUG] Exact zero-shot matches: {exact_zs_matches}")
                logger.info(f"[ARGO DEBUG] TMDB keyword matches: {tmdb_keyword_matches}")

            return (float(score), int(tmdb_keyword_matches))

        except Exception as e:
            logger.warning(f"Query score error for '{candidate_movie}': {e}")
            return (0.0, 0)
    
    def compute_all_signals(self,
                           candidate_movies: List[str],
                           query_keywords: Optional[List[str]] = None,
                           query_tags: Optional[List[str]] = None,
                           reference_movie: Optional[str] = None,
                           user_id: Optional[int] = None,
                           rare_tags: Optional[List[str]] = None,
                           candidate_metadata: Optional[Dict] = None,
                           parsed_query: Optional[any] = None,
                           concept_groups: Optional[Dict[str, List[str]]] = None,
                           entity_nouns: Optional[List[str]] = None,
                           concept_to_expansions: Optional[Dict[str, List[str]]] = None,
                           situation_outcome_data: Optional[Dict] = None) -> List[SignalScores]:
        """
        Compute all 6 signals for each candidate movie.

        Args:
            candidate_movies: List of candidate movie titles
            query_keywords: Optional query keywords
            query_tags: Optional query tags for tag matching
            reference_movie: Optional reference movie for similarity
            user_id: Optional user ID for CF
            rare_tags: Optional list of rare tags for boosting
            entity_nouns: Optional list of entity nouns for NOUN ENTITY BOOST
            concept_to_expansions: Maps concept -> expansions for concept-based scoring (1 match per concept max)

        Returns:
            List of SignalScores objects
        """
        # DEBUG: Log what we're working with
        logger.info(f"   [SignalFusion] Computing signals for {len(candidate_movies)} movies")
        logger.info(f"   [SignalFusion] query_keywords: {query_keywords}")
        logger.info(f"   [SignalFusion] query_tags: {query_tags}")

        results = []
        
        for movie in candidate_movies:
            scores = SignalScores(movie_title=movie)
            metadata = candidate_metadata.get(movie, {}) if candidate_metadata else {}

            # Signal 1: CF (or fallback to weighted_rating / vote_average)
            if user_id is not None:
                scores.cf_score = self.compute_cf_score(user_id, movie)
            elif metadata:
                # FALLBACK: Use Bayesian weighted_rating as pseudo-CF score (prevents low-vote bias)
                weighted_rating = metadata.get('weighted_rating', 0)
                if weighted_rating > 0:
                    scores.cf_score = float(weighted_rating) / 2.0  # Scale to 0-5
                elif metadata.get('vote_average', 0) > 0:
                    # Second fallback: raw vote_average if weighted_rating not available
                    scores.cf_score = float(metadata['vote_average']) / 2.0  # Scale to 0-5

            # Signal 2: Content similarity (STRENGTHENED fallback)
            if reference_movie is not None:
                scores.content_score = self.compute_content_score(reference_movie, movie)
            elif metadata and query_keywords:
                # STRENGTHENED FALLBACK: Multi-field keyword matching for content similarity
                # This replaces the need for embedding similarity when no reference movie exists
                total_matches = 0

                # Check TMDB keywords (high weight)
                movie_keywords = metadata.get('keywords', [])
                if isinstance(movie_keywords, list) and movie_keywords:
                    keywords_str = " ".join([str(kw).lower() for kw in movie_keywords])
                    keyword_matches = sum(1 for kw in query_keywords
                                         if kw.lower() in keywords_str)
                    total_matches += keyword_matches * 2.0  # High weight

                # Check overview (medium weight)
                overview = metadata.get('overview', '')
                if overview and isinstance(overview, str):
                    overview_matches = sum(1 for kw in query_keywords
                                          if kw.lower() in overview.lower())
                    total_matches += overview_matches * 1.0  # Medium weight

                # Check genres (low weight)
                genres = metadata.get('genres', [])
                if isinstance(genres, list) and genres:
                    genres_str = " ".join([str(g).lower() for g in genres])
                    genre_matches = sum(1 for kw in query_keywords
                                       if kw.lower() in genres_str)
                    total_matches += genre_matches * 0.5  # Low weight

                # Normalize (max score per keyword: 3.5)
                if len(query_keywords) > 0:
                    scores.content_score = min(1.0, total_matches / (len(query_keywords) * 3.5))

            # Signal 3: Theme matching (STRENGTHENED fallback)
            if reference_movie is not None:
                scores.theme_score = self.compute_theme_score(reference_movie, movie)
            elif metadata and query_tags:
                # STRENGTHENED FALLBACK: Multi-field thematic matching
                # Uses TMDB metadata to approximate LDA topic similarity
                total_matches = 0

                # Check TMDB keywords for thematic concepts (highest weight)
                movie_keywords = metadata.get('keywords', [])
                if isinstance(movie_keywords, list) and movie_keywords:
                    keywords_str = " ".join([str(kw).lower() for kw in movie_keywords])
                    theme_matches = sum(1 for tag in query_tags
                                       if tag.lower() in keywords_str)
                    total_matches += theme_matches * 2.5  # Highest weight for themes

                # Check overview for thematic words (medium weight)
                overview = metadata.get('overview', '')
                if overview and isinstance(overview, str):
                    overview_matches = sum(1 for tag in query_tags
                                          if tag.lower() in overview.lower())
                    total_matches += overview_matches * 1.5  # Medium weight

                # Check tagline (if available) for thematic resonance
                tagline = metadata.get('tagline', '')
                if tagline and isinstance(tagline, str):
                    tagline_matches = sum(1 for tag in query_tags
                                         if tag.lower() in tagline.lower())
                    total_matches += tagline_matches * 1.0  # Taglines are concise but thematic

                # Normalize (max score per tag: 5.0)
                if len(query_tags) > 0:
                    scores.theme_score = min(1.0, total_matches / (len(query_tags) * 5.0))

            # Signal 4: Sentiment matching (or fallback to cast/actor matching)
            if reference_movie is not None:
                scores.sentiment_score = self.compute_sentiment_score(reference_movie, movie)
            elif query_keywords and metadata:
                # FALLBACK: Check if query mentions actors (e.g., "bill murray movies")
                # Extract potential actor names from query
                query_text = " ".join(query_keywords)
                movie_cast = metadata.get('cast', [])

                # Handle numpy arrays and lists
                if movie_cast is not None and len(movie_cast) > 0:
                    # Convert to string and check for matches
                    cast_str = " ".join([str(actor).lower() for actor in movie_cast])
                    query_words = query_text.lower().split()

                    # Check for name matches (2+ consecutive words)
                    if len(query_words) >= 2:
                        for i in range(len(query_words) - 1):
                            potential_name = f"{query_words[i]} {query_words[i+1]}"
                            if potential_name in cast_str:
                                scores.sentiment_score = 1.0  # Perfect match for actor
                                logger.debug(f"Actor match: '{potential_name}' found in {movie}")
                                break

            # Signal 5: Tag matching (with concept coverage if available)
            if concept_groups:
                # Use concept coverage scoring (AND logic, not OR)
                scores.tag_score = self.compute_concept_coverage_score(concept_groups, movie, rare_tags)
            elif query_tags:
                # Fallback to standard tag matching
                scores.tag_score = self.compute_tag_score(query_tags, movie, rare_tags)

            # Signal 6: Query relevance (STRENGTHENED with TMDB metadata)
            # Use situation+outcome scoring if data provided, otherwise standard scoring
            if situation_outcome_data:
                scores.query_score, scores.tmdb_keyword_matches = self.compute_situation_outcome_score(
                    movie, metadata, situation_outcome_data
                )
            elif query_keywords:
                scores.query_score, scores.tmdb_keyword_matches = self.compute_query_score(
                    query_keywords, movie, metadata, query_tags, entity_nouns, concept_to_expansions
                )

            results.append(scores)
        
        return results


# Convenience function
def compute_signals(candidate_movies: List[str], 
                   query_keywords: Optional[List[str]] = None,
                   **kwargs) -> List[Dict]:
    """
    Convenience function to compute signals.
    
    Args:
        candidate_movies: List of candidate movie titles
        query_keywords: Optional query keywords
        **kwargs: Additional arguments for compute_all_signals
        
    Returns:
        List of score dictionaries
    """
    fusion = SignalFusion()
    results = fusion.compute_all_signals(
        candidate_movies=candidate_movies,
        query_keywords=query_keywords,
        **kwargs
    )
    return [r.to_dict() for r in results]


# Testing code
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING SIGNAL FUSION")
    print("="*60 + "\n")
    
    # Initialize
    print("Initializing signal fusion...")
    fusion = SignalFusion()
    
    # Test with sample movies
    test_movies = ["Toy Story", "The Matrix", "Inception"]
    
    print(f"\nComputing signals for test movies: {test_movies}")
    print("-" * 60)
    
    results = fusion.compute_all_signals(
        candidate_movies=test_movies,
        query_keywords=["action", "adventure"],
        query_tags=["action", "sci-fi"],
        reference_movie="Toy Story",
        user_id=1
    )
    
    print("\nResults:")
    for result in results:
        print(f"\n{result.movie_title}:")
        print(f"  CF Score:        {result.cf_score:.3f}")
        print(f"  Content Score:   {result.content_score:.3f}")
        print(f"  Theme Score:     {result.theme_score:.3f}")
        print(f"  Sentiment Score: {result.sentiment_score:.3f}")
        print(f"  Tag Score:       {result.tag_score:.3f}")
        print(f"  Query Score:     {result.query_score:.3f}")
    
    print("\n" + "="*60)
    print("TESTING COMPLETE! ✅")
    print("="*60)
    print("\n💡 Usage in other modules:")
    print("  from src.signal_fusion import SignalFusion")
    print("  fusion = SignalFusion()")
    print("  signals = fusion.compute_all_signals(candidates, ...)")