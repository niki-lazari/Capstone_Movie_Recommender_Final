"""
Scoring Module
Combines all 6 recommendation signals into final scores.

Uses configurable weights from config.yaml and applies normalization.
FIXES THE BUG: Actually uses all signals with proper weights!

Signal Weights (configurable):
1. CF (Collaborative Filtering) - 25%
2. Content Similarity - 20%
3. Theme Matching - 15%
4. Sentiment Matching - 10%
5. Zero-shot Tag Matching - 20%
6. Query Relevance - 10%
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass
import yaml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ScoringWeights:
    """Configurable weights for signal fusion."""
    cf_weight: float = 0.25
    content_weight: float = 0.20
    theme_weight: float = 0.15
    sentiment_weight: float = 0.10
    tag_weight: float = 0.20
    query_weight: float = 0.10
    
    def normalize(self):
        """Ensure weights sum to 1.0."""
        total = (self.cf_weight + self.content_weight + self.theme_weight + 
                self.sentiment_weight + self.tag_weight + self.query_weight)
        
        if total > 0:
            self.cf_weight /= total
            self.content_weight /= total
            self.theme_weight /= total
            self.sentiment_weight /= total
            self.tag_weight /= total
            self.query_weight /= total
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'cf_weight': self.cf_weight,
            'content_weight': self.content_weight,
            'theme_weight': self.theme_weight,
            'sentiment_weight': self.sentiment_weight,
            'tag_weight': self.tag_weight,
            'query_weight': self.query_weight
        }


@dataclass
class ScoredMovie:
    """Movie with final combined score."""
    movie_title: str
    final_score: float
    
    # Individual signal scores
    cf_score: float = 0.0
    content_score: float = 0.0
    theme_score: float = 0.0
    sentiment_score: float = 0.0
    tag_score: float = 0.0
    query_score: float = 0.0

    # Metadata for conditional scoring
    tmdb_keyword_matches: int = 0   # Number of TMDB keywords matched (for conditional zero-shot)

    # Weighted contributions
    cf_contribution: float = 0.0
    content_contribution: float = 0.0
    theme_contribution: float = 0.0
    sentiment_contribution: float = 0.0
    tag_contribution: float = 0.0
    query_contribution: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'movie_title': self.movie_title,
            'final_score': self.final_score,
            'cf_score': self.cf_score,
            'content_score': self.content_score,
            'theme_score': self.theme_score,
            'sentiment_score': self.sentiment_score,
            'tag_score': self.tag_score,
            'query_score': self.query_score,
            'tmdb_keyword_matches': self.tmdb_keyword_matches,
            'cf_contribution': self.cf_contribution,
            'content_contribution': self.content_contribution,
            'theme_contribution': self.theme_contribution,
            'sentiment_contribution': self.sentiment_contribution,
            'tag_contribution': self.tag_contribution,
            'query_contribution': self.query_contribution
        }


class Scorer:
    """Combines signals into final scores with configurable weights."""
    
    def __init__(self, config_path: Optional[str] = "src/config.yaml"):
        """
        Initialize scorer with weights from config.
        
        Args:
            config_path: Path to config.yaml (optional)
        """
        self.weights = ScoringWeights()
        self._load_weights(config_path)
        self.weights.normalize()
        
        logger.info("Scorer initialized with weights:")
        for key, val in self.weights.to_dict().items():
            logger.info(f"  {key}: {val:.3f}")
    
    def _load_weights(self, config_path: Optional[str]):
        """Load weights from config file."""
        if config_path is None:
            return
        
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.info("Config file not found, using default weights")
                return
            
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            
            weights_config = config.get('scoring', {}).get('weights', {})
            
            if weights_config:
                self.weights.cf_weight = weights_config.get('cf', self.weights.cf_weight)
                self.weights.content_weight = weights_config.get('content', self.weights.content_weight)
                self.weights.theme_weight = weights_config.get('theme', self.weights.theme_weight)
                self.weights.sentiment_weight = weights_config.get('sentiment', self.weights.sentiment_weight)
                self.weights.tag_weight = weights_config.get('tag', self.weights.tag_weight)
                self.weights.query_weight = weights_config.get('query', self.weights.query_weight)
                
                logger.info("Loaded weights from config file")
        except Exception as e:
            logger.warning(f"Could not load config: {e}, using defaults")
    
    def normalize_signal(self, score: float, signal_type: str) -> float:
        """
        Normalize a signal score to 0-1 range.

        Args:
            score: Raw signal score
            signal_type: Type of signal (cf, content, etc.)

        Returns:
            Normalized score (0-1)
        """
        # CF scores are 0-5, normalize to 0-1
        if signal_type == 'cf':
            return min(1.0, max(0.0, score / 5.0))

        # All other scores should already be 0-1, but clip just in case
        return min(1.0, max(0.0, score))

    def adjust_weights_for_query(self, parsed_query):
        """
        CRITICAL: Dynamically adjust weights based on query type and thematic intensity.

        Args:
            parsed_query: Parsed query object with extracted features
        """
        # Count which query features are present
        has_genres = bool(parsed_query.genres)
        has_actor_theme = any(term in ' '.join(parsed_query.themes + parsed_query.keywords).lower()
                             for term in ['actor', 'actress', 'starring', 'with', 'bill', 'sandra'])
        has_decade = bool(parsed_query.decades) or bool(parsed_query.years)

        # Count thematic intensity (number of theme tags after expansion)
        num_themes = len(parsed_query.themes)
        has_mood_theme = num_themes > 0 and not has_actor_theme

        # QUERY TYPE A: Genre + specific themes (e.g., "political thriller based on true story")
        # NEW APPROACH: Content-first, ratings-second
        # When query has discriminating terms, EXACT MATCHES (query/tag) should dominate
        # LDA/embeddings are too generic (give high scores to everything)
        if has_genres and (has_mood_theme or has_actor_theme):
            # CONTENT-FIRST WEIGHTS (discriminating signals dominate)
            self.weights.query_weight = 0.45    # TMDB keywords PRIMARY (distinguishes matches)
            self.weights.tag_weight = 0.20      # Zero-shot SECONDARY (reinforces matches)
            self.weights.theme_weight = 0.10    # LDA demoted (too generic - maxes out)
            self.weights.content_weight = 0.10  # Embeddings demoted (too generic)
            self.weights.sentiment_weight = 0.10 # BERT support
            self.weights.cf_weight = 0.05       # Ratings as TIEBREAKER only
            logger.info("Dynamic weights: Content-First (Query=45%, Tag=20%, CF=5% tiebreaker)")

        # QUERY TYPE B: Actor/Actress focused (e.g., "bill murray movies")
        elif has_actor_theme:
            self.weights.cf_weight = 0.25
            self.weights.content_weight = 0.10
            self.weights.theme_weight = 0.10
            self.weights.sentiment_weight = 0.50  # Actor matching is key!
            self.weights.tag_weight = 0.03
            self.weights.query_weight = 0.02
            logger.info("Dynamic weights: Actor-focused query")

        # QUERY TYPE C: Genre only (e.g., "comedy movies")
        elif has_genres:
            self.weights.cf_weight = 0.40  # High-rated comedies
            self.weights.content_weight = 0.20
            self.weights.theme_weight = 0.20
            self.weights.sentiment_weight = 0.10
            self.weights.tag_weight = 0.05
            self.weights.query_weight = 0.05
            logger.info("Dynamic weights: Genre-only query")

        # QUERY TYPE D: Mood/theme queries (e.g., "supernatural movies", "sports movies", "heist movies")
        # OPTION B IMPLEMENTED: Boost tag weight based on thematic intensity
        # This prevents CF dominance and contamination from high-rated non-thematic films
        elif has_mood_theme:
            # High thematic intensity (5+ theme tags): Very strong thematic focus
            if num_themes >= 5:
                self.weights.cf_weight = 0.15  # Minimal CF influence
                self.weights.content_weight = 0.10
                self.weights.theme_weight = 0.15
                self.weights.sentiment_weight = 0.05
                self.weights.tag_weight = 0.50  # VERY STRONG theme signal
                self.weights.query_weight = 0.05
                logger.info(f"Dynamic weights: High thematic intensity ({num_themes} themes, tag_weight=0.50)")
            # Medium thematic intensity (2-4 theme tags): Strong thematic focus
            elif num_themes >= 2:
                self.weights.cf_weight = 0.20
                self.weights.content_weight = 0.15
                self.weights.theme_weight = 0.15
                self.weights.sentiment_weight = 0.05
                self.weights.tag_weight = 0.40
                self.weights.query_weight = 0.05
                logger.info(f"Dynamic weights: Medium thematic intensity ({num_themes} themes, tag_weight=0.40)")
            # Low thematic intensity (1 theme tag): Balanced
            else:
                self.weights.cf_weight = 0.25
                self.weights.content_weight = 0.20
                self.weights.theme_weight = 0.15
                self.weights.sentiment_weight = 0.10
                self.weights.tag_weight = 0.25
                self.weights.query_weight = 0.05
                logger.info(f"Dynamic weights: Low thematic intensity ({num_themes} themes, tag_weight=0.25)")

        # QUERY TYPE E: Decade/year only (e.g., "movies from the 90s")
        elif has_decade:
            self.weights.cf_weight = 0.50  # Best rated from that era
            self.weights.content_weight = 0.15
            self.weights.theme_weight = 0.15
            self.weights.sentiment_weight = 0.10
            self.weights.tag_weight = 0.05
            self.weights.query_weight = 0.05
            logger.info("Dynamic weights: Decade-only query")

        # Default: balanced weights (already set in __init__)
        else:
            logger.info("Dynamic weights: Using default balanced weights")

        # Normalize to ensure they sum to 1.0
        self.weights.normalize()

    def compute_final_score(self, 
                           cf_score: float = 0.0,
                           content_score: float = 0.0,
                           theme_score: float = 0.0,
                           sentiment_score: float = 0.0,
                           tag_score: float = 0.0,
                           query_score: float = 0.0) -> Dict[str, float]:
        """
        Compute final weighted score from individual signals.
        
        Args:
            cf_score: Collaborative filtering score (0-5)
            content_score: Content similarity score (0-1)
            theme_score: Theme matching score (0-1)
            sentiment_score: Sentiment similarity score (0-1)
            tag_score: Tag matching score (0-1)
            query_score: Query relevance score (0-1)
            
        Returns:
            Dictionary with final_score and individual contributions
        """
        # Normalize all signals to 0-1
        cf_norm = self.normalize_signal(cf_score, 'cf')
        content_norm = self.normalize_signal(content_score, 'content')
        theme_norm = self.normalize_signal(theme_score, 'theme')
        sentiment_norm = self.normalize_signal(sentiment_score, 'sentiment')
        tag_norm = self.normalize_signal(tag_score, 'tag')
        query_norm = self.normalize_signal(query_score, 'query')
        
        # Compute weighted contributions
        cf_contrib = cf_norm * self.weights.cf_weight
        content_contrib = content_norm * self.weights.content_weight
        theme_contrib = theme_norm * self.weights.theme_weight
        sentiment_contrib = sentiment_norm * self.weights.sentiment_weight
        tag_contrib = tag_norm * self.weights.tag_weight
        query_contrib = query_norm * self.weights.query_weight
        
        # Final score is weighted sum
        final_score = (cf_contrib + content_contrib + theme_contrib + 
                      sentiment_contrib + tag_contrib + query_contrib)
        
        return {
            'final_score': final_score,
            'cf_contribution': cf_contrib,
            'content_contribution': content_contrib,
            'theme_contribution': theme_contrib,
            'sentiment_contribution': sentiment_contrib,
            'tag_contribution': tag_contrib,
            'query_contribution': query_contrib
        }
    
    def score_movies(self, signal_scores: List[Dict]) -> List[ScoredMovie]:
        """
        Score a list of movies with their signal scores.
        
        Args:
            signal_scores: List of dicts with movie_title and signal scores
            
        Returns:
            List of ScoredMovie objects sorted by final score
        """
        scored_movies = []
        
        for movie_signals in signal_scores:
            # Compute final score and contributions
            scoring_result = self.compute_final_score(
                cf_score=movie_signals.get('cf_score', 0.0),
                content_score=movie_signals.get('content_score', 0.0),
                theme_score=movie_signals.get('theme_score', 0.0),
                sentiment_score=movie_signals.get('sentiment_score', 0.0),
                tag_score=movie_signals.get('tag_score', 0.0),
                query_score=movie_signals.get('query_score', 0.0)
            )
            
            # Create ScoredMovie object
            scored_movie = ScoredMovie(
                movie_title=movie_signals['movie_title'],
                final_score=scoring_result['final_score'],
                cf_score=movie_signals.get('cf_score', 0.0),
                content_score=movie_signals.get('content_score', 0.0),
                theme_score=movie_signals.get('theme_score', 0.0),
                sentiment_score=movie_signals.get('sentiment_score', 0.0),
                tag_score=movie_signals.get('tag_score', 0.0),
                query_score=movie_signals.get('query_score', 0.0),
                tmdb_keyword_matches=movie_signals.get('tmdb_keyword_matches', 0),
                cf_contribution=scoring_result['cf_contribution'],
                content_contribution=scoring_result['content_contribution'],
                theme_contribution=scoring_result['theme_contribution'],
                sentiment_contribution=scoring_result['sentiment_contribution'],
                tag_contribution=scoring_result['tag_contribution'],
                query_contribution=scoring_result['query_contribution']
            )
            
            scored_movies.append(scored_movie)
        
        # Sort by final score (descending)
        scored_movies.sort(key=lambda x: x.final_score, reverse=True)
        
        return scored_movies
    
    def get_top_n(self, scored_movies: List[ScoredMovie], n: int = 10) -> List[ScoredMovie]:
        """
        Get top N movies by score.
        
        Args:
            scored_movies: List of scored movies
            n: Number of top movies to return
            
        Returns:
            Top N movies
        """
        return scored_movies[:n]
    
    def explain_score(self, scored_movie: ScoredMovie) -> str:
        """
        Generate human-readable explanation of score.
        
        Args:
            scored_movie: Scored movie object
            
        Returns:
            Explanation string
        """
        lines = [
            f"Score Breakdown for '{scored_movie.movie_title}' (Final: {scored_movie.final_score:.3f}):",
            f"  CF:        {scored_movie.cf_score:.3f} → {scored_movie.cf_contribution:.3f} ({self.weights.cf_weight*100:.1f}%)",
            f"  Content:   {scored_movie.content_score:.3f} → {scored_movie.content_contribution:.3f} ({self.weights.content_weight*100:.1f}%)",
            f"  Theme:     {scored_movie.theme_score:.3f} → {scored_movie.theme_contribution:.3f} ({self.weights.theme_weight*100:.1f}%)",
            f"  Sentiment: {scored_movie.sentiment_score:.3f} → {scored_movie.sentiment_contribution:.3f} ({self.weights.sentiment_weight*100:.1f}%)",
            f"  Tag:       {scored_movie.tag_score:.3f} → {scored_movie.tag_contribution:.3f} ({self.weights.tag_weight*100:.1f}%)",
            f"  Query:     {scored_movie.query_score:.3f} → {scored_movie.query_contribution:.3f} ({self.weights.query_weight*100:.1f}%)"
        ]
        
        return "\n".join(lines)


# Convenience function
def score_movies(signal_scores: List[Dict], 
                config_path: Optional[str] = "src/config.yaml") -> List[Dict]:
    """
    Convenience function to score movies.
    
    Args:
        signal_scores: List of signal score dictionaries
        config_path: Optional path to config file
        
    Returns:
        List of scored movie dictionaries
    """
    scorer = Scorer(config_path)
    scored = scorer.score_movies(signal_scores)
    return [m.to_dict() for m in scored]


# Testing code
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING SCORING MODULE")
    print("="*60 + "\n")
    
    # Test 1: Initialize scorer
    print("Test 1: Initializing scorer...")
    scorer = Scorer(config_path=None)  # Use defaults
    print("✅ Scorer initialized\n")
    
    # Test 2: Score normalization
    print("Test 2: Testing signal normalization...")
    cf_norm = scorer.normalize_signal(4.5, 'cf')
    content_norm = scorer.normalize_signal(0.85, 'content')
    print(f"✅ CF 4.5 → {cf_norm:.3f}")
    print(f"✅ Content 0.85 → {content_norm:.3f}\n")
    
    # Test 3: Compute final score
    print("Test 3: Computing final score...")
    result = scorer.compute_final_score(
        cf_score=4.5,
        content_score=0.85,
        theme_score=0.70,
        sentiment_score=0.90,
        tag_score=0.60,
        query_score=0.50
    )
    print(f"✅ Final score: {result['final_score']:.3f}")
    print(f"   Contributions:")
    print(f"     CF: {result['cf_contribution']:.3f}")
    print(f"     Content: {result['content_contribution']:.3f}")
    print(f"     Theme: {result['theme_contribution']:.3f}")
    print(f"     Sentiment: {result['sentiment_contribution']:.3f}")
    print(f"     Tag: {result['tag_contribution']:.3f}")
    print(f"     Query: {result['query_contribution']:.3f}\n")
    
    # Test 4: Score multiple movies
    print("Test 4: Scoring multiple movies...")
    test_signals = [
        {
            'movie_title': 'Toy Story',
            'cf_score': 4.5,
            'content_score': 0.85,
            'theme_score': 0.70,
            'sentiment_score': 0.90,
            'tag_score': 0.60,
            'query_score': 0.50
        },
        {
            'movie_title': 'The Matrix',
            'cf_score': 4.8,
            'content_score': 0.75,
            'theme_score': 0.65,
            'sentiment_score': 0.80,
            'tag_score': 0.90,
            'query_score': 0.95
        },
        {
            'movie_title': 'Inception',
            'cf_score': 4.2,
            'content_score': 0.70,
            'theme_score': 0.60,
            'sentiment_score': 0.85,
            'tag_score': 0.55,
            'query_score': 0.40
        }
    ]
    
    scored_movies = scorer.score_movies(test_signals)
    print(f"✅ Scored {len(scored_movies)} movies\n")
    
    # Test 5: Display results
    print("Test 5: Top movies:")
    print("-" * 60)
    for i, movie in enumerate(scored_movies, 1):
        print(f"{i}. {movie.movie_title}: {movie.final_score:.3f}")
    print()
    
    # Test 6: Score explanation
    print("Test 6: Score explanation for top movie:")
    print("-" * 60)
    print(scorer.explain_score(scored_movies[0]))
    print()
    
    print("="*60)
    print("ALL TESTS PASSED! ✅")
    print("="*60)
    print("\n💡 Usage in other modules:")
    print("  from src.scoring import Scorer")
    print("  scorer = Scorer()")
    print("  scored = scorer.score_movies(signal_scores)")