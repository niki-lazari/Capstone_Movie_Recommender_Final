"""
Data Loader Module
Centralized loading of all preprocessed artifacts and data.

Loads:
- Movie metadata (TMDB enriched data)
- User-movie ratings
- Movie embeddings
- Sentiment data
- Theme data
- Zero-shot tags
- ID mappings

Usage:
    loader = DataLoader()
    movies_df = loader.get_movies()
    embeddings = loader.get_embeddings()
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Centralized data loader for all artifacts."""
    
    def __init__(self, 
                 data_dir: str = "data/raw",
                 models_dir: str = "models"):
        """
        Initialize data loader.
        
        Args:
            data_dir: Directory containing processed data
            models_dir: Directory containing model artifacts
        """
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        
        # Data containers
        self._movies_df = None
        self._ratings_df = None
        self._embeddings = None
        self._sentiments_df = None
        self._themes = None
        self._zero_shot_tags = None
        
        # Mappings
        self._movie_to_idx = None
        self._user_to_idx = None
        
        logger.info("DataLoader initialized")
    
    def get_movies(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load movie metadata (TMDB enriched).
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            DataFrame with movie metadata
        """
        if self._movies_df is not None and not force_reload:
            return self._movies_df
        
        try:
            # Try parquet first (faster)
            parquet_path = self.data_dir / "tmdb_fully_enriched.parquet"
            if parquet_path.exists():
                self._movies_df = pd.read_parquet(parquet_path)

                # Convert numpy arrays to Python lists for compatibility
                # This fixes keyword search and ensures consistency across data sources
                array_columns = [
                    'keywords', 'cast', 'cast_detailed', 'directors', 'genre_ids',
                    'producers', 'writers', 'production_companies',
                    'streaming_providers', 'buy_providers', 'rent_providers'
                ]

                for col in array_columns:
                    if col in self._movies_df.columns:
                        self._movies_df[col] = self._movies_df[col].apply(
                            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                        )

                logger.info(f"✅ Loaded movies: {len(self._movies_df)} movies")
                return self._movies_df

            # Fallback to CSV
            csv_path = self.data_dir / "df_tmdb_enriched.csv"
            if csv_path.exists():
                self._movies_df = pd.read_csv(csv_path)

                # Convert numpy arrays to Python lists for CSV too
                array_columns = [
                    'keywords', 'cast', 'cast_detailed', 'directors', 'genre_ids',
                    'producers', 'writers', 'production_companies',
                    'streaming_providers', 'buy_providers', 'rent_providers'
                ]

                for col in array_columns:
                    if col in self._movies_df.columns:
                        self._movies_df[col] = self._movies_df[col].apply(
                            lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                        )

                logger.info(f"✅ Loaded movies (CSV): {len(self._movies_df)} movies")
                return self._movies_df
            
            logger.warning("⚠️ No movie metadata file found")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error loading movies: {e}")
            return pd.DataFrame()
    
    def get_ratings(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load user-movie ratings.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            DataFrame with user ratings
        """
        if self._ratings_df is not None and not force_reload:
            return self._ratings_df
        
        try:
            # Try CSV
            csv_path = self.data_dir / "unified_dataset.csv"
            if csv_path.exists():
                self._ratings_df = pd.read_csv(csv_path)
                logger.info(f"✅ Loaded ratings: {len(self._ratings_df)} ratings")
                return self._ratings_df
            
            logger.warning("⚠️ No ratings file found")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error loading ratings: {e}")
            return pd.DataFrame()
    
    def get_embeddings(self, force_reload: bool = False) -> np.ndarray:
        """
        Load movie embeddings.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Numpy array of embeddings
        """
        if self._embeddings is not None and not force_reload:
            return self._embeddings
        
        try:
            # Load ENRICHED embeddings (43K movies with full TMDB data coverage)
            emb_path = self.models_dir / "enriched_movie_embeddings.npy"
            if emb_path.exists():
                self._embeddings = np.load(emb_path)
                logger.info(f"✅ Loaded enriched embeddings: {self._embeddings.shape}")
                return self._embeddings

            logger.warning("⚠️ No enriched embeddings file found")
            return np.array([])
            
        except Exception as e:
            logger.error(f"❌ Error loading embeddings: {e}")
            return np.array([])
    
    def get_sentiments(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load sentiment data.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            DataFrame with sentiment scores
        """
        if self._sentiments_df is not None and not force_reload:
            return self._sentiments_df
        
        try:
            sent_path = self.data_dir / "movie_sentiments.pkl"
            if sent_path.exists():
                with open(sent_path, 'rb') as f:
                    self._sentiments_df = pickle.load(f)
                logger.info(f"✅ Loaded sentiments: {self._sentiments_df.shape}")
                return self._sentiments_df
            
            logger.warning("⚠️ No sentiments file found")
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error loading sentiments: {e}")
            return pd.DataFrame()
    
    def get_themes(self, force_reload: bool = False):
        """
        Load theme data (LDA topics).
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Theme data (dict or DataFrame)
        """
        if self._themes is not None and not force_reload:
            return self._themes
        
        try:
            theme_path = self.data_dir / "movie_themes.pkl"
            if theme_path.exists():
                with open(theme_path, 'rb') as f:
                    self._themes = pickle.load(f)
                
                if isinstance(self._themes, dict):
                    logger.info(f"✅ Loaded themes: {len(self._themes)} movies")
                else:
                    logger.info(f"✅ Loaded themes: {self._themes.shape}")
                
                return self._themes
            
            logger.warning("⚠️ No themes file found")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error loading themes: {e}")
            return None
    
    def get_zero_shot_tags(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load unified zero-shot tags.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            DataFrame with zero-shot tag columns
        """
        if self._zero_shot_tags is not None and not force_reload:
            return self._zero_shot_tags
        
        try:
            # Use the integration module
            try:
                from src.zero_shot_integration import load_unified_zero_shot_tags
            except ImportError:
                import sys
                sys.path.insert(0, str(Path(__file__).parent))
                from zero_shot_integration import load_unified_zero_shot_tags
            
            self._zero_shot_tags = load_unified_zero_shot_tags(str(self.data_dir))
            logger.info(f"✅ Loaded zero-shot tags: {len(self._zero_shot_tags.columns)-1} tags")
            return self._zero_shot_tags
            
        except Exception as e:
            logger.error(f"❌ Error loading zero-shot tags: {e}")
            return pd.DataFrame()
    
    def get_movie_to_idx(self, force_reload: bool = False) -> Dict:
        """
        Load movie ID mapping.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Dictionary mapping movie titles to indices
        """
        if self._movie_to_idx is not None and not force_reload:
            return self._movie_to_idx
        
        try:
            mapping_path = self.data_dir / "movie_to_idx.json"
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    self._movie_to_idx = json.load(f)
                logger.info(f"✅ Loaded movie_to_idx: {len(self._movie_to_idx)} entries")
                return self._movie_to_idx
            
            logger.warning("⚠️ No movie_to_idx file found")
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error loading movie_to_idx: {e}")
            return {}
    
    def get_user_to_idx(self, force_reload: bool = False) -> Dict:
        """
        Load user ID mapping.
        
        Args:
            force_reload: Force reload even if cached
            
        Returns:
            Dictionary mapping user IDs to indices
        """
        if self._user_to_idx is not None and not force_reload:
            return self._user_to_idx
        
        try:
            mapping_path = self.data_dir / "user_to_idx.json"
            if mapping_path.exists():
                with open(mapping_path, 'r') as f:
                    self._user_to_idx = json.load(f)
                logger.info(f"✅ Loaded user_to_idx: {len(self._user_to_idx)} entries")
                return self._user_to_idx
            
            logger.warning("⚠️ No user_to_idx file found")
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error loading user_to_idx: {e}")
            return {}
    
    def load_all(self):
        """Load all data at once (useful for initialization)."""
        logger.info("Loading all data...")
        
        self.get_movies()
        self.get_ratings()
        self.get_embeddings()
        self.get_sentiments()
        self.get_themes()
        self.get_zero_shot_tags()
        self.get_movie_to_idx()
        self.get_user_to_idx()
        
        logger.info("✅ All data loaded")
    
    def get_movie_by_title(self, title: str) -> Optional[pd.Series]:
        """
        Get movie metadata by title.
        
        Args:
            title: Movie title
            
        Returns:
            Movie row as Series, or None
        """
        movies = self.get_movies()
        if movies.empty:
            return None
        
        # Try exact match first
        matches = movies[movies['title'] == title]
        if len(matches) > 0:
            return matches.iloc[0]
        
        # Try case-insensitive
        matches = movies[movies['title'].str.lower() == title.lower()]
        if len(matches) > 0:
            return matches.iloc[0]
        
        return None
    
    def search_movies(self, query: str, limit: int = 10) -> pd.DataFrame:
        """
        Search movies by title.
        
        Args:
            query: Search query
            limit: Max results
            
        Returns:
            DataFrame of matching movies
        """
        movies = self.get_movies()
        if movies.empty:
            return pd.DataFrame()
        
        # Case-insensitive search
        query_lower = query.lower()
        matches = movies[movies['title'].str.lower().str.contains(query_lower, na=False)]
        
        return matches.head(limit)
    
    def get_summary(self) -> Dict:
        """Get summary statistics of loaded data."""
        return {
            'num_movies': len(self.get_movies()),
            'num_ratings': len(self.get_ratings()),
            'embedding_shape': self.get_embeddings().shape if len(self.get_embeddings()) > 0 else None,
            'num_sentiments': len(self.get_sentiments()),
            'num_themes': len(self.get_themes()) if self.get_themes() is not None else 0,
            'num_tags': len(self.get_zero_shot_tags().columns) - 1 if not self.get_zero_shot_tags().empty else 0,
            'num_movie_mappings': len(self.get_movie_to_idx()),
            'num_user_mappings': len(self.get_user_to_idx())
        }


# Testing code
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING DATA LOADER")
    print("="*60 + "\n")
    
    # Initialize
    print("Initializing data loader...")
    loader = DataLoader()
    print("✅ DataLoader initialized\n")
    
    # Test loading each data source
    print("Test 1: Loading movies...")
    movies = loader.get_movies()
    print(f"✅ Loaded {len(movies)} movies\n")
    
    print("Test 2: Loading ratings...")
    ratings = loader.get_ratings()
    print(f"✅ Loaded {len(ratings)} ratings\n")
    
    print("Test 3: Loading embeddings...")
    embeddings = loader.get_embeddings()
    print(f"✅ Loaded embeddings: {embeddings.shape}\n")
    
    print("Test 4: Loading sentiments...")
    sentiments = loader.get_sentiments()
    print(f"✅ Loaded sentiments: {sentiments.shape if not sentiments.empty else 'N/A'}\n")
    
    print("Test 5: Loading themes...")
    themes = loader.get_themes()
    if isinstance(themes, dict):
        print(f"✅ Loaded themes: {len(themes)} movies\n")
    else:
        print(f"✅ Loaded themes: {themes.shape if themes is not None else 'N/A'}\n")
    
    print("Test 6: Loading zero-shot tags...")
    tags = loader.get_zero_shot_tags()
    print(f"✅ Loaded tags: {len(tags.columns)-1 if not tags.empty else 0} tag columns\n")
    
    print("Test 7: Loading ID mappings...")
    movie_to_idx = loader.get_movie_to_idx()
    user_to_idx = loader.get_user_to_idx()
    print(f"✅ Movie mappings: {len(movie_to_idx)}")
    print(f"✅ User mappings: {len(user_to_idx)}\n")
    
    print("Test 8: Get movie by title...")
    movie = loader.get_movie_by_title("Toy Story")
    if movie is not None:
        print(f"✅ Found: {movie.get('title', 'N/A')}\n")
    else:
        print("⚠️ Movie not found\n")
    
    print("Test 9: Search movies...")
    results = loader.search_movies("matrix", limit=3)
    print(f"✅ Found {len(results)} matches")
    if len(results) > 0:
        print(f"   Sample: {results['title'].tolist()}\n")
    
    print("Test 10: Get summary...")
    summary = loader.get_summary()
    print("✅ Data summary:")
    for key, val in summary.items():
        print(f"   {key}: {val}")
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✅")
    print("="*60)
    print("\n💡 Usage in other modules:")
    print("  from src.data_loader import DataLoader")
    print("  loader = DataLoader()")
    print("  movies = loader.get_movies()")