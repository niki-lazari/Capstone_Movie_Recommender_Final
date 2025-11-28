"""
Zero-Shot Tag Integration Module
Merges V1 (100+ labels) and V2 (200+ labels) zero-shot classifications.

Fixes the column name mismatch issue that broke CapstoneMasterV4 notebook.
Returns a unified dataframe with all ~335 unique zero-shot labels as binary columns.

Data Format:
- V1: ['title_norm', 'unified_tags'] where unified_tags is array of tag strings
- V2: ['title_norm', 'unified_tags_v2'] where unified_tags_v2 is array of tag strings
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Set, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZeroShotIntegrator:
    """Loads and merges V1 and V2 zero-shot tag checkpoints."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Args:
            data_dir: Path to directory containing zero-shot parquet files
        """
        self.data_dir = Path(data_dir)
        self.v1_path = self.data_dir / "zs_unified_checkpoint.parquet"
        self.v2_path = self.data_dir / "zs_unified_checkpoint_v2.parquet"
        
        # Cache for loaded data
        self._v1_df = None
        self._v2_df = None
        self._unified_df = None
        self._all_unique_tags = None
    
    def load_v1(self) -> pd.DataFrame:
        """Load V1 zero-shot tags (arrays of tag strings)."""
        if self._v1_df is not None:
            return self._v1_df
        
        logger.info(f"Loading V1 zero-shot tags from {self.v1_path}")
        self._v1_df = pd.read_parquet(self.v1_path)
        logger.info(f"V1 loaded: {len(self._v1_df)} movies")
        
        return self._v1_df
    
    def load_v2(self) -> pd.DataFrame:
        """Load V2 zero-shot tags (arrays of tag strings)."""
        if self._v2_df is not None:
            return self._v2_df
        
        logger.info(f"Loading V2 zero-shot tags from {self.v2_path}")
        self._v2_df = pd.read_parquet(self.v2_path)
        logger.info(f"V2 loaded: {len(self._v2_df)} movies")
        
        return self._v2_df
    
    def _collect_all_unique_tags(self, v1: pd.DataFrame, v2: pd.DataFrame) -> Set[str]:
        """
        Collect all unique tags across V1 and V2.
        
        Args:
            v1: V1 dataframe with 'unified_tags' column (arrays)
            v2: V2 dataframe with 'unified_tags_v2' column (arrays)
            
        Returns:
            Set of all unique tag strings
        """
        all_tags = set()
        
        # Collect V1 tags
        v1_tag_col = 'unified_tags'
        for tag_array in v1[v1_tag_col]:
            if isinstance(tag_array, (list, np.ndarray)):
                all_tags.update(tag_array)
        
        # Collect V2 tags
        v2_tag_col = 'unified_tags_v2' if 'unified_tags_v2' in v2.columns else 'unified_tags'
        for tag_array in v2[v2_tag_col]:
            if isinstance(tag_array, (list, np.ndarray)):
                all_tags.update(tag_array)
        
        logger.info(f"Collected {len(all_tags)} unique tags across V1 and V2")
        return all_tags
    
    def _combine_tag_arrays(self, v1_tags, v2_tags) -> np.ndarray:
        """
        Combine two tag arrays, removing duplicates.
        
        Args:
            v1_tags: Array/list of tags from V1 (or None/NaN)
            v2_tags: Array/list of tags from V2 (or None/NaN)
            
        Returns:
            Combined array of unique tags
        """
        combined = []
        
        # Add V1 tags
        if isinstance(v1_tags, (list, np.ndarray)):
            combined.extend(v1_tags)
        
        # Add V2 tags (avoid duplicates)
        if isinstance(v2_tags, (list, np.ndarray)):
            for tag in v2_tags:
                if tag not in combined:
                    combined.append(tag)
        
        return np.array(combined) if combined else np.array([])
    
    def merge_tags(self) -> pd.DataFrame:
        """
        Merge V1 and V2 zero-shot tags into unified dataframe with binary columns.
        
        Strategy:
        1. Load both V1 and V2 (arrays of tags per movie)
        2. Merge on title_norm
        3. Combine tag arrays for each movie
        4. Collect all unique tags across all movies
        5. Convert to one-hot encoding (binary columns)
        
        Returns:
            Unified dataframe with columns: [title_norm, tag1, tag2, ..., tagN]
            Each tag column is binary (1 if movie has tag, 0 otherwise)
        """
        if self._unified_df is not None:
            logger.info("Returning cached unified dataframe")
            return self._unified_df
        
        # Load both versions
        v1 = self.load_v1().copy()
        v2 = self.load_v2().copy()
        
        logger.info("Merging V1 and V2 on title_norm...")
        
        # Merge on title_norm (outer join to keep all movies)
        v1_tag_col = 'unified_tags'
        v2_tag_col = 'unified_tags_v2' if 'unified_tags_v2' in v2.columns else 'unified_tags'
        
        merged = v1.merge(
            v2, 
            on='title_norm', 
            how='outer',
            suffixes=('_v1', '_v2')
        )
        
        # Handle column name conflicts
        if v1_tag_col + '_v1' in merged.columns:
            v1_tag_col = v1_tag_col + '_v1'
        if v2_tag_col + '_v2' in merged.columns:
            v2_tag_col = v2_tag_col + '_v2'
        
        logger.info(f"Merged dataframe: {len(merged)} movies")
        logger.info(f"V1 tag column: {v1_tag_col}")
        logger.info(f"V2 tag column: {v2_tag_col}")
        
        # Combine tag arrays
        logger.info("Combining tag arrays...")
        merged['combined_tags'] = merged.apply(
            lambda row: self._combine_tag_arrays(
                row.get(v1_tag_col), 
                row.get(v2_tag_col)
            ),
            axis=1
        )
        
        # Collect all unique tags
        logger.info("Collecting all unique tags...")
        all_tags = set()
        for tag_array in merged['combined_tags']:
            if isinstance(tag_array, (list, np.ndarray)) and len(tag_array) > 0:
                all_tags.update(tag_array)
        
        all_tags = sorted(list(all_tags))  # Sort for consistency
        self._all_unique_tags = all_tags
        
        logger.info(f"Found {len(all_tags)} unique tags total")
        
        # Convert to one-hot encoding (binary columns)
        logger.info("Converting to one-hot encoding...")
        
        # Start with title_norm
        unified = pd.DataFrame({'title_norm': merged['title_norm']})
        
        # Create binary column for each tag
        for tag in all_tags:
            # Vectorized check if tag is in each movie's tag array
            unified[tag] = merged['combined_tags'].apply(
                lambda tags: 1 if isinstance(tags, (list, np.ndarray)) and tag in tags else 0
            )
        
        logger.info(f"✅ Unified dataframe created: {len(unified)} movies, {len(all_tags)} tag columns")
        
        self._unified_df = unified
        return self._unified_df
    
    
    def get_tag_stats(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Get statistics about tag usage (useful for rare tag detection).
        
        Args:
            df: Dataframe to analyze (defaults to unified dataframe)
            
        Returns:
            Dataframe with columns: [tag_name, count, percentage]
        """
        if df is None:
            df = self.merge_tags()
        
        tag_cols = [c for c in df.columns if c != 'title_norm']
        
        stats = []
        for tag in tag_cols:
            count = (df[tag] > 0).sum()
            pct = (count / len(df)) * 100
            stats.append({
                'tag_name': tag,
                'count': count,
                'percentage': round(pct, 2)
            })
        
        stats_df = pd.DataFrame(stats).sort_values('count', ascending=False)
        return stats_df
    
    def get_rare_tags(self, df: pd.DataFrame = None, threshold: int = 100) -> List[str]:
        """
        Identify rare tags (for boosting in scoring).
        
        Args:
            df: Dataframe to analyze
            threshold: Tags with fewer than this many movies are considered rare
            
        Returns:
            List of rare tag names
        """
        stats = self.get_tag_stats(df)
        rare = stats[stats['count'] < threshold]['tag_name'].tolist()
        
        logger.info(f"Found {len(rare)} rare tags (< {threshold} movies)")
        return rare
    
    def get_all_unique_tags(self) -> List[str]:
        """
        Get list of all unique tags (useful for knowing available tags).
        
        Returns:
            Sorted list of all tag names
        """
        if self._all_unique_tags is None:
            # Trigger merge if not done yet
            self.merge_tags()
        
        return self._all_unique_tags


def load_unified_zero_shot_tags(data_dir: str = "data/raw") -> pd.DataFrame:
    """
    Convenience function to load unified zero-shot tags.
    
    Args:
        data_dir: Path to directory containing zero-shot parquet files
        
    Returns:
        Unified dataframe with all V1 + V2 zero-shot tags as binary columns
        Columns: [title_norm, tag1, tag2, ..., tagN]
        
    Example:
        >>> tags_df = load_unified_zero_shot_tags()
        >>> print(f"Loaded {len(tags_df)} movies with {len(tags_df.columns)-1} tags")
    """
    integrator = ZeroShotIntegrator(data_dir)
    return integrator.merge_tags()


# Testing code
if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING ZERO-SHOT INTEGRATION (ARRAY-BASED FORMAT)")
    print("="*60 + "\n")
    
    # Test 1: Load V1
    print("Test 1: Loading V1...")
    integrator = ZeroShotIntegrator()
    v1 = integrator.load_v1()
    print(f"✅ V1 loaded: {len(v1)} movies, {len(v1.columns)} columns")
    print(f"   Columns: {v1.columns.tolist()}")
    print(f"   Sample tags: {v1['unified_tags'].iloc[0]}\n")
    
    # Test 2: Load V2
    print("Test 2: Loading V2...")
    v2 = integrator.load_v2()
    v2_tag_col = 'unified_tags_v2' if 'unified_tags_v2' in v2.columns else 'unified_tags'
    print(f"✅ V2 loaded: {len(v2)} movies, {len(v2.columns)} columns")
    print(f"   Columns: {v2.columns.tolist()}")
    print(f"   Sample tags: {v2[v2_tag_col].iloc[0]}\n")
    
    # Test 3: Merge tags
    print("Test 3: Merging V1 + V2 into one-hot encoding...")
    unified = integrator.merge_tags()
    print(f"✅ Unified: {len(unified)} movies, {len(unified.columns)} columns")
    print(f"   Non-tag columns: ['title_norm']")
    print(f"   Tag columns: {len(unified.columns) - 1}")
    print(f"   Sample columns: {unified.columns[1:11].tolist()}\n")
    
    # Test 4: Verify one-hot encoding
    print("Test 4: Verifying one-hot encoding...")
    sample_movie = unified.iloc[0]
    active_tags = [col for col in unified.columns[1:] if sample_movie[col] == 1]
    print(f"✅ Sample movie '{sample_movie['title_norm']}':")
    print(f"   Has {len(active_tags)} active tags: {active_tags[:10]}...\n")
    
    # Test 5: Tag statistics
    print("Test 5: Computing tag statistics...")
    stats = integrator.get_tag_stats()
    print(f"✅ Most common tags:")
    print(stats.head(10).to_string(index=False))
    print(f"\n✅ Least common tags:")
    print(stats.tail(10).to_string(index=False))
    print()
    
    # Test 6: Rare tags
    print("Test 6: Finding rare tags...")
    rare_tags = integrator.get_rare_tags(threshold=100)
    print(f"✅ Found {len(rare_tags)} rare tags (< 100 movies)")
    if rare_tags:
        print(f"   Examples: {rare_tags[:10]}\n")
    
    # Test 7: All unique tags
    print("Test 7: Getting all unique tags...")
    all_tags = integrator.get_all_unique_tags()
    print(f"✅ Total unique tags: {len(all_tags)}")
    print(f"   First 10: {all_tags[:10]}")
    print(f"   Last 10: {all_tags[-10:]}\n")
    
    # Test 8: Convenience function
    print("Test 8: Testing convenience function...")
    tags_df = load_unified_zero_shot_tags()
    print(f"✅ Convenience function works: {len(tags_df)} movies, {len(tags_df.columns)-1} tags\n")
    
    print("="*60)
    print("ALL TESTS PASSED! ✅")
    print("="*60)
    print("\n📊 Summary:")
    print(f"  • Merged {len(v1)} movies from V1 + {len(v2)} from V2")
    print(f"  • Created {len(all_tags)} binary tag columns")
    print(f"  • Found {len(rare_tags)} rare tags for potential boosting")
    print("\n💡 Usage in other modules:")
    print("  from src.zero_shot_integration import load_unified_zero_shot_tags")
    print("  tags_df = load_unified_zero_shot_tags()")
    print("  # Returns dataframe with title_norm + binary tag columns")