"""
Semantic Tag Expansion System
Automatically expands query terms to include semantically similar zero-shot tags.

Strategy:
1. Manual mappings for high-priority concepts (quality control)
2. Embedding-based similarity for automatic expansion (broad coverage)
3. Integration with query parser for all queries
"""

import numpy as np
from typing import List, Dict, Set, Optional
from sklearn.metrics.pairwise import cosine_similarity
import logging
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class SemanticTagExpander:
    """Expands query terms to semantically similar zero-shot tags."""

    # Manual mappings for high-priority concepts (curated for quality)
    MANUAL_MAPPINGS = {
        # Historical concepts
        'historical': ['historical', 'historical epic', 'historical fiction', 'period drama',
                      'ancient rome', 'ancient greece', 'medieval', 'war film', 'civil war',
                      'vietnam war', 'world war i', 'world war ii', 'samurai', 'epic scope',
                      'military', 'biography', 'based on true story'],

        'ancient': ['ancient rome', 'ancient greece', 'historical epic', 'medieval',
                   'period drama', 'epic scope'],

        'war': ['war film', 'military', 'vietnam war', 'world war i', 'world war ii',
               'civil war', 'combat', 'fighting', 'battlefield'],

        # Revenge & conflict concepts
        'revenge': ['revenge', 'justice vs revenge', 'fighting', 'combat',
                   'identity crisis', 'overcoming adversity', 'loss and grief',
                   'intense', 'dark', 'bittersweet', 'vendetta'],

        'justice': ['justice vs revenge', 'revenge', 'vigilante', 'crime',
                   'thriller', 'detective', 'investigation'],

        # Sci-fi concepts
        'sci-fi': ['sci-fi', 'science fiction', 'cyberpunk', 'dystopian',
                  'post-apocalyptic', 'time travel', 'space opera', 'alien',
                  'robot ai', 'futuristic', 'parallel universe'],

        'science': ['sci-fi', 'science fiction', 'robot ai', 'futuristic',
                   'scientific', 'experimental'],

        # Psychological concepts
        'psychological': ['psychological', 'cerebral', 'mind-bending', 'mystery',
                         'identity crisis', 'mental illness', 'paranoia', 'surreal'],

        'thriller': ['thriller', 'suspenseful', 'mystery', 'intense', 'crime',
                    'noir', 'detective', 'investigation'],

        # Emotional concepts
        'dark': ['dark', 'noir', 'gritty', 'bleak', 'intense', 'disturbing',
                'bittersweet', 'tragic'],

        'inspiring': ['inspiring', 'uplifting', 'feel-good', 'heartwarming',
                     'overcoming adversity', 'triumph', 'hopeful'],

        'comedy': ['comedy', 'funny', 'hilarious', 'absurdist', 'parody',
                  'satire', 'slapstick', 'dark comedy'],

        # Character-focused
        'protagonist': ['male protagonist', 'female protagonist', 'strong female lead',
                       'hero', 'anti-hero', 'character-driven'],

        'female lead': ['female protagonist', 'strong female lead', 'feminist',
                       'girl power', 'matriarch'],

        # Genre mixing
        'horror': ['horror', 'scary', 'disturbing', 'supernatural', 'monster',
                  'slasher', 'gothic', 'paranormal'],

        'action': ['action', 'fighting', 'combat', 'martial arts', 'chase',
                  'explosive', 'adrenaline', 'intense'],

        'romance': ['romance', 'romantic', 'love story', 'relationship',
                   'wedding', 'courtship', 'passion'],

        'drama': ['drama', 'character-driven', 'emotional', 'intense',
                 'family drama', 'coming-of-age', 'life-changing'],
    }

    # TMDB Keyword mappings for concepts that need expanded keyword search
    # These map query concepts to actual TMDB keywords (not zero-shot tags)
    KEYWORD_MAPPINGS = {
        # Female lead/protagonist concepts - maps to actual TMDB keywords
        # IMPORTANT: Include female→woman/girl variants for proper expansion
        'strong female lead': [
            'female protagonist', 'female lead', 'female hero', 'strong woman',
            'independent woman', 'heroine', 'woman protagonist', 'feminist',
            'woman power', 'girl power', 'powerful woman', 'tough woman',
            'woman fighter', 'female detective', 'female spy', 'female assassin',
            'female cop', 'female lawyer', 'female journalist', 'female scientist',
            'career woman', 'businesswoman', 'woman in charge', 'fearless woman',
            # "Dark" female lead archetypes (femme fatale, etc.)
            'femme fatale', 'female psychopath', 'female stalker', 'female killer',
            'manipulative woman', 'dangerous woman', 'woman antagonist',
            # woman/girl variants for broader matching
            'young woman', 'teenage girl', 'woman', 'girl'
        ],
        'female protagonist': [
            'female protagonist', 'female lead', 'female hero', 'heroine',
            'woman protagonist', 'strong woman', 'independent woman',
            'femme fatale', 'female detective', 'female cop',
            'young woman', 'woman', 'girl'
        ],
        'female lead': [
            'female lead', 'female protagonist', 'female hero', 'heroine',
            'strong woman', 'independent woman', 'woman protagonist',
            'femme fatale', 'female psychopath',
            'young woman', 'woman', 'girl'
        ],
        # Generic 'female' expansion - for queries like "female..." anything
        'female': [
            'female', 'woman', 'girl', 'mother', 'daughter', 'wife', 'sister',
            'female protagonist', 'female lead', 'young woman', 'teenage girl',
            'heroine', 'femme fatale', 'strong woman', 'independent woman'
        ],

        # Male lead concepts
        'strong male lead': [
            'male protagonist', 'male lead', 'male hero', 'strong man',
            'action hero', 'tough guy', 'hero', 'protagonist'
        ],
        'male': [
            'male', 'man', 'boy', 'father', 'son', 'husband', 'brother',
            'male protagonist', 'male lead', 'young man', 'teenage boy', 'hero'
        ],

        # Psychological thriller concepts (common query pattern)
        'psychological thriller': [
            'psychological thriller', 'psychological horror', 'psychological drama',
            'psychological suspense', 'psychopath', 'psychopathic', 'mind games',
            'manipulation', 'obsession', 'stalker', 'stalking', 'paranoia',
            'mental illness', 'twisted', 'disturbed', 'deranged'
        ],
        'psychological': [
            'psychological thriller', 'psychological horror', 'psychological drama',
            'psychological suspense', 'psychopath', 'psychopathic', 'psycho',
            'sociopath', 'sociopathic', 'mind games', 'manipulation', 'obsession',
            'paranoia', 'mental illness', 'mentally ill', 'disturbed', 'deranged',
            'twisted', 'neurotic', 'unstable', 'insane', 'insanity', 'madness'
        ],

        # Dark thriller concepts
        'dark thriller': [
            'psychological thriller', 'erotic thriller', 'neo-noir', 'noir',
            'femme fatale', 'psychopath', 'serial killer', 'murder', 'obsession',
            'stalker', 'revenge', 'dark secret', 'twisted', 'disturbed'
        ],
        'erotic thriller': [
            'erotic thriller', 'erotic', 'femme fatale', 'sexual obsession',
            'seduction', 'manipulation', 'neo-noir', 'dangerous woman'
        ],

        # Award-related expansions (universal for any "best X" query)
        'award': [
            'academy award', 'oscar', 'golden globe', 'bafta', 'cannes',
            'award-winning', 'award winner', 'best picture', 'best actor',
            'best actress', 'best director', 'critically acclaimed'
        ],
        'oscar': [
            'academy award', 'oscar', 'oscar winner', 'oscar winning',
            'academy award winner', 'best picture', 'best actor', 'best actress'
        ],
        'best actress': [
            'best actress', 'academy award', 'oscar', 'golden globe',
            'leading actress', 'female lead', 'female protagonist'
        ],
        'best actor': [
            'best actor', 'academy award', 'oscar', 'golden globe',
            'leading actor', 'male lead', 'male protagonist'
        ],

        # Dark/noir concepts - maps to actual TMDB keywords for dark-themed movies
        'dark': [
            'dark', 'darkness', 'dark secret', 'dark past', 'dark comedy',
            'dark hero', 'dark heroine', 'noir', 'neo-noir', 'film noir',
            'disturbing', 'twisted', 'sinister', 'grim', 'bleak',
            'psychological thriller', 'psychopath', 'serial killer',
            'nightmare', 'insanity', 'paranoia', 'obsession', 'madness',
            'tragedy', 'tragic', 'murder', 'death', 'dying', 'dying and death',
            'sadism', 'sadistic', 'macabre', 'brutal', 'brutality', 'violence',
            'violent', 'intense', 'suspense', 'suspenseful', 'menacing', 'ominous',
            'foreboding', 'creepy', 'eerie', 'haunting', 'chilling', 'morbid',
            'sinister', 'malevolent', 'evil', 'wicked', 'deadly', 'lethal'
        ],

        # Lead/protagonist concepts
        'lead': [
            'lead', 'protagonist', 'female lead', 'male lead',
            'female protagonist', 'male protagonist', 'leading lady',
            'leading man', 'hero', 'heroine', 'main character'
        ],
    }

    def __init__(self, all_tags: List[str], all_keywords: Optional[List[str]] = None, models_dir: str = "models"):
        """
        Initialize with list of all available zero-shot tags and TMDB keywords.

        Args:
            all_tags: List of all 322 zero-shot tags
            all_keywords: Optional list of all TMDB keywords (22k+ keywords)
            models_dir: Directory containing tag/keyword embeddings (default: "models")
        """
        self.all_tags = set(all_tags)
        self.all_tags_list = sorted(list(all_tags))  # Keep ordered list for indexing
        self.models_dir = Path(models_dir)
        self.tag_embeddings = None
        self.tag_to_idx = {tag: i for i, tag in enumerate(self.all_tags_list)}

        # Keyword support
        if all_keywords:
            self.all_keywords = set(all_keywords)
            self.all_keywords_list = sorted(list(all_keywords))
            self.keyword_embeddings = None
            self.keyword_to_idx = {kw: i for i, kw in enumerate(self.all_keywords_list)}
        else:
            self.all_keywords = None
            self.all_keywords_list = None
            self.keyword_embeddings = None
            self.keyword_to_idx = None

        self.bert_model = None

        # OPTIMIZATION: Cache for query term embeddings to avoid recomputation
        # Makes repeat queries with same terms instant (e.g., "action movies" then "action thrillers")
        self._embedding_cache = {}

        logger.info(f"SemanticTagExpander initialized with {len(self.all_tags)} tags" +
                   (f" and {len(self.all_keywords)} keywords" if all_keywords else ""))

    def _load_tag_embeddings(self):
        """Load pre-computed tag embeddings from disk."""
        if self.tag_embeddings is not None:
            return  # Already loaded

        embeddings_path = self.models_dir / "tag_embeddings.npy"
        tag_list_path = self.models_dir / "tag_list.pkl"

        if not embeddings_path.exists():
            logger.warning(f"Tag embeddings not found at {embeddings_path}")
            logger.warning("Run generate_tag_embeddings.py first!")
            return

        # Load embeddings and tag list
        self.tag_embeddings = np.load(embeddings_path)
        with open(tag_list_path, 'rb') as f:
            saved_tag_list = pickle.load(f)

        # Verify tag list matches
        if saved_tag_list != self.all_tags_list:
            logger.warning("Saved tag list doesn't match current tags - regenerate embeddings!")
            self.tag_embeddings = None
            return

        logger.info(f"Loaded tag embeddings: {self.tag_embeddings.shape}")

    def _load_keyword_embeddings(self):
        """Load pre-computed keyword embeddings from disk."""
        if self.keyword_embeddings is not None:
            return  # Already loaded

        if not self.all_keywords:
            logger.warning("No keywords available - initialize with all_keywords parameter")
            return

        embeddings_path = self.models_dir / "keyword_embeddings.npy"
        keyword_list_path = self.models_dir / "keyword_list.pkl"

        if not embeddings_path.exists():
            logger.warning(f"Keyword embeddings not found at {embeddings_path}")
            logger.warning("Run generate_keyword_embeddings.py first!")
            return

        # Load embeddings and keyword list
        self.keyword_embeddings = np.load(embeddings_path)
        with open(keyword_list_path, 'rb') as f:
            saved_keyword_list = pickle.load(f)

        # Verify keyword list matches
        if saved_keyword_list != self.all_keywords_list:
            logger.warning("Saved keyword list doesn't match current keywords - regenerate embeddings!")
            self.keyword_embeddings = None
            return

        logger.info(f"Loaded keyword embeddings: {self.keyword_embeddings.shape}")

    def _load_bert_model(self):
        """Load BERT model for encoding query terms on-the-fly."""
        if self.bert_model is not None:
            return  # Already loaded

        try:
            from sentence_transformers import SentenceTransformer
            self.bert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Loaded SentenceTransformer model for query encoding")
        except Exception as e:
            logger.warning(f"Failed to load BERT model: {e}")

    def _find_similar_tags(self, term: str, top_k: int = 10, min_similarity: float = 0.3) -> List[tuple]:
        """
        Find tags most similar to the query term using BERT embeddings.

        Args:
            term: Query term to expand
            top_k: Number of similar tags to return
            min_similarity: Minimum cosine similarity threshold (0-1)

        Returns:
            List of (tag, similarity_score) tuples, sorted by similarity
        """
        # Load embeddings and model if needed
        self._load_tag_embeddings()
        self._load_bert_model()

        if self.tag_embeddings is None or self.bert_model is None:
            logger.warning("Embeddings or BERT model not available - falling back to fuzzy matching")
            return []

        # OPTIMIZATION: Check cache first before computing embedding
        cache_key = f"tag_{term}"
        if cache_key in self._embedding_cache:
            term_embedding = self._embedding_cache[cache_key]
        else:
            # Encode the query term
            term_embedding = self.bert_model.encode(
                [term],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            # Cache for future queries
            self._embedding_cache[cache_key] = term_embedding

        # Compute cosine similarities to all tags
        similarities = cosine_similarity(term_embedding, self.tag_embeddings)[0]

        # Get top-k similar tags above threshold
        similar_indices = np.argsort(similarities)[::-1][:top_k]
        results = []

        for idx in similar_indices:
            similarity = similarities[idx]
            if similarity >= min_similarity:
                tag = self.all_tags_list[idx]
                results.append((tag, float(similarity)))

        return results

    def _find_similar_keywords(self, term: str, top_k: int = 10, min_similarity: float = 0.3) -> List[tuple]:
        """
        Find TMDB keywords most similar to the query term using BERT embeddings.

        Args:
            term: Query term to expand
            top_k: Number of similar keywords to return
            min_similarity: Minimum cosine similarity threshold (0-1)

        Returns:
            List of (keyword, similarity_score) tuples, sorted by similarity
        """
        # Load embeddings and model if needed
        self._load_keyword_embeddings()
        self._load_bert_model()

        if self.keyword_embeddings is None or self.bert_model is None:
            logger.warning("Keyword embeddings or BERT model not available")
            return []

        # OPTIMIZATION: Check cache first before computing embedding
        cache_key = f"keyword_{term}"
        if cache_key in self._embedding_cache:
            term_embedding = self._embedding_cache[cache_key]
        else:
            # Encode the query term
            term_embedding = self.bert_model.encode(
                [term],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            # Cache for future queries
            self._embedding_cache[cache_key] = term_embedding

        # Compute cosine similarities to all keywords
        similarities = cosine_similarity(term_embedding, self.keyword_embeddings)[0]

        # Get top-k similar keywords above threshold
        similar_indices = np.argsort(similarities)[::-1][:top_k]
        results = []

        for idx in similar_indices:
            similarity = similarities[idx]
            if similarity >= min_similarity:
                keyword = self.all_keywords_list[idx]
                results.append((keyword, float(similarity)))

        return results

    def expand_term(self, term: str, method: str = 'manual', top_k: int = 10, min_similarity: float = 0.3,
                   search_keywords: bool = False) -> Set[str]:
        """
        Expand a single query term to semantically similar tags or keywords.

        Args:
            term: Query term (e.g., 'historical', 'revenge', 'ancient')
            method: 'manual' (curated mappings), 'embedding' (BERT similarity), or 'hybrid'
            top_k: Number of similar items to return for embedding method
            min_similarity: Minimum cosine similarity threshold (0-1)
            search_keywords: If True, search TMDB keywords instead of zero-shot tags

        Returns:
            Set of expanded tags or keywords (including original if it exists)
        """
        term_lower = term.lower()
        expanded = set()

        # Choose which set to search
        if search_keywords:
            all_items = self.all_keywords
            all_items_list = self.all_keywords_list
            find_similar_fn = self._find_similar_keywords
            item_type = "keywords"
        else:
            all_items = self.all_tags
            all_items_list = self.all_tags_list
            find_similar_fn = self._find_similar_tags
            item_type = "tags"

        # ALWAYS do substring matching for keywords (more robust than embeddings alone)
        if search_keywords:
            substring_matches = set()

            # Handle plural/singular variations
            search_terms = [term_lower]
            if term_lower.endswith('s') and len(term_lower) > 3:
                # Also search for singular form (e.g., 'animals' -> also search 'animal')
                search_terms.append(term_lower[:-1])
            else:
                # Also search for plural form (e.g., 'animal' -> also search 'animals')
                search_terms.append(term_lower + 's')

            for item in all_items:
                # Match if ANY search term is contained in keyword OR keyword is contained in search term
                # This catches: 'animals' -> 'animal attack', 'animal horror', etc.
                for search_term in search_terms:
                    if search_term in item or item in search_term:
                        substring_matches.add(item)
                        break

            if substring_matches:
                expanded.update(substring_matches)
                logger.info(f"Substring matching: '{term}' -> {len(substring_matches)} keywords")

        # Try manual mapping first
        if method == 'manual' or method == 'hybrid':
            # Use KEYWORD_MAPPINGS for keywords, MANUAL_MAPPINGS for tags
            mappings = self.KEYWORD_MAPPINGS if search_keywords else self.MANUAL_MAPPINGS

            if term_lower in mappings:
                manual_items = mappings[term_lower]
                # Only include items that exist in our dataset
                matched_items = [item for item in manual_items if item in all_items]
                expanded.update(matched_items)
                logger.info(f"Manual keyword expansion: '{term}' -> {len(matched_items)} {item_type}")

                if method == 'manual':
                    return expanded

        # If no manual mapping or using embedding method (or adding to substring matches for keywords)
        if method == 'embedding' or (method == 'hybrid' and len(expanded) == 0):
            # Use BERT embedding similarity to find related items
            similar_items = find_similar_fn(term_lower, top_k=top_k, min_similarity=min_similarity)

            if similar_items:
                # Add similar items
                for item, similarity in similar_items:
                    expanded.add(item)

                logger.info(f"Embedding expansion: '{term}' -> {len(similar_items)} {item_type} (sim >= {min_similarity:.2f})")

                # Log top 5 for debugging
                if similar_items:
                    top_5 = similar_items[:5]
                    logger.debug(f"  Top 5: {[(item, f'{sim:.3f}') for item, sim in top_5]}")
            else:
                # Fallback to fuzzy matching if embedding fails
                logger.info(f"Embedding expansion failed for '{term}' - using fuzzy fallback")
                if term_lower in all_items:
                    expanded.add(term_lower)

                # Fuzzy matching for common variations
                for item in all_items:
                    if term_lower in item or item in term_lower:
                        expanded.add(item)

        # Always include exact match if it exists
        if term_lower in all_items:
            expanded.add(term_lower)

        return expanded

    def expand_query_terms(self, terms: List[str], method: str = 'hybrid') -> Set[str]:
        """
        Expand multiple query terms.

        Args:
            terms: List of query terms
            method: 'manual', 'embedding', or 'hybrid' (try manual first)

        Returns:
            Set of all expanded tags
        """
        all_expanded = set()

        for term in terms:
            expanded = self.expand_term(term, method=method)
            all_expanded.update(expanded)

        logger.info(f"Expanded {len(terms)} query terms -> {len(all_expanded)} tags")
        return all_expanded

    def expand_query_terms_grouped(self, terms: List[str], method: str = 'hybrid') -> Dict[str, List[str]]:
        """
        Expand multiple query terms and track which tags came from which concept.

        This is the FIX for the OR vs AND problem:
        - Query "historical revenge" should match movies with BOTH concepts
        - Not just movies with historical OR revenge

        Args:
            terms: List of query terms (e.g., ['historical', 'revenge'])
            method: 'manual', 'embedding', or 'hybrid'

        Returns:
            Dict mapping each concept to its expanded tags
            Example: {
                'historical': ['historical', 'ancient rome', 'medieval', ...],
                'revenge': ['revenge', 'fighting', 'combat', ...]
            }
        """
        concept_groups = {}

        for term in terms:
            expanded = self.expand_term(term, method=method)
            if len(expanded) > 0:
                concept_groups[term] = list(expanded)
                logger.info(f"  Concept '{term}' -> {len(expanded)} tags")

        total_tags = sum(len(tags) for tags in concept_groups.values())
        logger.info(f"Expanded {len(terms)} concepts -> {len(concept_groups)} groups with {total_tags} total tags")
        return concept_groups

    def get_expansion_stats(self, terms: List[str]) -> Dict:
        """
        Get statistics about expansion for debugging/logging.

        Args:
            terms: List of query terms

        Returns:
            Dict with expansion statistics
        """
        stats = {
            'input_terms': terms,
            'expansions': {}
        }

        for term in terms:
            expanded = self.expand_term(term, method='hybrid')
            stats['expansions'][term] = {
                'count': len(expanded),
                'tags': sorted(list(expanded))
            }

        return stats


# Convenience function
def create_expander(all_tags: List[str]) -> SemanticTagExpander:
    """Create a SemanticTagExpander instance."""
    return SemanticTagExpander(all_tags)


if __name__ == "__main__":
    # Test the expander
    print("\n" + "="*80)
    print("TESTING SEMANTIC TAG EXPANDER")
    print("="*80 + "\n")

    # Sample tags (in real use, load all 322 from zero-shot integration)
    sample_tags = [
        'historical', 'historical epic', 'historical fiction',
        'ancient rome', 'ancient greece', 'medieval',
        'revenge', 'justice vs revenge', 'fighting', 'combat',
        'war film', 'military', 'samurai',
        'sci-fi', 'time travel', 'cyberpunk',
        'thriller', 'psychological', 'cerebral',
    ]

    expander = SemanticTagExpander(sample_tags)

    # Test cases
    test_queries = [
        ['historical', 'revenge'],
        ['sci-fi', 'time travel'],
        ['psychological', 'thriller'],
        ['war', 'ancient'],
    ]

    for query_terms in test_queries:
        print(f"\nQuery terms: {query_terms}")
        print("-" * 80)

        expanded = expander.expand_query_terms(query_terms)
        print(f"Expanded to {len(expanded)} tags:")
        for tag in sorted(expanded):
            print(f"  - {tag}")

        # Show stats
        stats = expander.get_expansion_stats(query_terms)
        print(f"\nDetailed breakdown:")
        for term, info in stats['expansions'].items():
            print(f"  '{term}' -> {info['count']} tags: {info['tags']}")

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
